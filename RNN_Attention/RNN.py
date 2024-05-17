import torch
import torch.nn as nn
import torch.nn.functional as F
from CharacterEncoding import SOW_token, EOW_token, CharEncoder
import random
import numpy as np
import wandb
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
plt.switch_backend('agg')
import matplotlib.ticker as ticker


class Encoder(nn.Module):
    def __init__(self,
                 inputSize: int,
                 embeddingSize: int,
                 hiddenSize: int,
                 cellType: str,
                 n_layers: int,
                 dropoutProb: float,
                 device: str):

        super(Encoder, self).__init__()
        self.hiddenSize = hiddenSize
        self.n_layers = n_layers
        self.dropoutProb = dropoutProb
        self.cellType = cellType
        self.embedding = nn.Embedding(inputSize, embeddingSize)
        self.device = device

        #Default - LSTM
        self.rnn = nn.LSTM(input_size = embeddingSize, hidden_size = hiddenSize,  num_layers = n_layers, dropout = dropoutProb)
        if cellType == "RNN":
            self.rnn = nn.RNN(input_size = embeddingSize, hidden_size = hiddenSize,  num_layers = n_layers, dropout = dropoutProb)
        elif cellType == "GRU":
            self.rnn = nn.GRU(input_size = embeddingSize, hidden_size = hiddenSize,  num_layers = n_layers, dropout = dropoutProb)

    def forward(self, input, hidden, state):
        embeddedOutput = self.embedding(input).view(1, 1, -1)

        if(self.cellType == "LSTM"):
            output, (hidden, state) = self.rnn(embeddedOutput, (hidden, state))
        else:
            output, hidden = self.rnn(embeddedOutput, hidden)

        return output, hidden, state

    def initHidden(self):
        return torch.zeros(self.n_layers, 1, self.hiddenSize, device=self.device)

class AttentionDecoder(nn.Module):
    def __init__(self,
                 OuputSize: int,
                 embeddingSize: int,
                 hiddenSize: int,
                 cellType: str,
                 nLayers: int,
                 dropoutProb: float,
                 device:str):

        super(AttentionDecoder, self).__init__()
        self.hiddenSize = hiddenSize
        self.nLayers = nLayers
        self.dropoutProb = dropoutProb
        self.cellType = cellType
        self.device = device
        self.embedding = nn.Embedding(OuputSize, embeddingSize)

	#Add attention layer
        self.attn = nn.Linear(hiddenSize + embeddingSize, 50)
        self.attn_combine = nn.Linear(hiddenSize + embeddingSize, hiddenSize)

        self.rnn = nn.LSTM(input_size = hiddenSize, hidden_size = hiddenSize,num_layers = nLayers,dropout = dropoutProb)
        if cellType == "RNN":
            self.rnn = nn.RNN(input_size = hiddenSize, hidden_size = hiddenSize,  num_layers = nLayers, dropout = dropoutProb)
        elif cellType == "GRU":
            self.rnn = nn.GRU(input_size = hiddenSize, hidden_size = hiddenSize,  num_layers = nLayers, dropout = dropoutProb)


        self.out = nn.Linear(hiddenSize, OuputSize)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, cell, encoder_outputs):
        embeddedOutputs = self.embedding(input).view(1, 1, -1)
	
	#Calculate attention weights using current input and previous hidden state
        attentionWeights = F.softmax(self.attn(torch.cat((embeddedOutputs[0], hidden[0]), 1)), dim=1)
        attentions = torch.bmm(attentionWeights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embeddedOutputs[0], attentions[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        if(self.cellType == "LSTM"):
            output, (hidden, cell) = self.rnn(output, (hidden, cell))
        else:
            output, hidden = self.rnn(output, hidden)

        output = self.softmax(self.out(output[0]))
        return output, hidden, cell, attentionWeights

    def initHidden(self):
        return torch.zeros(self.nLayers, 1, self.hiddenSize, device=self.device)


class RNNModel:
    def __init__(self, input_charEncoding:CharEncoder, output_charEncoding:CharEncoder, data, config: dict, device: str):
        self.input_charEncoding = input_charEncoding
        self.output_charEncoding = output_charEncoding

        self.input_size = self.input_charEncoding.n_chars
        self.output_size = self.output_charEncoding.n_chars
        self.device = device

        self.train_data = []

        for index, words in data.iterrows():
            input_tensor = self.input_charEncoding.getTensorFromWord(words[0])
            output_tensor = self.output_charEncoding.getTensorFromWord(words[1])
            self.train_data.append([input_tensor, output_tensor, words[1]])

        self.encoder = Encoder(inputSize = self.input_size,
                             embeddingSize = config["embedding_size"],
                             hiddenSize = config["HiddenLayerSize"],
                             cellType = config["cellType"],
                             n_layers = config["nLayers"],
                             dropoutProb = config["dropoutProb"],
                             device=self.device).to(self.device)

        self.decoder = AttentionDecoder(OuputSize = self.output_size,
                             embeddingSize = config["embedding_size"],
                             hiddenSize = config["HiddenLayerSize"],
                             cellType = config["cellType"],
                             nLayers = config["nLayers"],
                             dropoutProb = config["dropoutProb"],
                             device=self.device).to(self.device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=config["learningRate"], weight_decay=config["weightdecay"])
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=config["learningRate"], weight_decay=config["weightdecay"])
        if(config["optimizer"] == "SGD"):
            self.encoder_optimizer = torch.optim.SGD(self.encoder.parameters(), lr = config["learningRate"], weight_decay=config["weightdecay"])
            self.decoder_optimizer = torch.optim.SGD(self.decoder.parameters(), lr = config["learningRate"], weight_decay=config["weightdecay"])



        self.criterion = nn.NLLLoss()
        self.max_length = 50

    def train_word(self, input_tensor, target_tensor):
        encoder_hidden = self.encoder.initHidden()
        encoder_state = self.encoder.initHidden()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(self.max_length, self.encoder.hiddenSize, device=self.device)

        loss = 0
        pred_word = ""

        for ei in range(input_length):
            encoder_output, encoder_hidden, encoder_state = self.encoder(input_tensor[ei], encoder_hidden, encoder_state)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOW_token]], device=self.device)
        decoder_hidden, decoder_state = encoder_hidden, encoder_state

        use_teacher_forcing = True #Set it true for training

        for i in range(target_length):
            decoder_output, decoder_hidden, decoder_state, attention= self.decoder(decoder_input, decoder_hidden, decoder_state, encoder_outputs)
            loss += self.criterion(decoder_output, target_tensor[i])

            topv, topi = decoder_output.topk(1)
            if self.output_charEncoding.index2char.__contains__(topi.item()):
                pred_word += self.output_charEncoding.index2char[topi.item()]

            if use_teacher_forcing:
                decoder_input = target_tensor[i]
            else:
                decoder_input = topi.squeeze().detach()
                if decoder_input.item() == EOW_token:
                    break


        loss_value = loss.item() / target_length

        return pred_word, loss, loss_value

    def train(self, batch_size = 32, nIterations=-1):
        total_loss = 0
        total_accuracy = 0
        batch_loss = 0
        idx = 0
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        random.shuffle(self.train_data)
        nIterations = len(self.train_data) if nIterations == -1 else nIterations

        for iter in range(1, nIterations+1):
            training_pair = self.train_data[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]
            target_word = training_pair[2]

            pred_word, loss, loss_val = self.train_word(input_tensor, target_tensor)
            batch_loss += loss
            idx += 1
            total_loss += loss_val
            if pred_word == target_word:
                total_accuracy += 1
            if idx == batch_size:
                batch_loss.backward()
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()
                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()
                idx = 0
                batch_loss = 0
        if idx!=0 :
            batch_loss.backward()
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

        avg_loss = total_loss/len(self.train_data)
        avg_accuracy = total_accuracy / len(self.train_data)
        return avg_loss, avg_accuracy

    def predict(self, word):
        with torch.no_grad():
            input_tensor = self.input_charEncoding.getTensorFromWord(word)
            input_length = input_tensor.size()[0]
            encoder_hidden = self.encoder.initHidden()
            encoder_state = self.encoder.initHidden()

            encoder_outputs = torch.zeros(self.max_length, self.encoder.hiddenSize, device=self.device)

            for ei in range(input_length):
                encoder_output, encoder_hidden, encoder_state = self.encoder(input_tensor[ei], encoder_hidden, encoder_state)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[SOW_token]], device=self.device)
            decoder_hidden, decoder_state = encoder_hidden, encoder_state

            pred_word = ""
            attentions = torch.zeros(self.max_length, self.max_length)
            i = 0

            for i in range(self.max_length):
                decoder_output, decoder_hidden, decoder_state, attention = self.decoder(decoder_input, decoder_hidden, decoder_state, encoder_outputs)
                attentions[i] = attention.data
                topv, topi = decoder_output.topk(1)

                if topi.item() == EOW_token:
                    break
                else:
                    if self.output_charEncoding.index2char.__contains__(topi.item()):
                        pred_word += self.output_charEncoding.index2char[topi.item()]

                #Inference mode, feed the prediction as next input
                decoder_input = topi.squeeze().detach()

            return pred_word, attentions[:i + 1]
    

    def plotAttention(self, input_word, output_word, attentions, index):
        input_word = list(input_word)
        output_word = list(output_word)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(attentions.cpu().numpy(), cmap='bone')
        fig.colorbar(cax)

        # Load Telugu font
        font_path = "Pothana2000.ttf"#"./Gurajada.ttf"  # Path to your Telugu font file
        telugu_font = fm.FontProperties(fname=font_path)

        # Set up axes
        ax.set_xticklabels([''] + input_word , rotation=90)
        ax.set_yticklabels([''] + output_word, fontproperties=telugu_font)

        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        image = f'Attention_{index}.png'
        plt.savefig(image)
        img = plt.imread(image)
        wandb.log({"Attention_plots": wandb.Image(img)})

    def test(self, data, plot = False):
        accuracy = 0
        pred_words = []
        for index, words in data.iterrows():
            pred_word, attentions = self.predict(words[0])
            pred_words.append(pred_word)
            if pred_word == words[1]:
                accuracy += 1
            if plot == True and index < 9:
                self.plotAttention(words[0], pred_word, attentions[:len(pred_word),:len(words[0])], index)
        return pred_words, accuracy / len(data)
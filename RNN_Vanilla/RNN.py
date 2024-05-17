import torch
import torch.nn as nn
import torch.nn.functional as F
from CharacterEncoding import SOW_token, EOW_token, CharEncoder
import random
import time
import numpy as np

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
        #Embedding layer
        embeddedOutput = self.embedding(input).view(1, 1, -1)

        #RNN cell
        if(self.cellType == "LSTM"):
            output, (hidden, state) = self.rnn(embeddedOutput, (hidden, state))
        else:
            output, hidden = self.rnn(embeddedOutput, hidden)
            
        return output, hidden, state
    
    def initHidden(self):
        return torch.zeros(self.n_layers, 1, self.hiddenSize, device=self.device)
    
class Decoder(nn.Module):
    def __init__(self,
                 OuputSize: int,
                 embeddingSize: int,
                 hiddenSize: int,
                 cellType: str,
                 n_layers: int,
                 dropoutProb: float,
                 device: str):

        super(Decoder, self).__init__()
        self.hiddenSize = hiddenSize
        self.n_layers = n_layers
        self.dropoutProb = dropoutProb
        self.cellType = cellType
        self.embedding = nn.Embedding(OuputSize, embeddingSize)
        self.device = device

        #Default - LSTM
        self.rnn = nn.LSTM(input_size = embeddingSize, hidden_size = hiddenSize,  num_layers = n_layers, dropout = dropoutProb)
        if cellType == "RNN":
            self.rnn = nn.RNN(input_size = embeddingSize, hidden_size = hiddenSize,  num_layers = n_layers, dropout = dropoutProb)
        elif cellType == "GRU":
            self.rnn = nn.GRU(input_size = embeddingSize, hidden_size = hiddenSize,  num_layers = n_layers, dropout = dropoutProb)
        
        self.out = nn.Linear(hiddenSize, OuputSize)
        #Softmax layer to get the probablilities for each character
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, state):
        #Embedding layer
        embeddedOutput = self.embedding(input).view(1, 1, -1)
        #Activation funtion
        output = F.relu(embeddedOutput)

        if(self.cellType == "LSTM"):
            output, (hidden, state) = self.rnn(output, (hidden, state))
        else:
            output, hidden = self.rnn(output, hidden)
            
        output = self.softmax(self.out(output[0]))
        return output, hidden, state
    
    def initHidden(self):
        return torch.zeros(self.n_layers, 1, self.hiddenSize, device=self.device)

class RNNModel:
    def __init__(self, input_charEncoding:CharEncoder, output_charEncoding:CharEncoder, data, config: dict, device: str):
        self.input_charEncoding = input_charEncoding
        self.output_charEncoding = output_charEncoding
        
        self.input_size = self.input_charEncoding.n_chars
        self.output_size = self.output_charEncoding.n_chars
        self.device = device

        self.train_data = []

        #From train data set get tensor representations of words
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
        
        self.decoder = Decoder(OuputSize = self.output_size,
                             embeddingSize = config["embedding_size"],
                             hiddenSize = config["HiddenLayerSize"],
                             cellType = config["cellType"],
                             n_layers = config["nLayers"],
                             dropoutProb = config["dropoutProb"],
                             device=self.device).to(self.device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=config["learningRate"])
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=config["learningRate"])
        if(config["optimizer"] == "SGD"):
            self.encoder_optimizer = torch.optim.SGD(self.encoder.parameters(), lr = config["learningRate"])
            self.decoder_optimizer = torch.optim.SGD(self.decoder.parameters(), lr = config["learningRate"])
            
            
        
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
            decoder_output, decoder_hidden, decoder_state = self.decoder(decoder_input, decoder_hidden, decoder_state)
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
        
        # loss.backward()
        # self.encoder_optimizer.step()
        # self.decoder_optimizer.step()

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

            #Forward propogation for each word
            pred_word, loss, loss_val = self.train_word(input_tensor, target_tensor)
            #Accumulate the loss, backpropogate batch wise
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
        #Used for validation and testing after training
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

            for di in range(self.max_length):
                decoder_output, decoder_hidden, decoder_state = self.decoder(decoder_input, decoder_hidden, decoder_state)
                topv, topi = decoder_output.topk(1)
                
                if topi.item() == EOW_token:
                    break
                else:
                    if self.output_charEncoding.index2char.__contains__(topi.item()):
                        pred_word += self.output_charEncoding.index2char[topi.item()]

                #Inference mode, feed the prediction as next input
                decoder_input = topi.squeeze().detach()

            return pred_word

    def test(self, data):
        #Used for validation and accuracy after training
        accuracy = 0
        for index, words in data.iterrows():
            pred_word = self.predict(words[0])
            if pred_word == words[1]:
                accuracy += 1
        return accuracy / len(data)

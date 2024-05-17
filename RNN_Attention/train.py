import torch
import wandb
import random
import pickle
import argparse
import pandas as pd
from RNN_attention import RNNModel
from CharacterEncoding import CharEncoder
import csv

random.seed()
torch.cuda.empty_cache()
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_csv_file(df, predicted_words, filename):
    input_words = []
    target_words = []
    for index, words in df.iterrows():
        input_words.append(words[0])
        target_words.append(words[1])
    # Combine the input, target, and predicted words into a list of tuples
    data = zip(input_words, target_words, predicted_words)

    # Open the file in write mode with newline='' to avoid extra newlines being added
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',')  # Set delimiter to comma
        # Write the header row if needed
        writer.writerow(['Input', 'Target', 'Predicted'])
        # Write each row of data
        for row in data:
            writer.writerow(row)

#Load data

#Replace with appropriate file paths
train_file_path = 'D:\\brunda\\aksharantar_sampled\\aksharantar_sampled\\tel\\tel_test_Copy.csv'  
val_file_path = 'D:\\brunda\\aksharantar_sampled\\aksharantar_sampled\\tel\\tel_valid_Copy.csv'
test_file_path = 'D:\\brunda\\aksharantar_sampled\\aksharantar_sampled\\tel\\tel_test_Copy.csv'


# Unsure about the encoding, so try 'utf-8' first and if does not work, try 'utf-16'
try:
    train_df = pd.read_csv(train_file_path, encoding='utf-8', header = None)
    val_df = pd.read_csv(val_file_path, encoding='utf-8', header = None)
    test_df = pd.read_csv(test_file_path, encoding='utf-8', header = None)
except UnicodeDecodeError:
    train_df = pd.read_csv(train_file_path, encoding='utf-16', header = None)
    val_df = pd.read_csv(val_file_path, encoding='utf-16', header = None)
    test_df = pd.read_csv(test_file_path, encoding='utf-16', header = None)

#Create encodings for all inputs and outputs
train_input_encoding = CharEncoder("English", device)
train_output_encoding = CharEncoder("Telugu", device)
for index, data in train_df.iterrows():
    train_input_encoding.encode(data[0])
    train_output_encoding.encode(data[1])


prev_validation_accuracy = 0

def save_model(model, name):
    save_path = "D:\\brunda\\code\\Assignment3\\RNN_Assignment3\\RNN_Assignment3\\Models\\" + name + ".pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)

def train_model(args, config = None):
    wandb.init(config = config, project=args.wandb_project, entity=args.wandb_entity)
    config = wandb.config
    
    #run_name = "Attention_{}_nLayers_{}_HiddenLayerSize_{}embeddingSize{}_dropout_{}_optimizer_{}_lr_{}_epochs_{}_batchsize_{}".format(config.cellType, config.nLayers, config.HiddenLayerSize, config.embedding_size, config.dropoutProb, config.optimizer,config.learningRate, config.epochs, config.batch_size)
    run_name = "Delete"
    wandb.run.name = run_name

    model = RNNModel(train_input_encoding, train_output_encoding, train_df, config, device)
    train_loss = 0
    train_accuracy = 0
    val_accuracy = 0
    epoch = 0
    
    for epoch in range(config.epochs):
        print("Epoch: {}".format(epoch + 1))
        train_loss, train_accuracy = model.train(config.batch_size)

        print("Training Loss: {} ", train_loss )
        _, val_accuracy = model.test(val_df, False)
        print("Validation Accuracy: {}", val_accuracy)
        
        wandb.log({"Epoch":epoch + 1, "TrainLoss":train_loss, "ValidationAccuracy": val_accuracy})

        if epoch > 0:
            if val_accuracy < 0.0001:
                print("Early stopping, Low validation accuracy")
                break

            if val_accuracy < 0.95 * prev_validation_accuracy:
                print("Early stopping, Decreasing validation accuracy")
                
                break
        prev_validation_accuracy = val_accuracy
    pred_words, test_accuracy = model.test(test_df, True)
    write_csv_file(test_df, pred_words, run_name)
    
    wandb.log( {"TrainAccuracy":train_accuracy})
    wandb.log({"ValidationAccuracy": val_accuracy})
    save_model(model, run_name)
    wandb.run.finish()
    return

def init_sweep(args):
    sweep_config = { "name" : "RNN1","method": "bayes"}
    metric = {
    "name" : "ValidationAccuracy",
    "goal" : "maximize"
    }
    sweep_config["metric"] = metric
    parameters_dict = {
    "epochs" : {"values" : [3]},
    "nLayers" : {"values":[1,2]},
    "embedding_size" : {"values":[16,32]},
    "HiddenLayerSize" : {"values":[256, 512]},
    "learningRate" : {"values":[1e-2, 1e-3]},
    "cellType" : {"values" : ["RNN", "GRU", "LSTM"]},
    "optimizer" : {"values":["SGD", "Adam"]},
    "batch_size" : {"values":[32, 64]},
    "dropoutProb" : {"values" : [0.0, 0.1]}
    }
    sweep_config["parameters"] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="RNN")
    wandb.agent(sweep_id, lambda: train_model(args), count=10)

if __name__ == "__main__":
    #Update the login key if the project visibility is not open
    wandb.login(key = "017101a520090630fd58ad8684de73bf54c45117")
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train Attention - Recurrent neural network with specified configurations')
    parser.add_argument('--wandb_project', '-wp', type=str, default='basic-intro', help='Weights & Biases project name') #basic-intro
    parser.add_argument('--wandb_entity', '-we', type=str, default='drbruap', help='Weights & Biases entity') #drbruap
    parser.add_argument('--epochs', '-e', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='Batch size for training')
    parser.add_argument('--embedding_size', '-es', type=int, default=16, help='Input embedding size')
    parser.add_argument('--nLayers', '-el', type=int, default=2, help='Number of layers in encoder & decoder')
    parser.add_argument('--HiddenLayerSize', '-hs', type=int, default=512, help='Size of hidden layer')
    parser.add_argument('--cellType', '-ct', type=str, default='LSTM', choices=['RNN', 'LSTM', 'GRU'], help='Type of RNN')
    parser.add_argument('--dropout_prob', '-dp', type=float, default=0.0, help='Droput probability')
    parser.add_argument('--learningRate', '-lr', type = float, default = 0.001, help = 'Learning Rate for Optimizer')
    parser.add_argument('--weightdecay', '-wd', type = float, default = 0.001, help = 'Weight decay for Optimizer')
    parser.add_argument('--optimizer', '-op', type = str, default = "Adam", choices=["SGD", "Adam"], help = "Algorithm for back propogataion")
    args = parser.parse_args()

    config = {
    "epochs" : args.epochs,
    "batch_size" : args.batch_size,
    "embedding_size" :args.embedding_size,
    "nLayers" : args.nLayers,
    "HiddenLayerSize" : args.HiddenLayerSize,
    "cellType" : args.cellType,
    "dropoutProb" : args.dropout_prob,
    "learningRate" : args.learningRate,
    "optimizer" : args.optimizer,
    "weightdecay" : args.weightdecay
    }
    #init_sweep(args)
    train_model(args, config)

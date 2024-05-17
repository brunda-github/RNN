# RNN

## Requirements
The following packages are required to train this model
1. torch
2. pandas
3. wandb
4. matplotlib

## Steps to train the vanilla RNN model
1. Install the required python packages as metioned above. 
2. Download the python modules (RNN.py, train.py, CharEncoding.py)
3. Update the train and test data variables (train_file_path , val_file_path, test_file_path) to appropriate dataset paths in train.py
4. Run the command by replacing myname myprojectname respectively. This will create a .pkl file after training(Update the path to save the pkl file accordingly)
#### python train.py --wandb_entity myname --wandb_project myprojectname
5. Run TestModel.py by updating the path and name of above generated pkl file and test data csv path to test the model
Note: Make sure wandb project visibility is open. If not, make sure to call wandb.login() before initialising wandb runs

5. Following command line arguments are supported for train.py
This python file can be executed to train a FFN model by passing required arguments as mentioned below

| Name                | Description                                                                           |
|---------------------|---------------------------------------------------------------------------------------|
| `wp`, `wandb_project` | Project name used to track experiments in Weights & Biases dashboard                |
| `we`, `wandb_entity`  | Wandb Entity used to track experiments in the Weights & Biases dashboard             |
| `e`, `epochs`          | Number of epochs to train the neural network                                         |
| `b`, `batch_size`      | Batch size used for training the neural network                                      |
| `lr`, `learning_rate`  | Learning rate used for optimizing model parameters                                    |
| `dp`, `dropout_prob`    Dropout probability                                     |
|`es`, `embedding_size`        | Input embeding size for encoder and decoder                                                                     |
|`el`,`nLayers`   | Number of layers in encoder and decoder                                                    |
| `hs`, `HiddenLayerSize`      | Hidden layer size of the cell                      |
|`ct`,`cellType`| RNN, LSTM, GRU|
| `op`, `optimizer`      | Optimizer used (`SGD`, `Adam`)                       |


---------------------------------------------------------

1. RNN.py - This file contains classes Encoder, Decoder and RNNModel built using torch module and provies the flexibility in defining the architechture to train the model
2. CharacterEncoding.py - This file defines CharEncoder class used to convert words to tensors for training the model.
3. TestModel.py - This module is used to load the model generated during training and test using test data
4. train.py - Used to execute the implementation/train the model

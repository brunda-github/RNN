import pickle
import pandas as pd
import wandb

wandb.login(key = "")
wandb.init(project='basic-intro', entity='drbruap')
config = {
    "epochs" : 3,
    "batch_size" : 64,
    "embedding_size" :16,
    "nLayers" : 2,
    "HiddenLayerSize" : 512,
    "cellType" : "LSTM",
    "dropoutProb" : 0,
    "learningRate" : 0.001,
    "optimizer" : "Adam"
    }
wandb. config = config
wandb.run.name = "TestAccuracy"
#Update with proper file path
test_file_path = 'tel_test.csv'


try:
    test_df = pd.read_csv(test_file_path, encoding='utf-8', header = None)
except UnicodeDecodeError:
    test_df = pd.read_csv(test_file_path, encoding='utf-16', header = None)

#Update with proper path and pkl file name
with open("LSTM_nLayers_2_HiddenLayerSize_512embeddingSize16_dropout_0_optimizer_Adam_lr_0.001_epochs_3_batchsize_64.pkl", 'rb') as f:
    model = pickle.load(f)
    
test_accuracy = model.test(test_df)
print(test_accuracy)
wandb.log({"TestAccurcay":test_accuracy})
wandb.run.finish()

    

# Node label prediction using GNN

In this project we use two GNN approaches for node label prediction.
    
    * Convolutional GNN (GCNConv)
    * Graph attention NN (GatConv)

In both cases we, allow a customizable number of node-embedding-update blocks, 
by passing a list of embedding dimensions (and number of heads in the case of GatConv). 

The model is trained on Planetoid data of global citations network.
The training follow a semi-supervised approach:

    - The graph nodes are split into training (~5%) validation and test. 
    - During training, only the training nodes have known labels (although the embedings 
    of the entire collection of nodes are used in the updates).

Tuning of hyperparameters and testing of best model:

    - The tuning of model hyperparameters (numebr of blocks, drop-out fractions, etc)
    is performed with Optuna on the validation set of nodes.
    - The best model out of the tuning is performed on the test set of nodes. 


Entropy loss series for tuning trials as well as model state dictionaries
are saved in `results` directory.

Running the model

Run `tasks/run_tuner.py` for tuning the model (whether Convolution or GraphAttention approaches)

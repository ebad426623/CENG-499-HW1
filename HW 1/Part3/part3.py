import torch
import torch.nn as nn
import numpy as np
import pickle
import itertools

# we load all the datasets of Part 3
# the train data is already shuffled, we don't need to shuffle it...
x_train, y_train = pickle.load(open("../datasets/part3_train_dataset.dat", "rb"))
x_validation, y_validation = pickle.load(open("../datasets/part3_validation_dataset.dat", "rb"))
x_test, y_test = pickle.load(open("../datasets/part3_test_dataset.dat", "rb"))

# We rescale each feature of data instances in the datasets
x_train = x_train/255.0
x_train = x_train.astype(np.float32)

x_test = x_test / 255.0
x_test = x_test.astype(np.float32)

x_validation = x_validation/255.0
x_validation = x_validation.astype(np.float32)

# and converting them into Pytorch tensors in order to be able to work with Pytorch
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train).to(torch.long)

x_validation = torch.from_numpy(x_validation)
y_validation = torch.from_numpy(y_validation).to(torch.long)

x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test).to(torch.long)

# Global
input_layer_units = 784
output_layer_units = 10


class MLPClassifier(nn.Module):
    def __init__(self, learning_rate: float, epoch_number: int, hidden_layer: int, hidden_layer_unit: int, activation_function):
        super(MLPClassifier, self).__init__()
        self.hidden_layer = hidden_layer
        self.hidden_layer_unit = hidden_layer_unit
        self.epoch = epoch_number
        self.lr = learning_rate
        self.activation = activation_function
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_layer_units, hidden_layer_unit))  # Input to first hidden layer
        
        for _ in range(1, hidden_layer):
            self.layers.append(nn.Linear(hidden_layer_unit, hidden_layer_unit))  # Hidden layers
        
        self.layers.append(nn.Linear(hidden_layer_unit, output_layer_units))       
        self.model = nn.Sequential(*self.layers)  
        # use this for error self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.model(x)
    

    def train(self, x_train, y_train):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        ce = nn.CrossEntropyLoss()

        for _ in range(self.epoch):
            optimizer.zero_grad()
            train_predictions = self.forward(x_train)
            loss = ce(train_predictions, y_train)
            loss.backward()
            optimizer.step()

    



def gridSearch():
    hidden_layer = [1, 2] #[1, 2, 3]
    hidden_layer_units = [10, 64]#[10, 100, 1000]
    epochs = [3, 4, 5]
    learning_rates = [0.0001, 0.001]#[0.0001, 0.001, 0.01, 0.1]
    activation_functions = [nn.Sigmoid]

    combinations = list(itertools.product(hidden_layer, hidden_layer_units, epochs, learning_rates, activation_functions))
    best_combo = combinations[0]
    count = 1

    for hl, hlu, e, lr, af in combinations:
            print("Combination ", count)
            print(f"Hidden Layers = {hl}, Hidden Layer Units = {hlu}, Epochs = {e}, Learning rate = {lr}, Activation Function = {af.__name__}")
            print()
            count += 1
 
            for _ in range (10):
                mlp = MLPClassifier(learning_rate = lr, epoch_number = e, hidden_layer = hl, hidden_layer_unit = hlu, activation_function=af)
                mlp.train(x_train, y_train)
            
    return best_combo

best_combo = gridSearch()
print(best_combo)


import torch
import torch.nn as nn
import numpy as np
import pickle

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


################# MY CODE STARTS HERE


sm = nn.Softmax(dim = 1)

# I am definining the MLP model
class MLP(nn.Module):
    def __init__(self, hidden_layers, hidden_units, activation_function):
        super(MLP, self).__init__()
        layers = []
        in_features = 784
        
        # hidden layers
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_features, hidden_units))
            layers.append(activation_function())
            in_features = hidden_units
        
        # output layer
        layers.append(nn.Linear(in_features, 10))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# evaluating model accuracy for the given dataset x & y
def evaluate_accuracy(model, x, y):
    with torch.no_grad():
        output = sm(model(x))
        predictions = torch.argmax(output, dim=1)
        accuracy = (predictions == y).float().mean().item()
    return accuracy
    
    
    
# I am defining the function to train the model
def train_model(model, epochs, learning_rate, x_train, y_train):
    output = None
    # loop over epochs for the current model
    for _ in range(epochs):
        output = model(x_train)
        loss = nn.CrossEntropyLoss()(output, y_train)
        
        # weight updates
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad
    
    



# FInally, we do grid search for hyperparameters
def grid_search():
    
    ## all the hyperparameter will be permuted from here
    
    #number of hidden layers
    hidden_layer_options = [1, 2]
    
    #number of units for each hidden layer
    hidden_unit_options = [64, 128]
    
    #learning rates
    learning_rate_options = [0.01, 0.001]  
    
    #epoch number (  takes alot of time to run :(   )
    epoch_options = [50]
    
    # only one activation function because theres already 16 permutations, two AF's will make it 32
    # I tried ReLu, Tanh, and Sigmoid, Tanh performed the best
    activation_function_options = [nn.Tanh]

    best_config = None
    best_accuracy = 0.0
    permutation_number = 1
    
    #printing labels to help understand the next print inside the nested loops
    print("Labels: [#Hidden layers, #Nodes in HLs, Learning rate, # of epochs, Activation function]\n")
    
    for hl in hidden_layer_options:
        for hu in hidden_unit_options:
            for lr in learning_rate_options:
                for epochs in epoch_options:
                    for af in activation_function_options:
                        
                        print("Running permutation: #%d" % permutation_number + ", configuration: [" + str(hl) + ", " + str(hu) + ", " + str(lr) + ", " + str(epochs) + ", Tanh]")
                        permutation_number += 1
                        accuracies = []
                        
                        #running 10 times for each permutation
                        for _ in range(10):
                            model = MLP(hidden_layers=hl, hidden_units=hu, activation_function=af)
                            train_model(model, epochs, lr, x_train, y_train)
                            accuracies.append(evaluate_accuracy(model, x_validation, y_validation))
                        
                        mean_accuracy = sum(accuracies)/10
                        confidence_interval = 1.96 * np.std(accuracies) / np.sqrt(10)
                        print("Mean Accuracy: %.3f," % mean_accuracy, "Confidence Interval: %.3f" %confidence_interval, "\n")
                        
                        #updating best_config if the new permutation is better than old best
                        if mean_accuracy > best_accuracy:
                            best_accuracy = mean_accuracy
                            best_config = [permutation_number, hl, hu, lr, epochs, af]

    return best_config

# final training on best model
def final_training_and_evaluation(best_config):
    
    #combining training and validation data
    x_train_val = torch.cat((x_train, x_validation), 0)
    y_train_val = torch.cat((y_train, y_validation), 0)
    accuracies = []
    
    for _ in range(10):
        model = MLP(hidden_layers=best_config[1], hidden_units=best_config[2], activation_function=best_config[5])
        train_model(model, best_config[4], best_config[3], x_train_val, y_train_val)
        accuracies.append(evaluate_accuracy(model, x_test, y_test))

    #final mean and conf int
    mean_accuracy = sum(accuracies)/10
    confidence_interval = 1.96 * np.std(accuracies) / np.sqrt(10)
    return mean_accuracy, confidence_interval



# running grid search 
best_config = grid_search()
print("Best permutation: #%d, configuration: " % best_config[0], "[" + str(best_config[1]) + ", " + str(best_config[2]) + ", " + str(best_config[3]) + ", " + str(best_config[4]) + ", Tanh]")

# running final training
mean_accuracy, confidence_interval = final_training_and_evaluation(best_config)
print(f"Test Mean Accuracy: %.4f," % mean_accuracy, "Test Confidence Interval: %.4f" % confidence_interval)
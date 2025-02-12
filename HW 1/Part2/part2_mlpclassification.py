import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt

# this function returns the neural network output for a given dataset and set of parameters
def forward_pass(w1, b1, w2, b2, input_data):
    """
    The network consists of 617 input layer nodes, 64 hidden layer nodes, and 3 output layer nodes
    The activation function of the hidden layer is tanh.
    The output layer should apply the softmax function to obtain posterior probability distribution. And the function should return this distribution
    Here you are expected to perform all the required operations for a forward pass over the network with the given dataset
    """

    """
    data_instances = input_data.shape[0]

    hidden_layer_output = torch.zeros(data_instances, hidden_layer)
    output_layer_output = torch.zeros(data_instances, output_layer)

    w1T = w1.T
    w2T = w2.T

    for i in range(data_instances):
        for j in range(hidden_layer):
            hidden_layer_output[i][j] = torch.tanh(b1[0][j] + torch.dot(input_data[i], w1T[j]))


    for i in range(data_instances):
        output_layer_output[i][0] = b2[0][0] + torch.dot(hidden_layer_output[i], w2T[0])
        output_layer_output[i][1] = b2[0][1] + torch.dot(hidden_layer_output[i], w2T[1])
        output_layer_output[i][2] = b2[0][2] + torch.dot(hidden_layer_output[i], w2T[2])

    output_layer_output = torch.softmax(output_layer_output, dim=1)
    """

    hidden_layer_input = input_data @ w1 + b1
    hidden_layer_output = torch.tanh(hidden_layer_input)

    output_layer_input = hidden_layer_output @ w2 + b2
    output_layer_output = torch.softmax(output_layer_input, dim=1)

    return output_layer_output

# we load all training, validation, and test datasets for the classification task
train_dataset, train_label = pickle.load(open("../datasets/part2_classification_train.dat", "rb"))
validation_dataset, validation_label = pickle.load(open("../datasets/part2_classification_validation.dat", "rb"))
test_dataset, test_label = pickle.load(open("../datasets/part2_classification_test.dat", "rb"))

# when you inspect the training dataset, you are going to see that the class instances are sequential (e.g., [1,1,1,1 ... 2,2,2,2,2 ... 3,3,3,3])
# we shuffle the training dataset by preserving instance-label relationship
indices = list(range(len(train_dataset)))
np.random.shuffle(indices)
train_dataset = np.array([train_dataset[i] for i in indices], dtype=np.float32)
train_label = np.array([train_label[i] for i in indices], dtype=np.float32)

# In order to be able to work with Pytorch, all datasets (and labels/ground truth) should be converted into a tensor
# since the datasets are already available as numpy arrays, we simply create tensors from them via torch.from_numpy()
train_dataset = torch.from_numpy(train_dataset)
train_label = torch.from_numpy(train_label)

validation_dataset = torch.from_numpy(validation_dataset)
validation_label = torch.from_numpy(validation_label)

test_dataset = torch.from_numpy(test_dataset)
test_label = torch.from_numpy(test_label)

# You are expected to create and initialize the parameters of the network
# Please do not forget to specify requires_grad=True for all parameters since they need to be trainable.


input_layer = 617
hidden_layer = 64
output_layer = 3


# w1 defines the parameters between the input layer and the hidden layer
# Here you are expected to initialize w1 via the Normal distribution (mean=0, std=1).
w1 = torch.normal(mean = 0, std = 1, size = (input_layer, hidden_layer), requires_grad = True)


# b1 defines the bias parameters for the hidden layer
# Here you are expected to initialize b1 via the Normal distribution (mean=0, std=1).
b1 = torch.normal(mean = 0, std = 1, size = (1, hidden_layer), requires_grad = True)


# w2 defines the parameters between the hidden layer and the output layer
# Here you are expected to initialize w2 via the Normal distribution (mean=0, std=1).
w2 = torch.normal(mean = 0, std = 1, size = (hidden_layer, output_layer), requires_grad = True)


# and finally, b2 defines the bias parameters for the output layer
# Here you are expected to initialize b2 via the Normal distribution (mean=0, std=1).
b2 = torch.normal(mean = 0, std = 1, size = (1, output_layer), requires_grad = True)


# you are expected to use the stochastic gradient descent optimizer
# w1, b1, w2 and b2 are the trainable parameters of the neural network
optimizer = torch.optim.SGD([w1, b1, w2, b2], lr=0.001)



# These arrays will store the loss values incurred at every training iteration
iteration_array = []
train_loss_array = []
validation_loss_array = []

# We are going to perform the backpropagation algorithm 'ITERATION' times over the training dataset
# After each pass, we are calculating the average/mean cross entropy loss over the validation dataset along with accuracy scores on both datasets.
ITERATION = 15000
for iteration in range(1, ITERATION+1):
    iteration_array.append(iteration)

    # we need to zero all the stored gradient values calculated from the previous backpropagation step.
    optimizer.zero_grad()
    # Using the forward_pass function, we are performing a forward pass over the network with the training data
    train_predictions = forward_pass(w1, b1, w2, b2, train_dataset)


    # Here you are expected to calculate the MEAN cross-entropy loss with respect to the network predictions and the training label

    train_crossentropy_log = train_label * torch.log(train_predictions)
    train_crossentropy_log_sum = -torch.sum(train_crossentropy_log, dim=1, keepdim=True)
    train_mean_crossentropy_loss = torch.mean(train_crossentropy_log_sum)

    
    train_loss_array.append(train_mean_crossentropy_loss.item())
    # We initiate the gradient calculation procedure to get gradient values with respect to the calculated loss
    train_mean_crossentropy_loss.backward()
    # After the gradient calculation, we update the neural network parameters with the calculated gradients.
    optimizer.step()

    # after each epoch on the training data we are calculating the loss and accuracy scores on the validation dataset
    # "with torch.no_grad()" disables gradient operations, since during testing the validation dataset, we don't need to perform any gradient operations
    with torch.no_grad():
        train_predictions = forward_pass(w1, b1, w2, b2, train_dataset)

        # Here you are expected to calculate the accuracy score on the training dataset by using the training labels.
        
        train_passed = torch.sum(torch.argmax(train_label, dim=1) == torch.argmax(train_predictions, dim=1)).item()
        train_accuracy = (train_passed / train_predictions.shape[0]) * 100 

        validation_predictions = forward_pass(w1, b1, w2, b2, validation_dataset)

        # Here you are expected to calculate the average/mean cross entropy loss for the validation dataset by using the validation dataset labels.
        validation_crossentropy_log = validation_label * torch.log(validation_predictions)
        validation_crossentropy_log_sum = -torch.sum(validation_crossentropy_log, dim=1, keepdim=True)
        validation_mean_crossentropy_loss = torch.mean(validation_crossentropy_log_sum)

        validation_loss_array.append(validation_mean_crossentropy_loss.item())

        # Similarly, here, you are expected to calculate the accuracy score on the validation dataset
        validation_passed = torch.sum(torch.argmax(validation_label, dim=1) == torch.argmax(validation_predictions, dim=1)).item()
        validation_accuracy = (validation_passed / validation_predictions.shape[0]) * 100 

    print("Iteration : %d - Train Loss %.4f - Train Accuracy : %.2f - Validation Loss : %.4f Validation Accuracy : %.2f" % (iteration, train_mean_crossentropy_loss.item(), train_accuracy, validation_mean_crossentropy_loss.item(), validation_accuracy))
   


# after completing the training, we calculate our network's accuracy score on the test dataset...
# Again, here we don't need to perform any gradient-related operations, so we are using torch.no_grad() function.

with torch.no_grad():
    test_predictions = forward_pass(w1, b1, w2, b2, test_dataset)
    # Here you are expected to calculate the network accuracy score on the test dataset...
    test_passed = torch.sum(torch.argmax(test_label, dim=1) == torch.argmax(test_predictions, dim=1)).item()
    test_accuracy = (train_passed / train_predictions.shape[0]) * 100
    print("Test accuracy : %.2f" % (float(test_accuracy)))

# We plot the loss versus iteration graph for both datasets (training and validation)
plt.plot(iteration_array, train_loss_array, label="Train Loss")
plt.plot(iteration_array, validation_loss_array, label="Validation Loss")
plt.legend()
plt.show()


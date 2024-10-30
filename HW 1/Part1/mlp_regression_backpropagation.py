import pickle
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x: np.array):
    return 1/(1+np.exp(-x))


class MLPRegressor:
    def __init__(self, learning_rate: float, epoch_number: int):
        # Please do not make any changes in this constructor

        # Weights are initialized temporarily for size clarification
        # These values are overwritten via the initialize_parameter function call belove.
        self.W = np.zeros((2, 3))  # of size (2, 3)
        self.W_bias = np.zeros((1, 3))  # of size (1, 3), row vector (1x3) matrix
        self.GAMMA = np.zeros((3, 1))  # of size (3, 1)
        self.GAMMA_bias = np.zeros((1, 1)) # of size (1, 1), row vector (1x1) matrix

        self.learning_rate = learning_rate
        self.epoch_number = epoch_number

        self.initialize_parameters()

    def initialize_parameters(self):
        # Please do not make any changes in this function

        # In order to be able to obtain the same results during training
        # Start with fixed initial weights
        self.W, self.W_bias, self.GAMMA, self.GAMMA_bias = pickle.load(open("../datasets/part1_regression_initial_weights.dat", "rb"))

        


    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: two-dimensional data numpy array fed to the network, size: (# of data instances, # of features)
        :return: two numpy arrays as the outputs of the hidden and output layer, the forward propagation result for the given data, sizes: (# of data instances, 3) (hidden layer output), (# of data instances, 1) (output layer output)

        Please implement the forward propagation procedure for the given data instances here
        This function should return two numpy arrays of sizes (# of data instances, 3) (hidden layer output), (# of data instances, 1) (output layer output)
        """
        data_instances = x.shape[0]

        hidden_layer_output = np.zeros((data_instances, 3))
        output_layer_output = np.zeros((data_instances, 1))

        wT = self.W.transpose()
        gT = self.GAMMA.transpose()

        for i in range(data_instances):
            hidden_layer_output[i][0] = sigmoid(self.W_bias[i][0] + np.dot(x[i], wT[0]))
            hidden_layer_output[i][1] = sigmoid(self.W_bias[i][1] + np.dot(x[i], wT[1]))
            hidden_layer_output[i][2] = sigmoid(self.W_bias[i][2] + np.dot(x[i], wT[2]))


        for i in range(data_instances):
            output_layer_output[i][0] = self.GAMMA_bias[i][0] + np.dot(hidden_layer_output[i], gT[i])


        return hidden_layer_output, output_layer_output

    def train(self, data_instances: np.ndarray, labels: np.ndarray):
        # Please only implement the part asked in this function
        # And please do not make any other changes on the already provided code pieces
        for iteration_count in range(1, self.epoch_number+1):

            for data_index in range(len(data_instances)):
                x = data_instances[data_index].reshape(1, -1)  # convert into a row vector, (1x2) matrix
                label = labels[data_index]
                hidden_layer_output, output_layer_output = self.forward(x)

                
                W_update = np.zeros_like(self.W)
                W_bias_update = np.zeros_like(self.W_bias)
                GAMMA_update = np.zeros_like(self.GAMMA)
                GAMMA_bias_update = np.zeros_like(self.GAMMA_bias)
                """
                    Please calculate the weight update rules for W, W_bias, GAMMA and GAMMA_bias matrices here
                    using the "x", "hidden_layer_output", "output_layer_output" and "label" variables defined above
                    
                    The amount of weight changes should be stored in "W_update", "W_bias_update", "GAMMA_update", "GAMMA_bias_update" variables.
                """

                #Derivatives

                dE_dO = 2 * (output_layer_output - label) 
                dO_dGAMMA = hidden_layer_output.flatten()

                GAMMA_update = np.transpose(dE_dO*dO_dGAMMA)
                GAMMA_bias_update = dE_dO 

                dO_dH = np.transpose(self.GAMMA)
                dH_dZ = dO_dGAMMA * (1 - dO_dGAMMA)
                dE_dZ = dE_dO * dO_dH * dH_dZ

                W_update = np.transpose(x) * dE_dZ
                W_bias_update = dE_dZ


                # After finding update values we are performing the weight updates
                self.W = self.W - self.learning_rate*W_update
                self.W_bias = self.W_bias - self.learning_rate*W_bias_update

                self.GAMMA = self.GAMMA - self.learning_rate*GAMMA_update
                self.GAMMA_bias = self.GAMMA_bias - self.learning_rate*GAMMA_bias_update

            # After each epoch on the dataset, calculate the mean squarred loss with the dataset.

            total_loss_value = 0.0
            for data_index in range(len(data_instances)):
                x = data_instances[data_index].reshape(1, -1) # convert into a row vector, 1x4 matrix
                label = labels[data_index]
                _, output = self.forward(x)
                total_loss_value += (label[0]-output[0][0])**2

            mean_mse = total_loss_value/len(data_instances)
            print(f"Epoch Number: {iteration_count} - Training Mean SE: {mean_mse:.3f}")

X, L = pickle.load(open("../datasets/part1_regression_dataset.dat", "rb"))
mlp = MLPRegressor(learning_rate=0.01, epoch_number=250)
mlp.train(X, L)

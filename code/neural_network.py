import math
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt


class CFARNeuralNet:

    def __init__(self,
                 x_train,
                 train_labels,
                 validation,
                 validation_labels,
                 activation,
                 activation_derivative,
                 hiddens,
                 learning_rate=0.3,
                 decay_rate=0.002,
                 xavier_init=True):
        self.x_train = x_train
        self.train_labels = train_labels
        self.validation = validation
        self.validation_labels = validation_labels
        self.accuracies = []
        self.tested_models = []

        self.n_samples, img_size = self.x_train.shape
        self.n_labels = np.unique(train_labels).size
        self.eta = learning_rate
        self.decay = decay_rate
        self.hiddens = hiddens
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.errors = np.array

        self.y_train = np.zeros((self.train_labels.shape[0], self.n_labels))
        for i in range(0, self.train_labels.shape[0]):
            self.y_train[i, self.train_labels[i].astype(int)] = 1

        n_input_layer = img_size
        n_output_layer = self.n_labels

        if xavier_init:
            self.weight1 = np.random.randn(self.hiddens[0], n_input_layer) * np.sqrt(1 / n_input_layer)
            if self.hiddens[1] > 0:
                self.weight2 = np.random.randn(self.hiddens[1], self.hiddens[0]) * np.sqrt(1 / self.hiddens[0])
                self.weight3 = np.random.randn(n_output_layer, self.hiddens[1]) * np.sqrt(1 / self.hiddens[1])
            else:
                self.weight2 = np.random.randn(n_output_layer, self.hiddens[0]) * np.sqrt(1 / self.hiddens[0])
        else:
            self.weight1 = np.random.uniform(0, 1, (self.hiddens[0], n_input_layer))
            self.weight2 = np.random.uniform(0, 1, (n_output_layer, self.hiddens[0]))

            self.weight1 = np.divide(self.weight1, np.matlib.repmat(np.sum(self.weight1, 1)[:, None], 1, n_input_layer))
            self.weight2 = np.divide(self.weight2,
                                     np.matlib.repmat(np.sum(self.weight2, 1)[:, None], 1, self.hiddens[0]))

            if self.hiddens[1] > 0:
                self.weight3 = np.random.uniform(0, 1, (n_output_layer, self.hiddens[1]))
                self.weight3 = np.divide(self.weight3,
                                         np.matlib.repmat(np.sum(self.weight3, 1)[:, None], 1, self.hiddens[1]))

                self.weight2 = np.random.uniform(0, 1, (self.hiddens[1], self.hiddens[1]))
                self.weight2 = np.divide(self.weight2,
                                         np.matlib.repmat(np.sum(self.weight2, 1)[:, None], 1, self.hiddens[0]))

        self.bias_weight1 = np.ones((self.hiddens[0],)) * (-self.x_train.mean())
        self.bias_weight2 = np.zeros((n_output_layer,))

        if self.hiddens[1] > 0:
            self.bias_weight3 = np.ones((n_output_layer,)) * (-0.5)
            self.bias_weight2 = np.ones((self.hiddens[1],)) * (-0.5)

    def __dropout(self, activation, dropout_prob=0.0001):
        if dropout_prob < 0 or dropout_prob > 1:
            return activation
        activation /= dropout_prob
        mult = np.random.rand(*activation.shape) < dropout_prob
        activation *= mult
        return activation

    def train(self, n_epochs=100, n_batches=100):
        self.errors = np.zeros((n_epochs,))
        batch_size = math.ceil(self.n_samples / n_batches)
        for i in range(0, n_epochs):
            shuffled_idxs = np.random.permutation(self.n_samples)
            for j in range(0, n_batches):
                delta_weight1 = np.zeros(self.weight1.shape)
                delta_weight2 = np.zeros(self.weight2.shape)

                delta_bias1 = np.zeros(self.bias_weight1.shape)
                delta_bias2 = np.zeros(self.bias_weight2.shape)

                delta_weight3 = np.array
                delta_bias3 = np.array
                if self.hiddens[1] > 0:
                    delta_weight3 = np.zeros(self.weight3.shape)
                    delta_bias3 = np.zeros(self.bias_weight3.shape)

                for k in range(0, batch_size):
                    idx = shuffled_idxs[j * batch_size + k]
                    x = self.x_train[idx]
                    desired_output = self.y_train[idx]

                    act1 = np.dot(self.weight1, x) + self.bias_weight1
                    out1 = self.activation(act1)

                    act2 = np.dot(self.weight2, out1) + self.bias_weight2
                    if self.hiddens[1] > 0:
                        out2 = self.activation(act2)

                        act3 = np.dot(self.weight3, out2) + self.bias_weight3
                        out3 = softmax(act3)
                        e_n = desired_output - out3

                        out3delta = e_n
                        delta_weight3 += np.outer(out3delta, out2)
                        delta_bias3 += out3delta

                        out2delta = self.activation_derivative(out2) * np.dot(self.weight3.T, out3delta)
                    else:
                        out2 = softmax(act2)
                        e_n = desired_output - out2
                        out2delta = e_n

                    delta_weight2 += np.outer(out2delta, out1)
                    delta_bias2 += out2delta

                    out1delta = self.activation_derivative(out1) * np.dot(self.weight2.T, out2delta)
                    delta_weight1 += np.outer(out1delta, x)
                    delta_bias1 += out1delta

                    if self.hiddens[1] > 0:
                        cost = - np.sum(desired_output * np.log(out3)) / self.n_samples
                    else:
                        cost = - np.sum(desired_output * np.log(out2)) / self.n_samples
                    self.errors[i] += cost

                self.weight1 += self.eta * delta_weight1 / batch_size
                self.weight2 += self.eta * delta_weight2 / batch_size

                self.bias_weight1 += self.eta * delta_bias1 / batch_size
                self.bias_weight2 += self.eta * delta_bias2 / batch_size

                if self.hiddens[1] > 0:
                    self.weight3 += self.eta * delta_weight3 / batch_size
                    self.bias_weight3 += self.eta * delta_bias3 / batch_size

            self.eta *= 1 / (1 + self.decay * i)
            print("Learning rate =", self.eta)

            print("Epoch", i + 1, ": error =", self.errors[i])
            print("Validation ", end='')
            self.tested_models.append(self.test(self.validation, self.validation_labels))
            if self.accuracies[i] \
                    < self.accuracies[i - 1] \
                    < self.accuracies[i - 2] \
                    < self.accuracies[i - 3]:
                self.__dict__.update(self.tested_models[i - 3].__dict__)
                print("Validation accuracy decreased. Stopped early.")
                break

    def plot_performance(self):
        plt.plot(np.trim_zeros(self.errors, trim='b'))
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title('Average error per epoch')
        plt.show()

    def test(self, x_test, test_labels):
        y_test = np.zeros((test_labels.shape[0], self.n_labels))
        for i in range(0, test_labels.shape[0]):
            y_test[i, test_labels[i].astype(int)] = 1
        n = x_test.shape[0]

        p_ra = 0
        correct_value = np.zeros((n,))
        predicted_value = np.zeros((n,))
        for i in range(0, n):
            x = x_test[i]
            y = y_test[i]
            correct_value[i] = np.argmax(y)

            act1 = np.dot(self.weight1, x) + self.bias_weight1
            out1 = self.activation(act1)

            act2 = np.dot(self.weight2, out1) + self.bias_weight2
            if self.hiddens[1] > 0:
                out2 = self.activation(act2)

                act3 = np.dot(self.weight3, out2) + self.bias_weight3
                out3 = softmax(act3)

                predicted_value[i] = np.argmax(out3)
            else:
                out2 = softmax(act2)
                predicted_value[i] = np.argmax(out2)

            if predicted_value[i] == correct_value[i]:
                p_ra = (p_ra + 1)

        accuracy = 100 * p_ra / n
        print("accuracy =", accuracy, "%")
        self.accuracies.append(accuracy)
        return self


# Activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def relu(x):
    return x * (x > 0)


def relu_derivative(x):
    return 1. * (x > 0)


def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps, keepdims=True)

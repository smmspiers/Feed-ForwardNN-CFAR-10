import neural_network as nn
import preprocessing as pp
import k_fold_cross_validation as kfx
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio


def main(args):
    message = """\
    There are three ways of running my neural network. Please select one.
    5 - Train 2 models, one with a ReLU activation function and the other with a Sigmoid activation 
        function. Both with 1 hidden layer and 50 neurons.
        Example: python3 run_network.py 5
    6 - Train 10 different models, each with a random number between 1-200 of hidden layer neurons.
        Example: python3 run_network.py 6
    7 - Train 1 model with 2 hidden layers each with an optional number of neurons.
        Example: python3 run_network.py 7 50 25
        (This would train a model wit type h 50 and 25 neurons on the first and second hidden layers respectively)"""
    if not args or not (args[0] == '5' or args[0] == '6' or args[0] == '7'):
        print(message)
        sys.exit()

    x_train, train_labels = import_training_data()
    x_train = pp.minmax_initialisation(x_train)
    pca_matrix = pp.get_pca_matrix(x_train)
    x_train = pp.reduce_dimensions(x_train, pca_matrix)

    x_test, test_labels = import_testing_data()
    x_test = pp.minmax_initialisation(x_test)
    x_test = pp.reduce_dimensions(x_test, pca_matrix)

    if args[0] == '5':
        method5(x_train, train_labels, x_test, test_labels)
    elif args[0] == '6':
        method6(x_train, train_labels, x_test, test_labels)
    elif args[0] == '7':
        method7(x_train, train_labels, x_test, test_labels, (int(args[1]), int(args[2])))


def method5(train_data, train_labels, test_data, test_labels):
    kfx.train_test_plot_model(train_data, train_labels, test_data, test_labels)
    kfx.train_test_plot_model(train_data, train_labels, test_data, test_labels,
                              activation=nn.sigmoid, activation_derivative=nn.sigmoid_derivative)


def method6(train_data, train_labels, test_data, test_labels, n_trials=5):
    hidden_neurons = np.arange(20, 201, 20)  # [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    accuracies = np.empty((n_trials, hidden_neurons.size))
    for i in range(n_trials):
        print("Trial", i + 1)
        for j in range(hidden_neurons.size):
            print("Initialising model with", hidden_neurons[j], "hidden layer neurons.")
            accuracies[i, j] = kfx.train_test_plot_model(train_data, train_labels, test_data, test_labels,
                                                         hiddens=(hidden_neurons[j], 0)).accuracies[-1]
    print(accuracies)
    avg_accuracies = np.mean(accuracies, axis=0)
    print(avg_accuracies)
    print(np.std(accuracies, axis=0))
    plt.errorbar(x=hidden_neurons, y=avg_accuracies, yerr=np.std(accuracies, axis=0))
    plt.xlabel('Number of hidden layer neurons')
    plt.ylabel('Accuracy (%)')
    plt.title('Average accuracy against hidden layer neurons')
    plt.show()


def method7(train_data, train_labels, test_data, test_labels, hiddens):
    kfx.train_test_plot_model(train_data, train_labels, test_data, test_labels, hiddens=hiddens)


def import_training_data():
    mat = spio.loadmat('../train_data', squeeze_me=True)
    train_data = mat['x_train']
    train_labels = mat['x_train_labs']
    return train_data, np.array(train_labels) - 1


def import_testing_data():
    mat2 = spio.loadmat('../test_data', squeeze_me=True)
    test_data = mat2['x_test']
    test_labels = mat2['x_test_labs']
    return test_data, np.array(test_labels) - 1


if __name__ == '__main__':
    main(sys.argv[1:])

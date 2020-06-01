import neural_network as nn
import numpy as np


def train_test_plot_model(train_data,
                          train_labels,
                          test_data,
                          test_labels,
                          activation=nn.relu,
                          activation_derivative=nn.relu_derivative,
                          hiddens=(50, 0)):
    neural_nets = generate_models_for_cross_valid(train_data,
                                                       train_labels,
                                                       activation,
                                                       activation_derivative,
                                                       hiddens)
    neural_net = perform_k_fold_cross_validation(neural_nets)
    # neural_net_relu = early_stopping_model_factory(train_data,
    #                                                train_labels,
    #                                                activation,
    #                                                activation_derivative,
    #                                                hiddens)
    neural_net.plot_performance()
    print("Test ", end='')
    return neural_net.test(test_data, test_labels)


def perform_k_fold_cross_validation(models):
    if len(models) == 1:
        return models.train
    accuracies = []
    for model in models:
        print("Fold", models.index(model) + 1, "/", len(models))
        model.train()
        model.plot_performance()
        accuracies.append(model.accuracies[-1])
    mean = sum(accuracies) / len(accuracies)
    print("Average accuracy over", len(models), "folds =", round(mean, 2), "%")
    return models[int(np.argmax(accuracies))]


def generate_models_for_cross_valid(x_train,
                                    x_labels,
                                    activation,
                                    activation_derivative,
                                    hiddens,
                                    k=5):
    groups = np.array_split(x_train, k)
    groups_labels = np.array_split(x_labels, k)
    models = []
    for i in range(k):
        validation_set = groups[i]
        validation_labels = groups_labels[i]
        train_set = np.concatenate(groups[:i] + groups[i + 1:])
        train_labels = np.concatenate(groups_labels[:i] + groups_labels[i + 1:])
        models.append(nn.CFARNeuralNet(train_set,
                                       train_labels,
                                       validation_set,
                                       validation_labels,
                                       activation,
                                       activation_derivative,
                                       hiddens))
    return models


def early_stopping_model_factory(x_train,
                                 train_labels,
                                 activation=nn.relu,
                                 activation_derivative=nn.relu_derivative,
                                 hiddens=(50, 0),
                                 train_val_split=0.2):
    train_size = int(x_train.shape[0] - train_val_split * x_train.shape[0])
    train_data = x_train[:train_size, :]
    validation = x_train[train_size:, :]

    labels_size = int(train_labels.shape[0] - train_val_split * train_labels.shape[0])
    validation_labels = train_labels[labels_size:]
    train_labels = train_labels[:labels_size]

    return nn.CFARNeuralNet(train_data,
                            train_labels,
                            validation,
                            validation_labels,
                            activation,
                            activation_derivative,
                            hiddens)

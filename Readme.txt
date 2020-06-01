There are the following three ways of running my code, which I used to produce my results.
The numbering is the same as the numbering for the questions in the assignment sheet.

5 - Train 2 models, one with a ReLU activation function and the other with a Sigmoid activation
    function. Both with 1 hidden layer and 50 neurons.
    Example: python3 run_network.py 5

6 - Train 10 different models, each with a random number between 1-200 of hidden layer neurons.
    Example: python3 run_network.py 6

7 - Train 1 model with 2 hidden layers each with an optional number of neurons.
    Example: python3 run_network.py 7 50 25
    (This would train a model with 50 and 25 neurons on the first and second hidden layers respectively)

(FYI to run you'll need to download the CFAR-10 dataset from https://www.cs.toronto.edu/~kriz/cifar.html and
place the files outside the code folder.)

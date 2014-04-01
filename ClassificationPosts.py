from NeuralNetwork import NeuralNetwork
from PostReader import PostReader


def main():
    base_directory = ""
    ignore_word_file = ""

    pr = PostReader(base_directory, ignore_word_file)

    nb_inputs = len(pr.get_word_set())

    neural_network = NeuralNetwork(nb_inputs)

    training_set = pr.get_training_set()
    neural_network.train(training_set)



if __name__ == "__main__":
    main()
from NeuralNetwork import NeuralNetwork
from PostReader import PostReader


def main():
    base_directory = "corpus_short"
    ignore_word_file = "corpus/frenchST.txt"

    print("post reading...")
    pr = PostReader(base_directory, ignore_word_file, False)

    print("creating neural network...")
    nb_hidden_neurons = 10
    neural_network = NeuralNetwork(pr.get_word_set(), nb_hidden_neurons)

    print("training...")
    neural_network.train(pr.get_training_set())

    print("classifying...")
    for msg in pr.get_verification_set():
        final = neural_network.classify(msg[0])
        print("result: %s    expected: %s" % (final, msg[1]))


if __name__ == "__main__":
    main()
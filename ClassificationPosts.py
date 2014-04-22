from NeuralNetwork import NeuralNetwork
from PostReader import PostReader


def main():
    base_directory = "corpus_short"
    ignore_word_file = "corpus/frenchST.txt"

    print ("post reading...")
    pr = PostReader(base_directory, ignore_word_file)

    print("creating neural network...")
    neural_network = NeuralNetwork(pr.get_word_set())

    print("training...")
    neural_network.train(pr.get_training_set())

    print("classifying...")
    for msg in pr.get_verification_set():
        print("result: %s    expected: %s" % (neural_network.classify(msg[0]), msg[1]))


if __name__ == "__main__":
    main()
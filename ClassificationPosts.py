from NeuralNetwork import NeuralNetwork
from PostReader import PostReader


def main():
    base_directory = "corpus_xor"
    ignore_word_file = "corpus/frenchST.txt"

    print("post reading...")
    pr = PostReader(base_directory, ignore_word_file, False)

    print("creating neural network...")
    nb_hidden_neurons = 20
    nb_max_iteration = 1000
    nn = NeuralNetwork(pr.get_word_set(), nb_hidden_neurons, nb_max_iteration)

    print("training...")
    nn.train(pr.get_training_set())

    print("verification...")
    verification_set = pr.get_verification_set()
    nb_correct = 0
    for msg in verification_set:
        final = NeuralNetwork.threshold(nn.classify(msg[0]))
        if final == msg[1]:
            nb_correct += 1

    print("=======================")
    print("verification set length: %s" % len(verification_set))
    print("nb correct classified  : %s" % nb_correct)
    print("=======================")


if __name__ == "__main__":
    main()
from NeuralNetwork import NeuralNetwork
from PostReader import PostReader
import time


def main():
    base_directory = "corpus_short"
    ignore_word_file = "corpus/frenchST.txt"

    print("post reading...")
    filtered = True
    pr = PostReader(base_directory, ignore_word_file, filtered)

    print("creating neural network...")
    nb_hidden_neurons = 20
    nb_max_iteration = 10
    nn = NeuralNetwork(pr.get_word_set(), nb_hidden_neurons, nb_max_iteration)

    print("training...")
    training_set = pr.get_training_set()
    t0 = time.clock()
    nb_iteration = nn.train(training_set)
    training_time = time.clock() - t0

    print("verification...")
    t0 = time.clock()
    verification_set = pr.get_verification_set()
    verification_time = time.clock() - t0
    nb_correct = 0
    for msg in verification_set:
        final = NeuralNetwork.threshold(nn.classify(msg[0]))
        if final == msg[1]:
            nb_correct += 1

    print("=======================")
    print("training set length    : %s" % len(training_set))
    print("nb hidden neurons      : %s" % nb_hidden_neurons)
    print("nb max iterations      : %s" % nb_max_iteration)
    print("nb iterations          : %s" % nb_iteration)
    print("verification set length: %s posts" % len(verification_set))
    print("nb correct classified  : %s posts" % nb_correct)
    print("rate                   : %i %%" % (nb_correct / len(verification_set) * 100))
    print("training time          : %i s" % training_time)
    print("verification time      : %i s" % verification_time)
    print("=======================")
    print("")


if __name__ == "__main__":
    main()
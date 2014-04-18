from NeuralNetwork import NeuralNetwork
from PostReader import PostReader


def main():
    base_directory = ""
    ignore_word_file = ""

    pr = PostReader(base_directory, ignore_word_file)

    word_set = ['hello', 'world', 'elm', 'neuchatel', 'hirzel']
    neural_network = NeuralNetwork(word_set)

    training_set = [
        ({'hello': 1.0, 'world': 1.0, 'elm': 1.0, 'neuchatel': 0.0, 'hirzel': 0.0}, 1.0),
        ({'hello': 1.0, 'world': 1.0, 'elm': 1.0, 'neuchatel': 0.0, 'hirzel': 0.0}, 1.0),
        ({'hello': 1.0, 'world': 1.0, 'elm': 1.0, 'neuchatel': 0.0, 'hirzel': 0.0}, 1.0),
        ({'hello': 1.0, 'world': 1.0, 'elm': 1.0, 'neuchatel': 0.0, 'hirzel': 0.0}, 1.0),
        ({'hello': 1.0, 'world': 1.0, 'elm': 1.0, 'neuchatel': 0.0, 'hirzel': 0.0}, 1.0),
        ({'hello': 0.0, 'world': 1.0, 'elm': 1.0, 'neuchatel': 0.0, 'hirzel': 0.0, }, 0.0),
        ({'hello': 0.0, 'world': 1.0, 'elm': 1.0, 'neuchatel': 0.0, 'hirzel': 0.0, }, 0.0),
        ({'hello': 0.0, 'world': 1.0, 'elm': 1.0, 'neuchatel': 0.0, 'hirzel': 0.0, }, 0.0),
        ({'hello': 0.0, 'world': 1.0, 'elm': 1.0, 'neuchatel': 0.0, 'hirzel': 0.0, }, 0.0),
        ({'hello': 0.0, 'world': 1.0, 'elm': 1.0, 'neuchatel': 0.0, 'hirzel': 0.0, }, 0.0),
        ({'hello': 0.0, 'world': 1.0, 'elm': 1.0, 'neuchatel': 0.0, 'hirzel': 0.0, }, 0.0),
        ({'hello': 0.0, 'world': 1.0, 'elm': 1.0, 'neuchatel': 0.0, 'hirzel': 0.0, }, 0.0),
    ]
    neural_network.train(training_set)

    msgs_to_classify = [
        ({'hello': 1.0, 'world': 1.0, 'elm': 1.0, 'neuchatel': 0.0, 'hirzel': 0.0}, 1.0),
        ({'hello': 1.0, 'world': 1.0, 'elm': 1.0, 'neuchatel': 0.0, 'hirzel': 0.0}, 1.0),
        ({'hello': 1.0, 'world': 1.0, 'elm': 1.0, 'neuchatel': 0.0, 'hirzel': 0.0}, 1.0),
        ({'hello': 1.0, 'world': 1.0, 'elm': 1.0, 'neuchatel': 0.0, 'hirzel': 0.0}, 1.0),
        ({'hello': 1.0, 'world': 1.0, 'elm': 1.0, 'neuchatel': 0.0, 'hirzel': 0.0}, 1.0),
        ({'hello': 0.0, 'world': 1.0, 'elm': 1.0, 'neuchatel': 0.0, 'hirzel': 0.0, }, 0.0),
        ({'hello': 0.0, 'world': 1.0, 'elm': 1.0, 'neuchatel': 0.0, 'hirzel': 0.0, }, 0.0),
        ({'hello': 0.0, 'world': 1.0, 'elm': 1.0, 'neuchatel': 0.0, 'hirzel': 0.0, }, 0.0),
        ({'hello': 0.0, 'world': 1.0, 'elm': 1.0, 'neuchatel': 0.0, 'hirzel': 0.0, }, 0.0),
        ({'hello': 0.0, 'world': 1.0, 'elm': 1.0, 'neuchatel': 0.0, 'hirzel': 0.0, }, 0.0),
        ({'hello': 0.0, 'world': 1.0, 'elm': 1.0, 'neuchatel': 0.0, 'hirzel': 0.0, }, 0.0),
        ({'hello': 0.0, 'world': 1.0, 'elm': 1.0, 'neuchatel': 0.0, 'hirzel': 0.0, }, 0.0),
    ]

    for msg in msgs_to_classify:
        print("result: %s    expected: %s" % (neural_network.classify(msg[0]), msg[1]))


if __name__ == "__main__":
    main()
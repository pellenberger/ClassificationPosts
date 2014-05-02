from NeuralNetwork import Neuron, NeuralNetwork
import unittest


class NeuronStaticTestcase(unittest.TestCase):
    def test_sigma(self):
        self.assertTrue(Neuron.equal(Neuron.sigma(0.0), 0.5))
        self.assertTrue(Neuron.equal(Neuron.sigma(1.0), 0.75))
        self.assertTrue(Neuron.equal(Neuron.sigma(-1.0), 0.25))

    def test_sigma_derivative(self):
        self.assertTrue(Neuron.equal(Neuron.sigma_derivative(0.0), 0.25))
        self.assertTrue(Neuron.equal(Neuron.sigma_derivative(-5), 0.0))
        self.assertTrue(Neuron.equal(Neuron.sigma_derivative(+5), 0.0))

    def test_sigma_derivative_with_sigma(self):
        for x in range(-5, 5):
            alternative = Neuron.sigma(x) - Neuron.sigma(x) * Neuron.sigma(x)
            self.assertAlmostEqual(Neuron.sigma_derivative(x), alternative, 5)

        for x in range(-5, 5):
            alternative = Neuron.sigma(x) * (1 - Neuron.sigma(x))
            self.assertAlmostEqual(Neuron.sigma_derivative(x), alternative, 5)

    def test_equal(self):
        self.assertTrue(Neuron.equal(0.0, 0.0))
        self.assertTrue(Neuron.equal(0.0, 0.01))
        self.assertTrue(Neuron.equal(0.0, 0.05))
        self.assertTrue(Neuron.equal(1.0, 0.95))
        self.assertTrue(Neuron.equal(1.0, 1.05))

        self.assertFalse(Neuron.equal(0.0, 1.0))
        self.assertFalse(Neuron.equal(1.0, 0.0))


class NeuronTestcase(unittest.TestCase):
    def setUp(self):
        self.neuron = Neuron(['a', 'b', 'c'])

        self.neuron.input_weights['a'] = 0.25
        self.neuron.input_weights['b'] = 0.50
        self.neuron.input_weights['c'] = 0.75

    def test_calc(self):
        self.assertAlmostEqual(self.neuron.calc({'a': 1.0, 'b': 0.0, 'c': 0.0}), 0.562177, 4)
        self.assertAlmostEqual(self.neuron.calc({'a': 0.0, 'b': 1.0, 'c': 0.0}), 0.622459, 4)
        self.assertAlmostEqual(self.neuron.calc({'a': 0.0, 'b': 0.0, 'c': 1.0}), 0.679179, 4)

        self.assertAlmostEqual(self.neuron.calc({'a': 1.0, 'b': 1.0, 'c': 1.0}), 0.817574, 4)
        self.assertAlmostEqual(self.neuron.calc({'a': 1.0, 'b': 1.0, 'c': 0.0}), 0.679179, 4)
        self.assertAlmostEqual(self.neuron.calc({'a': 1.0, 'b': 0.0, 'c': 1.0}), 0.731059, 4)
        self.assertAlmostEqual(self.neuron.calc({'a': 0.0, 'b': 1.0, 'c': 1.0}), 0.7773, 4)


class NeuralNetworkTestcase(unittest.TestCase):
    def setUp(self):
        self.nn = NeuralNetwork(['a', 'b'], 2)

        self.nn.hidden_neurons[0].input_weights['a'] = 0.25
        self.nn.hidden_neurons[0].input_weights['b'] = 0.50
        self.nn.hidden_neurons[0].bias = 0.0

        self.nn.hidden_neurons[1].input_weights['a'] = 0.75
        self.nn.hidden_neurons[1].input_weights['b'] = 0.75
        self.nn.hidden_neurons[1].bias = 0.0

        self.nn.final_neuron.input_weights[0] = 0.5
        self.nn.final_neuron.input_weights[1] = 0.5
        self.nn.final_neuron.bias = 0.0

    def test_calc(self):
        self.nn.classify({'a': 1.0, 'b': 0.0})
        self.assertAlmostEquals(self.nn.final_neuron.last_output, 0.650373, 5)


class NeuralNetworkXORTestcase(unittest.TestCase):
    def setUp(self):
        self.nn = NeuralNetwork(['a', 'b'], 2)

        self.nn.hidden_neurons[0].input_weights['a'] = 1.0
        self.nn.hidden_neurons[0].input_weights['b'] = 1.0
        self.nn.hidden_neurons[0].bias = 0.0

        self.nn.hidden_neurons[1].input_weights['a'] = 1.0
        self.nn.hidden_neurons[1].input_weights['b'] = 1.0
        self.nn.hidden_neurons[1].bias = 0.0

        self.nn.final_neuron.input_weights[0] = -1
        self.nn.final_neuron.input_weights[1] = 1
        self.nn.final_neuron.bias = 0.0

    def test_classifiy(self):
        self.assertAlmostEquals(self.nn.classify({'a': 1.0, 'b': 0.0}), 1.0, 5)
        self.assertAlmostEquals(self.nn.classify({'a': 0.0, 'b': 1.0}), 1.0, 5)
        self.assertAlmostEquals(self.nn.classify({'a': 1.0, 'b': 1.0}), 0.0, 5)
        self.assertAlmostEquals(self.nn.classify({'a': 0.0, 'b': 0.0}), 0.0, 5)


    def test_train(self):
        self.nn = NeuralNetwork(['a', 'b'], 2)
        self.nn.train([[{'a': 1.0, 'b': 0.0}, 1.0]])
        self.nn.train([[{'a': 0.0, 'b': 1.0}, 1.0]])
        self.nn.train([[{'a': 1.0, 'b': 0.0}, 1.0]])
        self.nn.train([[{'a': 0.0, 'b': 1.0}, 1.0]])
        self.nn.train([[{'a': 1.0, 'b': 0.0}, 1.0]])
        self.nn.train([[{'a': 0.0, 'b': 1.0}, 1.0]])
        self.nn.train([[{'a': 1.0, 'b': 0.0}, 1.0]])
        self.nn.train([[{'a': 0.0, 'b': 1.0}, 1.0]])


        self.assertAlmostEquals(self.nn.classify({'a': 1.0, 'b': 0.0}), 1.0, 5)
        self.assertAlmostEquals(self.nn.classify({'a': 0.0, 'b': 1.0}), 1.0, 5)
        self.assertAlmostEquals(self.nn.classify({'a': 1.0, 'b': 1.0}), 0.0, 5)
        self.assertAlmostEquals(self.nn.classify({'a': 0.0, 'b': 0.0}), 0.0, 5)


        self.nn.hidden_neurons[0].input_weights




if __name__ == '__main__':
    unittest.main()
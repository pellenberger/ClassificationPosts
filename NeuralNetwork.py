import random
import math


class Neuron:
    def __init__(self, inputs):
        self.nb_inputs = len(inputs)
        self.input_weights = {}
        self._last_inputs = {}
        self.last_output = 0.0

        self._init_input_weights(inputs)
        pass

    def _init_input_weights(self, inputs):
        for i in inputs:
            self.input_weights[i] = random.random()

        print(u"Neuron init configuration: %s" % self.input_weights)

    def calc(self, input_vector):
        assert len(input_vector) == self.nb_inputs

        weighted_values = [self.input_weights[key] * input_vector[key] for key in input_vector.keys()]

        self._last_inputs = input_vector
        self.last_output = Neuron.sigma(sum(weighted_values))
        return self.last_output

    def apply_back_propagation(self, learning_rate, delta_k):
        for k in self.input_weights.keys():
            delta_weight = -learning_rate * delta_k * self._last_inputs[k]
            self.input_weights[k] += delta_weight

    @staticmethod
    def sigma(x):
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def sigma_derivative(x):
        sigma_x = Neuron.sigma(x)
        return sigma_x - sigma_x * sigma_x

    @staticmethod
    def equal(actual, prediction):
        return abs(actual - prediction) < 0.1


class NeuralNetwork:
    NB_MAX_ITERATIONS = 10
    NB_HIDDEN_NEURON = 3
    ALPHA_LEARNING_RATE = 0.2

    def __init__(self, inputs):
        self.nb_max_inputs = len(inputs)
        self.hidden_neurons = []
        print("Init final neuron: ")
        self.final_neuron = Neuron(range(0, NeuralNetwork.NB_HIDDEN_NEURON))

        print("Init hidden neuron: ")
        self._init_hidden_neuron(inputs)

    def _init_hidden_neuron(self, inputs):
        for i in range(0, NeuralNetwork.NB_HIDDEN_NEURON):
            self.hidden_neurons.append(Neuron(inputs))

    def train(self, learning_set):
        convergence = False

        safety_counter = 0
        while not convergence:
            no_changement = True
            for entry in learning_set:
                actual = self.classify(entry[0])
                prediction = entry[1]
                if not Neuron.equal(actual, prediction):
                    delta_k = NeuralNetwork.delta_k(actual, prediction)

                    weights_jk = [self.final_neuron.input_weights[k] for k in self.final_neuron.input_weights.keys()]

                    delta_j_dict = {}
                    for neuron in self.hidden_neurons:
                        delta_j_dict[neuron] = NeuralNetwork.delta_j(neuron.last_output, delta_k, weights_jk)

                    self.final_neuron.apply_back_propagation(NeuralNetwork.ALPHA_LEARNING_RATE, delta_k)

                    for neuron in self.hidden_neurons:
                        neuron.apply_back_propagation(NeuralNetwork.ALPHA_LEARNING_RATE, delta_j_dict[neuron])

                    no_changement = False

            safety_counter += 1

            print("Iteration %s" % safety_counter)
            if no_changement or safety_counter >= NeuralNetwork.NB_MAX_ITERATIONS:
                convergence = True
                print("Convergence reached")

        print(u"Neuron final configuration: %s" % self.final_neuron.input_weights)
        for neuron in self.hidden_neurons:
            print(u"Neuron final configuration: %s" % neuron.input_weights)

    @staticmethod
    def hidden_layer_error(output_i, delta_j):
        return output_i * delta_j

    @staticmethod
    def output_layer_error(output_j, delta_k):
        return output_j * delta_k

    @staticmethod
    def delta_k(output_k, target_k):
        return output_k * (1 - output_k) * (output_k - target_k)

    @staticmethod
    def delta_j(output_j, delta_k, weights_jk):
        return output_j * (1 - output_j) * sum([w * delta_k for w in weights_jk])

    def classify(self, input_vector):
        assert len(input_vector) == self.nb_max_inputs

        hidden_neuron_results = {}
        for i in range(0, NeuralNetwork.NB_HIDDEN_NEURON):
            hidden_neuron_results[i] = self.hidden_neurons[i].calc(input_vector)

        final = self.final_neuron.calc(hidden_neuron_results)
        print("final: %s" % final)
        return 1.0 if Neuron.equal(final, 1.0) else 0.0


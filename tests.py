from unittest import TestCase, main

from layer import Layer
from neuron import Neuron, count_arrays_product
from perceptron import Perceptron


class NeuronTestCase(TestCase):
    def test_count_arrays_product(self):
        self.assertEqual(
            first=count_arrays_product([-1, 1, -1], [-1, 1, 1]),
            second=[1, 1, -1],
        )

    def test_count_arrays_product_wrong_lengths_arguments(self):
        with self.assertRaises(ValueError):
            count_arrays_product([1, -1, -1], [0, 0])

        with self.assertRaises(ValueError):
            count_arrays_product([0], [1, -1, -1, -8])

    def test_Neuron__init__(self):
        neuron = Neuron(inputs_number=4)
        self.assertEqual(len(neuron.weights), 5)

        neuron = Neuron(inputs_number=47)
        self.assertEqual(len(neuron.weights), 48)

    def test_Neuron__init__wrong_inputs_number(self):
        with self.assertRaises(expected_exception=ValueError):
            Neuron(inputs_number=1)

        with self.assertRaises(expected_exception=ValueError):
            Neuron(inputs_number=0)

    def test_Neuron__call__(self):
        neuron = object.__new__(Neuron)
        neuron.weights = [-1, 1, -1, 1, -1]

        # The case with weighted_sum > 0
        self.assertEqual(neuron(inputs_values=[1, -1, 1, -1]), 1)

        # The case with weighted_sum < 0
        self.assertEqual(neuron(inputs_values=[-1, 1, -1, 1]), 0)

        # The case with weighted_sum = 0
        neuron.weights = [-1, 1, -1, 1]
        self.assertEqual(neuron(inputs_values=[1, 1, 1]), 1)

    def test_Neuron_delete_weight(self):
        neuron = object.__new__(Neuron)
        neuron.weights = [-1, 1, -1, 1, -1]

        neuron._delete_weight(weight_number=1)
        self.assertEqual(neuron.weights, [-1, 1, 1, -1])

        neuron._delete_weight(weight_number=0)
        self.assertEqual(neuron.weights, [-1, 1, -1])

    def test_Neuron_delete_weight_over_minimal(self):
        neuron = Neuron(inputs_number=3)

        neuron._delete_weight(weight_number=1)

        with self.assertRaises(expected_exception=PermissionError):
            neuron._delete_weight(weight_number=1)


class LayerTestCase(TestCase):
    def test_Layer__init__(self):
        layer = Layer(neuron_inputs_number=7, neurons_number=9)
        self.assertEqual(len(layer.neurons), 9)
        for neuron in layer.neurons:
            self.assertEqual(len(neuron.weights), 8)

    def test_Layer__init__wrong_arguments(self):
        with self.assertRaises(ValueError):
            Layer(neuron_inputs_number=3, neurons_number=0)

        with self.assertRaises(ValueError):
            Layer(neuron_inputs_number=1, neurons_number=3)

    def test_Layer__call__(self):
        layer = Layer(neuron_inputs_number=2, neurons_number=2)
        neurons_weights = [[-1, 1, 1], [1, -1, 1]]
        for neuron, new_weights in zip(layer.neurons, neurons_weights):
            neuron.weights = new_weights
        self.assertEqual(layer(inputs_values=[-1, 1]), [0, 1])

        layer = Layer(neuron_inputs_number=2, neurons_number=4)
        neurons_weights = [[-1, 1, 1], [1, -1, 1], [-1, 1, -1], [1, 1, 1]]
        for neuron, new_weights in zip(layer.neurons, neurons_weights):
            neuron.weights = new_weights
        self.assertEqual(layer(inputs_values=[-1, 1]), [0, 1, 0, 1])

    def test_Layer_add_neuron(self):
        layer = Layer(neuron_inputs_number=2, neurons_number=1)
        self.assertEqual(len(layer.neurons), 1)
        layer._add_neuron()
        self.assertEqual(len(layer.neurons), 2)
        self.assertEqual(
            first=len(layer.neurons[0].weights),
            second=len(layer.neurons[1].weights),
        )

    def test_Layer_insert_neuron(self):
        layer = Layer(neuron_inputs_number=2, neurons_number=3)
        neuron_0 = layer.neurons[0]
        neuron_1 = layer.neurons[1]
        neuron_2 = layer.neurons[2]

        layer._insert_neuron(index=2)
        self.assertEqual(len(layer.neurons), 4)

        self.assertIs(neuron_0, layer.neurons[0])
        self.assertIs(neuron_1, layer.neurons[1])
        self.assertIs(neuron_2, layer.neurons[3])

        for neuron in layer.neurons:
            self.assertEqual(len(neuron.weights), 3)

    def test_Laye_add_weights(self):
        layer = Layer(neuron_inputs_number=10, neurons_number=3)
        layer._add_weights()
        for neuron in layer.neurons:
            self.assertEqual(len(neuron.weights), 12)

    def test_Laye_delete_weights(self):
        layer = Layer(neuron_inputs_number=3, neurons_number=3)
        neuron = layer.neurons[0]

        bias_weight = neuron.weights[0]
        weight_1 = neuron.weights[1]
        weight_3 = neuron.weights[3]

        layer._delete_weights(weight_number=2)
        self.assertEqual(bias_weight, neuron.weights[0])
        self.assertEqual(weight_1, neuron.weights[1])
        self.assertEqual(weight_3, neuron.weights[2])

        for neuron in layer.neurons:
            self.assertEqual(len(neuron.weights), 3)


class PerceptronTestCase(TestCase):
    def test_Perceptron__init__(self):
        structure = [2, 3, 10, 1]
        perceptron = Perceptron(structure)
        self.assertEqual(len(perceptron.layers), len(structure) - 1)
        for index, layer in enumerate(perceptron.layers):
            self.assertEqual(len(layer.neurons), structure[index + 1])
            for neuron in layer.neurons:
                self.assertEqual(len(neuron.weights), structure[index] + 1)


if __name__ == '__main__':
    main()

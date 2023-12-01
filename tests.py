from unittest import TestCase, main

from neuron import Neuron, count_arrays_product


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


if __name__ == '__main__':
    main()

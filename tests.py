from pathlib import Path
from time import time
from unittest import TestCase, main

from layer import Layer
from neuron import Neuron, count_arrays_product
from perceptron import Perceptron
from recurrent import Recurrent
from utils import (check_dir_path_slash_ending, conv_int_to_list,
                   conv_list_to_int, dict_sum, get_element_by_adress,
                   get_index_by_adress, get_unicode_characters_by_ranges,
                   make_simple_structure, mean, measure_execution_time, mix_in,
                   split_by_evenodd_position, split_by_volumes)


class UtilsTestCase(TestCase):
    def test_split_by_volumes_wrong_args(self):
        with self.assertRaises(ValueError):
            split_by_volumes([50], [3, 1])
        with self.assertRaises(ValueError):
            split_by_volumes([10, 1, 4, 16, 8], [3])
        with self.assertRaises(ValueError):
            split_by_volumes([10, 1, 4, 16, 8], [2, 3, 0])
        with self.assertRaises(ValueError):
            split_by_volumes([10, 1, 4, 0, 16, 8], [2, 3, 2])

    def test_split_by_volumes(self):
        self.assertEqual(
            first=split_by_volumes([10, 1, 4, 0, 16, 8], [2, 3, 1]),
            second=[[10, 1], [4, 0, 16], [8]],
        )
        self.assertEqual(
            first=split_by_volumes([10, 1, 4, 0, 16, 8], [3, 2]),
            second=[[10, 1, 4], [0, 16], [8]],
        )

    def test_get_unicode_characters_by_ranges(self):
        ranges = [(0x0020, 0x0026), (0x0400, 0x0408)]
        self.assertEqual(
            first=get_unicode_characters_by_ranges(ranges),
            second=' !"#$%&ЀЁЂЃЄЅІЇЈ',
        )
        self.assertEqual(
            first=len(get_unicode_characters_by_ranges(ranges)),
            second=16,
        )
        ranges = [(0x0408, 0x0400), (0x0020, 0x0026)]
        with self.assertRaises(ValueError):
            get_unicode_characters_by_ranges(ranges)

    def test_mean(self):
        with self.assertRaises(ValueError):
            mean([10])
        with self.assertRaises(ValueError):
            mean([])
        self.assertEqual(mean((8, 6, 16)), 10)

    def test_dict_sum(self):
        with self.assertRaises(ValueError):
            dict_sum(dict())
        with self.assertRaises(ValueError):
            dict_sum(dict(A=10))
        self.assertEqual(
            first=dict_sum(dict(A=50, B=27, C=23, D=1)),
            second=101,
        )

    def test_conv_int_to_list(self):
        self.assertEqual(
            first=conv_int_to_list(50, 9),
            second=[0, 0, 0, 1, 1, 0, 0, 1, 0],
        )
        with self.assertRaises(ValueError):
            conv_int_to_list(50, 3)

    def test_conv_list_to_int(self):
        self.assertEqual(
            first=conv_list_to_int([0, 0, 0, 1, 1, 0, 0, 1, 0]),
            second=50,
        )
        with self.assertRaises(ValueError):
            conv_list_to_int([0, 0, 0, 1, 9, 0, 0, 1, 0])

    def test_get_index_by_adress(self):
        sequence=[2, 10, 5, 7, 33, 97, 8, 125]
        self.assertEqual(
            first=get_index_by_adress(sequence, adress=[0, 1, 0]),
            second=2,
        )
        self.assertEqual(
            first=get_index_by_adress(sequence, adress=[1, 1, 1]),
            second=7,
        )
        with self.assertRaises(ValueError):
            get_index_by_adress(sequence=[22], adress=[0])

    def test_get_element_by_adress(self):
        sequence=[2, 10, 5, 7, 33, 97, 8, 125]
        self.assertEqual(
            first=get_element_by_adress(sequence, adress=[0, 1, 0]),
            second=5,
        )
        self.assertEqual(
            first=get_element_by_adress(sequence, adress=[1, 0, 1]),
            second=97,
        )
        with self.assertRaises(ValueError):
            get_element_by_adress(sequence=[128], adress=[1, 0, 1])

    def check_dir_path_slash_ending(self):
        with self.assertRaises(ValueError):
            check_dir_path_slash_ending('/usr')

    def test_make_simple_structure(self):
        self.assertEqual(
            first=make_simple_structure(
                inputs_number=3,
                intermediate_layers_number=4,
                intermediate_layers_neurons_number=5,
                outputs_number=20
            ),
            second=[3, 5, 5, 5, 5, 20],
        )

    def test_measure_execution_time(self):
        print_decorated = measure_execution_time(print)
        self.assertIsNot(print, print_decorated)
        start = time()
        testing_time = print_decorated('Testing print')
        self.assertIs(type(testing_time), float)
        self.assertGreater(time() - start, testing_time)

    def test_mix_in(self):
        class B:
            def method_5(self):
                pass
            def method_6(self):
                pass

        @mix_in(B)
        class A:
            def method_1(self):
                pass
            def method_2(self):
                pass
            def method_3(self):
                pass
            
        self.assertIs(A.__dict__['method_5'], B.method_5)
        self.assertIs(A.__dict__['method_6'], B.method_6)

    def test_split_by_evenodd_position(self):
        self.assertEqual(
            first=split_by_evenodd_position([0, 1, 2, 3, 4, 5]),
            second=([0, 2, 4], [1, 3, 5]),
        )
        with self.assertRaises(ValueError):
            split_by_evenodd_position([0, 1, 2, 3, 4])
        with self.assertRaises(ValueError):
            split_by_evenodd_position([4])


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

        neuron._delete_weight(index=2)
        self.assertEqual(neuron.weights, [-1, 1, 1, -1])

        neuron._delete_weight(index=1)
        self.assertEqual(neuron.weights, [-1, 1, -1])

    def test_Neuron_delete_weight_over_minimal(self):
        neuron = Neuron(inputs_number=3)
        with self.assertRaises(expected_exception=PermissionError):
            neuron._delete_weight(index=0)

        neuron._delete_weight(index=1)
        with self.assertRaises(expected_exception=PermissionError):
            neuron._delete_weight(index=1)


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
        neuron_inputs_number = 12
        layer = Layer(neuron_inputs_number, neurons_number=3)
        neuron = layer.neurons[0]

        bias_weight = neuron.weights[0]
        weight_2 = neuron.weights[2]
        weight_3 = neuron.weights[3]

        layer._delete_weights(index=1)
        self.assertEqual(bias_weight, neuron.weights[0])
        self.assertEqual(weight_2, neuron.weights[1])
        self.assertEqual(weight_3, neuron.weights[2])

        for neuron in layer.neurons:
            self.assertEqual(len(neuron.weights), neuron_inputs_number)


class PerceptronTestCase(TestCase):
    def test_Perceptron__init__(self):
        structure = [3, 3]
        with self.assertRaises(ValueError):
            perceptron = Perceptron(structure)

        structure = [2, 3, 10, 1]
        perceptron = Perceptron(structure)
        self.assertEqual(len(perceptron.layers), len(structure) - 1)
        for index, layer in enumerate(perceptron.layers):
            self.assertEqual(len(layer.neurons), structure[index + 1])
            for neuron in layer.neurons:
                self.assertEqual(len(neuron.weights), structure[index] + 1)

    def test_Perceptron_init_with_weights(self):
        structure  = [2, 2, 2]
        weights = [1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, -1]
        with self.assertRaises(ValueError):
            perceptron = Perceptron.init_with_weights(structure, weights[:-2])
        perceptron = Perceptron.init_with_weights(structure, weights)
        real_weights = list()
        for layer in perceptron.layers:
            for neuron in layer.neurons:
                real_weights.extend(neuron.weights)
        self.assertEqual(weights, real_weights)

    def test_Perceptron_structure(self):
        structures = [[2, 2, 1], [2, 3, 10, 1], [17, 10, 67, 4], [3, 3, 3, 1]]
        for structure in structures:
            self.assertEqual(Perceptron(structure).structure, structure)

    def test_Perceptron__call__(self):
        structure  = [2, 2, 2]
        weights = [1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, -1]
        perceptron = Perceptron.init_with_weights(structure, weights)
        self.assertEqual(perceptron([0, 1]), [1, 0])

    def test_all_neurons(self):
        structure = [7, 3, 8, 2, 10]
        perceptron = Perceptron(structure)
        self.assertEqual(len(perceptron.all_neurons), 23)

    def test_all_weights(self):
        structure = [7, 3, 8, 2, 10]
        perceptron = Perceptron(structure)
        self.assertEqual(len(perceptron.all_weights), 104)

    def test_save_and_load(self):
        structure = [7, 3, 8, 2, 10]
        perceptron_1 = Perceptron(structure)
        self.assertFalse(
            Path(
                f'{str(Path(__file__).parent.absolute())}/testing.perceptron',
            ).exists(),
        )
        with self.assertRaises(ValueError):
            perceptron_1.save(
                dir_path=str(Path(__file__).parent.absolute()),
                file_name='testing',
            )
        perceptron_1.save(
            dir_path=str(Path(__file__).parent.absolute()) + '/',
            file_name='testing',
        )
        self.assertTrue(
            Path(
                f'{str(Path(__file__).parent.absolute())}/testing.perceptron',
            ).exists(),
        )
        perceptron_2 = Perceptron.load(
            f'{str(Path(__file__).parent.absolute())}/testing.perceptron',
        )
        self.assertEqual(perceptron_1, perceptron_2)
        Path(
            f'{str(Path(__file__).parent.absolute())}/testing.perceptron',
        ).unlink()


class RecurrentTestCase(TestCase):
    def test_Recurrent__init__(self):
        with self.assertRaises(ValueError):
            Recurrent(inputs_number=0, outputs_number=7)
        with self.assertRaises(ValueError):
            Recurrent(inputs_number=12, outputs_number=0)
        self.assertEqual(Recurrent().structure, [26, *6*[10], 334])

    def test_save_and_load(self):
        recurrent_1 = Recurrent()
        self.assertFalse(
            Path(
                f'{str(Path(__file__).parent.absolute())}/testing.recurrent',
            ).exists(),
        )
        with self.assertRaises(ValueError):
            recurrent_1.save(
                dir_path=str(Path(__file__).parent.absolute()),
                file_name='testing',
            )
        recurrent_1.save(
            dir_path=str(Path(__file__).parent.absolute()) + '/',
            file_name='testing',
        )
        self.assertTrue(
            Path(
                f'{str(Path(__file__).parent.absolute())}/testing.recurrent',
            ).exists(),
        )
        recurrent_2 = Recurrent.load(
            f'{str(Path(__file__).parent.absolute())}/testing.recurrent',
        )
        self.assertEqual(recurrent_1, recurrent_2)
        Path(
            f'{str(Path(__file__).parent.absolute())}/testing.recurrent',
        ).unlink()
        Path(
            f'{str(Path(__file__).parent.absolute())}/testing.perceptron',
        ).unlink()


if __name__ == '__main__':
    main()

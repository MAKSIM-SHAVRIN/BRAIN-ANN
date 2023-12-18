from json import dump, load
from pathlib import Path
from time import time

from perceptron import Perceptron
from utils import (check_dir_path_slash_ending, conv_int_to_list, dict_sum,
                   get_element_by_adress, get_index_by_adress,
                   split_by_volumes)


class Recurrent(Perceptron):
    # Controlling signals
    STOP_SIGNAL = [0, 0, 1]
    SKIP_SIGNAL = [0, 1, 0]
    REPEAT_SIGNAL = [1, 0, 0]
    STOP_REFLECTIONS_SIGNAL = [1, 1, 1]
    NOTHING_SIGNAL = [0, 0, 0]

    # Transforming signals
    ADD_NEURON_SIGNAL = [1, 0, 0]
    DELETE_NEURON_SIGNAL = [0, 1, 1]

    ADD_WRITTING_MEMORY_SIGNAL = [0, 0, 1]
    DELETE_WRITTING_MEMORY_SIGNAL = [1, 1, 0]

    ADD_READING_MEMORY_SIGNAL = [1, 0, 1]
    DELETE_READING_MEMORY_SIGNAL = [0, 1, 0]

    # Net structure
    INITIAL_MIDDLE_LAYERS_STRUCTURE: list = 6*[10]
    INITIAL_WRITING_MEMORY_CELLS_NUMBER: int = 5
    INITIAL_READING_MEMORY_CELLS_NUMBER: int = 5
    ADRESS_POWER: int = 10

    MEMORY_CELL_STRUCTURE = dict(
        layer_adress=ADRESS_POWER,
        neuron_adress=ADRESS_POWER,
        weight_adress=ADRESS_POWER,
    )
    REFORMING_NEURONS_STRUCTURE = dict(
        signal_neurons_number=3,
        layer_adress=ADRESS_POWER,
        neuron_adress=ADRESS_POWER,
    )
    SIGNAL_NEURONS_NUMBER: int = 3
    TIME_INPUTS_NUMBER: int = 5
    REFLECTIONS_INPUTS_NUMBER: int = 8


    def __init__(self, inputs_number: int = 8, outputs_number: int = 8):
        if inputs_number < 1:
            raise ValueError('Recurrent must have at least one input')
        if outputs_number < 1:
            raise ValueError('Recurrent must have at least one output')

        self.inputs_number = inputs_number
        self.outputs_number = outputs_number

        super().__init__(self._initial_perceptron_structure)
        return None  # Added for visual end of the method

    @property
    def _initial_perceptron_structure(self) -> list[int]:
        structure = [
            self._initial_inputs_number,
            *self.INITIAL_MIDDLE_LAYERS_STRUCTURE,
            self._initial_outputs_number,
        ]
        return structure

    @property
    def _initial_inputs_number(self) -> int:
        perceptron_inputs_number = dict_sum(
            dict(
                signifying_inputs=self.inputs_number,
                time_inputs=self.TIME_INPUTS_NUMBER,
                reflections_counter_inputs=self.REFLECTIONS_INPUTS_NUMBER,
                reading_memory_inputs=self.INITIAL_READING_MEMORY_CELLS_NUMBER,
            ),
        )
        return perceptron_inputs_number

    @property
    def _initial_outputs_number(self) -> int:
        memory_cell_neurons_number = dict_sum(self.MEMORY_CELL_STRUCTURE)
        w_memory_neurons_number = self.INITIAL_WRITING_MEMORY_CELLS_NUMBER\
            * memory_cell_neurons_number
        r_memory_neurons_number = self.INITIAL_READING_MEMORY_CELLS_NUMBER\
            * memory_cell_neurons_number

        perceptron_outputs_number = dict_sum(
            dict(
                signifying_outputs=self.outputs_number,
                signal_neurons=self.SIGNAL_NEURONS_NUMBER,
                reforming_neurons=dict_sum(self.REFORMING_NEURONS_STRUCTURE),
                writing_memory_neurons=w_memory_neurons_number,
                reading_memory_neurons=r_memory_neurons_number,
            ),
        )
        return perceptron_outputs_number

    def save(self, dir_path: str, file_name: str) -> str:
        check_dir_path_slash_ending(dir_path)

        file = f'{dir_path}{file_name}.recurrent'
        with open(file, mode='w', encoding='ascii') as filebuffer:
            dump(
                obj=dict(
                    inputs_number=self.inputs_number,
                    outputs_number=self.outputs_number,
                ),
                fp=filebuffer,
            )
        super().save(dir_path, file_name)
        return file

    @classmethod
    def load(cls, file: str):
        with open(file, mode='r', encoding='ascii') as filebuffer:
            dictionary = load(fp=filebuffer)
        dir_path_and_name = str(Path(file).parent.absolute())\
            + '/'\
            + str(Path(file).stem)
        perceptron = Perceptron.load(f'{dir_path_and_name}.perceptron')

        recurrent = object.__new__(cls)
        recurrent.layers = perceptron.layers
        recurrent.inputs_number = dictionary['inputs_number']
        recurrent.outputs_number = dictionary['outputs_number']
        return recurrent

    @property
    def reading_memory_cells_number(self) -> int:
        return self.structure[0]\
            - self.inputs_number\
            - self.TIME_INPUTS_NUMBER\
            - self.REFLECTIONS_INPUTS_NUMBER

    @property
    def writing_memory_cells_number(self) -> int:
        memory_cell_neurons_number = dict_sum(self.MEMORY_CELL_STRUCTURE)
        reading_memory_neurons_number = self.reading_memory_cells_number\
            * memory_cell_neurons_number
        writing_memory_neurons_number = self.structure[-1]\
            - self.outputs_number\
            - self.SIGNAL_NEURONS_NUMBER\
            - dict_sum(self.REFORMING_NEURONS_STRUCTURE)\
            - reading_memory_neurons_number
        resoult = writing_memory_neurons_number\
            / dict_sum(self.MEMORY_CELL_STRUCTURE)
        return round(resoult)

    @property
    def outputs_structure(self) -> dict:
        writing_memory_neurons = self.writing_memory_cells_number\
            * dict_sum(self.MEMORY_CELL_STRUCTURE)
        reading_memory_neurons = self.reading_memory_cells_number\
            * dict_sum(self.MEMORY_CELL_STRUCTURE)
        return dict(
            signifying_outputs=self.outputs_number,
            signal_neurons=self.SIGNAL_NEURONS_NUMBER,
            reforming_neurons=dict_sum(self.REFORMING_NEURONS_STRUCTURE),
            writing_memory_neurons=writing_memory_neurons,
            reading_memory_neurons=reading_memory_neurons,
        )

    def _write_weights(self, writing_memory: list[int]):
        for _ in range(self.writing_memory_cells_number):
            (
                layer_outputs, neuron_outputs, weight_outputs,
                writing_memory,

            ) = split_by_volumes(
                list_for_split=writing_memory,
                volumes=self.MEMORY_CELL_STRUCTURE.values(),
            )

            neurons = get_element_by_adress(self.layers, layer_outputs).neurons
            weights = get_element_by_adress(neurons, neuron_outputs,).weights
            weight_index = get_index_by_adress(weights, weight_outputs)

            weights[weight_index] = -weights[weight_index]

    def _read_weights(self, reading_memory: list[int]):
        reading_memory_inputs_binary_list = list()
        for _ in range(self.reading_memory_cells_number):
            (
                layer_outputs, neuron_outputs, weight_outputs,
                reading_memory,

            ) = split_by_volumes(
                list_for_split=reading_memory,
                volumes=self.MEMORY_CELL_STRUCTURE.values(),
            )

            neurons = get_element_by_adress(self.layers, layer_outputs).neurons
            weights = get_element_by_adress(neurons, neuron_outputs,).weights
            weight_index = get_index_by_adress(weights, weight_outputs)

            reading_memory_inputs_binary_list.append(weights[weight_index])
        return reading_memory_inputs_binary_list

    def _add_neuron(self, layer_adress: list[int]):
        # `self.layers[:-1]` exclude last layer cuz it contains signal clasters
        layer_index = get_index_by_adress(self.layers[:-1], layer_adress)
        self.layers[layer_index]._add_neuron()
        # add weights for each neuron of the next layer
        self.layers[layer_index + 1]._add_weights()

    def _delete_neuron(self, layer_adress: list[int], neuron_adress: list):
        # `self.layers[:-1]` exclude last layer cuz it contains signal clasters
        layer_index = get_index_by_adress(self.layers[:-1], layer_adress)
        layer = self.layers[layer_index]
        neuron_index = get_index_by_adress(layer.neurons, neuron_adress)
        layer.neurons.pop(neuron_index)
        # delete due weights for each neuron of the next layer
        self.layers[layer_index + 1]._delete_weights(neuron_index)

    def _add_writing_memory_neurons(self):
        last_layer = self.layers[-1]
        inputs_number = len(self.layers[-2].neurons)
        index = self.outputs_number\
            + self.SIGNAL_NEURONS_NUMBER\
            + dict_sum(self.REFORMING_NEURONS_STRUCTURE)
        for _ in range(dict_sum(self.MEMORY_CELL_STRUCTURE)):
            last_layer._insert_neuron(index)

    def _delete_writing_memory_neurons(self):
        if self.writing_memory_cells_number:
            last_layer = self.layers[-1]
            index = self.outputs_number\
                + self.SIGNAL_NEURONS_NUMBER\
                + dict_sum(self.REFORMING_NEURONS_STRUCTURE)
            for _ in range(dict_sum(self.MEMORY_CELL_STRUCTURE)):
                last_layer.neurons.pop(index)

    def _add_reading_memory_neurons(self):
        last_layer = self.layers[-1]
        inputs_number = len(self.layers[-2].neurons)
        for _ in range(dict_sum(self.MEMORY_CELL_STRUCTURE)):
            last_layer._add_neuron()
        # add reading memory input
        self.layers[0]._add_weights()

    def _delete_reading_memory_neurons(self):
        if self.reading_memory_cells_number:
            last_layer = self.layers[-1]
            for _ in range(dict_sum(self.MEMORY_CELL_STRUCTURE)):
                last_layer.neurons.pop()
            # delete reading memory input
            self.layers[0]._delete_weights(weight_number=self.structure[0])

    def _transform(self, transforming_outputs: list[int]):
        signal, layer_adress, neuron_adress = split_by_volumes(
            list_for_split=transforming_outputs,
            volumes=self.REFORMING_NEURONS_STRUCTURE.values(),
        )
        # Add or delete neuron
        if signal == self.ADD_NEURON_SIGNAL:
            self._add_neuron(layer_adress)

        elif signal == self.DELETE_NEURON_SIGNAL:
            self._delete_neuron(layer_adress, neuron_adress)

        # Add or delete writting memory neurons
        elif signal == self.ADD_WRITTING_MEMORY_SIGNAL:
            self._add_writing_memory_neurons()

        elif signal == self.DELETE_WRITTING_MEMORY_SIGNAL:
            self._delete_writing_memory_neurons()

        # Add or delete reading memory neurons
        elif signal == self.ADD_READING_MEMORY_SIGNAL:
            self._add_reading_memory_neurons()

        elif signal == self.DELETE_READING_MEMORY_SIGNAL:
            self._delete_reading_memory_neurons()

    def _introspect(
        self, writting_memory: list[int], reading_memory: list[int],
    ) -> list[int]:
        # Read weights
        reading_memory_inputs = self._read_weights(reading_memory)
        # Write weights
        self._write_weights(writting_memory)
        return reading_memory_inputs

    def __call__(
        self, inputs: list[list[int]],
        time_limit: float=None, steps_limit: int=None,
        transform=True, introspect=True,
        just_last_resoult=False, do_not_skip_and_repeat=False,

    ) -> list[list[int]]:

        # Start of timer
        if time_limit:
            start_time = time()

        # Start of steps counting
        if steps_limit:
            steps_counter = 0

        # initial controlling signal is always do nothing
        controlling_signal = self.NOTHING_SIGNAL

        # Save initial request for use it if reflections stopped
        initial_inputs = inputs

        # Fill initial reading_memory_inputs by zero values
        reading_memory_inputs = [0,] * self.reading_memory_cells_number

        # Reflections loop
        while True:
            inputs = initial_inputs
            for reflection in range(2 ** self.REFLECTIONS_INPUTS_NUMBER - 1):
                resoults = list()
                # Iterations
                for inputs_values in inputs:
                    if not do_not_skip_and_repeat:
                        if controlling_signal == self.SKIP_SIGNAL:
                            controlling_signal = self.NOTHING_SIGNAL
                            continue
                    # Repeating
                    while True:
                        # Stop by time limit
                        if time_limit and time_limit < time() - start_time:
                            controlling_signal = self.STOP_SIGNAL
                            break

                        # Stop by steps limit
                        if steps_limit and steps_limit < steps_counter:
                            controlling_signal = self.STOP_SIGNAL
                            break

                        # Convert time data to lists of binary signals
                        time_binary_list = conv_int_to_list(
                            number=int(time()),
                            width=self.TIME_INPUTS_NUMBER,
                        )
                        reflections_binary_list = conv_int_to_list(
                            number=reflection,
                            width=self.REFLECTIONS_INPUTS_NUMBER,
                        )

                        # Get outputs as list of binary signals and
                        # Split binary list to valuable binary lists
                        (
                            signifying_outputs,
                            controlling_signal,
                            transforming,
                            writting_memory,
                            reading_memory,

                        ) = split_by_volumes(
                                list_for_split=super().__call__(
                                    [
                                        *inputs_values,
                                        *time_binary_list,
                                        *reflections_binary_list,
                                        *reading_memory_inputs,
                                    ],
                                ),
                                volumes=self.outputs_structure.values(),
                            )

                        # Introspecton
                        if introspect:
                            self._introspect(writting_memory, reading_memory)

                        # Transforming
                        if transform:
                            self._transform(transforming_outputs=transforming)

                        # Add character to list of resoults
                        if just_last_resoult:
                            resoults = [signifying_outputs,]
                        else:
                            resoults.append(signifying_outputs)

                        # Increase steps counter
                        steps_counter += 1

                        # Stop repeating
                        if do_not_skip_and_repeat:
                            break
                        if controlling_signal != self.REPEAT_SIGNAL:
                            break

                    # Stop iterations
                    if controlling_signal == self.STOP_SIGNAL:
                        break
                    elif controlling_signal == self.STOP_REFLECTIONS_SIGNAL:
                        break

                # Stop reflection
                if controlling_signal == self.STOP_SIGNAL:
                    break
                elif controlling_signal == self.STOP_REFLECTIONS_SIGNAL:
                    break
                # Resoults are empty or not enough for the next reflection
                elif resoults == list() or just_last_resoult:
                    break

                # Prepare request for new reflection from results
                inputs = resoults

            # Stop reflections loop
            if controlling_signal == self.STOP_SIGNAL:
                break
        return resoults

    def __eq__(self, o):
        return self.layers == o.layers\
            and self.inputs_number == o.inputs_number\
            and self.outputs_number == o.outputs_number

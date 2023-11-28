from json import dump, load
from time import time

from perceptron import Perceptron
from utils import (conv_int_to_list, conv_list_to_int, dict_sum,
                   split_by_volumes, get_index_by_fraction)


class Recurrent(Perceptron):
    SIGNALS = dict(
        STOP_SIGNAL=[0, 0, 1],
        SKIP_SIGNAL=[0, 1, 0],
        REPEAT_SIGNAL=[1, 0, 0],
        STOP_REFLECTIONS_SIGNAL=[1, 1, 1],

        ADD_NEURON_SIGNAL=[1, 0, 0],
        DELETE_NEURON_SIGNAL=[0, 1, 1],

        ADD_WRITTING_MEMORY_SIGNAL=[0, 0, 1],
        DELETE_WRITTING_MEMORY_SIGNAL=[1, 1, 0],
        ADD_READING_MEMORY_SIGNAL=[1, 0, 1],
        DELETE_READING_MEMORY_SIGNAL=[0, 1, 0],
    )

    MEMORY_CELL_STRUCTURE = dict(
        layer_adress=10,
        neuron_adress=10,
        weight_adress=10,
    )
    REFORMING_NEURONS_STRUCTURE = dict(
        signal_neurons_number=3,
        layer_adress=10,
        neuron_adress=10,
    )
    SIGNAL_NEURONS_NUMBER: int = 3
    TIME_INPUTS_NUMBER: int = 5
    REFLECTIONS_INPUTS_NUMBER: int = 8

    def __init__(
        self,
        signifying_inputs_number: int = 8, signifying_outputs_number: int = 8,
        middle_layers_structure: list = 6*[10],
        writing_memory_cells_number: int = 5,
        reading_memory_cells_number: int = 5,
    ):
        self.signifying_inputs_number = signifying_inputs_number
        self.signifying_outputs_number = signifying_outputs_number

        memory_cell_neurons_number = dict_sum(self.MEMORY_CELL_STRUCTURE)
        writing_memory_neurons_number = writing_memory_cells_number\
            * memory_cell_neurons_number
        reading_memory_neurons_number = reading_memory_cells_number\
            * memory_cell_neurons_number
        inputs_number = dict_sum(
            dict(
                signifying_inputs=self.signifying_inputs_number,
                time_inputs=self.TIME_INPUTS_NUMBER,
                reflections_counter_inputs=self.REFLECTIONS_INPUTS_NUMBER,
                reading_memory_inputs=reading_memory_neurons_number,
            ),
        )
        outputs_number = dict_sum(
            dict(
                signifying_outputs=self.signifying_outputs_number,
                signal_neurons=self.SIGNAL_NEURONS_NUMBER,
                reforming_neurons=dict_sum(self.REFORMING_NEURONS_STRUCTURE),
                writing_memory_neurons=writing_memory_neurons_number,
                reading_memory_neurons=reading_memory_neurons_number,
            )
        )
        super().__init__(
            [inputs_number, *middle_layers_structure, outputs_number],
        )

    def save(self, file: str):
        with open(file=file, mode='w', encoding='ascii') as filebuffer:
            dump(
                obj=dict(
                    signifying_inputs_number=self.signifying_inputs_number,
                    signifying_outputs_number=self.signifying_outputs_number,
                ),
                fp=filebuffer,
            )
        super().save(file=f'{file}.perceptron')

    @classmethod
    def load(cls, file: str):
        with open(file=file, mode='r', encoding='ascii') as filebuffer:
            dctnry = load(fp=filebuffer)
        perceptron = super().load(file=f'{file}.perceptron')

        recurrent = cls.__new__(cls)
        recurrent.layers = perceptron.layers
        recurrent\
            .signifying_inputs_number = dctnry['signifying_inputs_number']
        recurrent\
            .signifying_outputs_number = dctnry['signifying_outputs_number']
        return recurrent

    @property
    def reading_memory_cells_number(self) -> int:
        return self.structure[0]\
            - self.signifying_inputs_number\
            - self.TIME_INPUTS_NUMBER\
            - self.REFLECTIONS_INPUTS_NUMBER

    @property
    def writing_memory_cells_number(self) -> int:
        memory_cell_neurons_number = dict_sum(self.MEMORY_CELL_STRUCTURE)
        reading_memory_neurons_number = self.reading_memory_cells_number\
            * memory_cell_neurons_number
        writing_memory_neurons_number = self.structure[-1]\
            - self.signifying_outputs_number\
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
            signifying_outputs=self.signifying_outputs_number,
            signal_neurons=self.SIGNAL_NEURONS_NUMBER,
            reforming_neurons=dict_sum(self.REFORMING_NEURONS_STRUCTURE),
            writing_memory_neurons=writing_memory_neurons,
            reading_memory_neurons=reading_memory_neurons,
        )

    def _add_neuron(self, layer_adress: list[int]):
        neuron_inputs_number = self.structure[layer_number]
        self.layers[layer_number].add_neuron(neuron_inputs_number)
        # add weights for each neuron of the next layer
        self.layers[layer_number + 1].add_weights()

    def _delete_neuron(self, layer_adress: list[int], neuron_adress: list[int]):
        self.layers[layer_number].neurons.pop(index=neuron_number)

    def _write_weights(self, writing_memory: list[int]):
        for _ in range(self.writing_memory_cells_number):
            (
                layer_outputs,
                neuron_outputs,
                weight_outputs,
                writing_memory,

            ) = split_by_volumes(
                list_for_split=writing_memory,
                volumes=self.MEMORY_CELL_STRUCTURE.values(),
            )
            neurons = self.layers[get_index_by_fraction(
                sequence=self.layers,
                fraction=1 / conv_list_to_int(layer_outputs),
            )].neurons

            weights = neurons[get_index_by_fraction(
                sequence=neurons,
                fraction=1 / conv_list_to_int(neuron_outputs),
            )].weights

            element_index = get_index_by_fraction(
                sequence=weights,
                fraction=1 / conv_list_to_int(weight_outputs),
            )

            weights[element_index] = -weights[element_index]

    def _read_weights(self, reading_memory: list[int]):
        reading_memory_inputs_binary_list = list()
        for _ in range(self.reading_memory_cells_number):
            (
                layer_outputs,
                neuron_outputs,
                weight_outputs,
                reading_memory,

            ) = split_by_volumes(
                list_for_split=reading_memory,
                volumes=self.MEMORY_CELL_STRUCTURE.values(),
            )

            neurons = self.layers[get_index_by_fraction(
                sequence=self.layers,
                fraction=1 / conv_list_to_int(layer_outputs),
            )].neurons

            weights = neurons[get_index_by_fraction(
                sequence=neurons,
                fraction=1 / conv_list_to_int(neuron_outputs),
            )].weights

            element_index = get_index_by_fraction(
                sequence=weights,
                fraction=1 / conv_list_to_int(weight_outputs),
            )

            reading_memory_inputs_binary_list.append(weights[element_index])
        return reading_memory_inputs_binary_list

    def __call__(self, inputs: list[list], time_limit=None):
        # Start of timer
        if time_limit:
            start_time = time()

        # Save initial request for use it if reflections stopped
        initial_request = inputs

        reading_memory_inputs = [0,] * self.reading_memory_cells_number

        # Reflections
        while True:
            for reflection in range(2 ** self.REFLECTIONS_INPUTS_NUMBER):
                resoults = list()
                for inputs_values in inputs:
                    while True:  # don't breaking if signal is repeat_signal

                        # Convert time data to lists of binary signals
                        time_binary_list = conv_int_to_list(
                            number=int(time()),
                            width=self.TIME_INPUTS_NUMBER,
                        )
                        reflections_binary_list = conv_int_to_list(
                            number=reflection,
                            width=self.REFLECTIONS_INPUTS_NUMBER,
                        )

                        # Get outputs as list of binary signals
                        outputs = super().__call__(
                            [
                                *inputs_values,
                                *time_binary_list,
                                *reflections_binary_list,
                                *reading_memory_inputs,
                            ],
                        )

                        # Split binary list to valuable binary lists
                        (
                            signifying_outputs,
                            signal,
                            reforming,
                            writing_memory,
                            reading_memory,

                        ) = split_by_volumes(
                                list_for_split=outputs,
                                volumes=self.outputs_structure.values(),
                            )

                        # Write weights
                        self._write_weights(writing_memory)

                        # Read weights
                        reading_memory_inputs = self._read_weights(
                            reading_memory,
                        )

                        # Add or delete neuron
                        (
                            reforming_signal,
                            reforming_layer_adress,
                            reforming_neuron_adress,

                        ) = split_by_volumes(
                            list_for_split=reforming,
                            volumes=self.REFORMING_NEURONS_STRUCTURE.values(),
                        )

                        if reforming_signal == self.SIGNALS['ADD_NEURON_SIGNAL']:
                            self._add_neuron(
                                layer_adress=reforming_layer_adress,
                            )

                        elif reforming_signal == self.SIGNALS['DELETE_NEURON_SIGNAL']:
                            self._delete_neuron(
                                layer_adress=reforming_layer_adress,
                                neuron_adress=reforming_neuron_adress,
                            )

                        # Add or delete memory neurons
                        last_layer_index = len(perceptron.layers)
                        if add_delete_memory == add_memory_signal:
                            for _ in range(dict_sum(self.memory_cell_structure)):
                                perceptron.add_neuron(last_layer_index)
                        if add_delete_memory == delete_memory_signal and memory_cells_number()>1:
                            for _ in range(dict_sum(self.memory_cell_structure)):
                                perceptron.delete_neuron(last_layer_index)

                        # Convert binary signals list to character
                        if controlling_signal == skip_signal:
                            response_character = ''
                        else:
                            raw_index = conv_list_to_int(character_outputs)
                            maximal_index = len(self.coding_string) - 1
                            if raw_index > maximal_index:
                                response_character = ''
                            else:
                                response_character = self.coding_string[raw_index]

                        # Add character to list of resoults
                        resoults.append(response_character)

                        # Stop by time limit
                        if time_limit and time_limit < time() - start_time:
                            controlling_signal = stop_signal

                        # Stop repeating
                        if controlling_signal != repeat_signal:
                            break

                    # Stop iterations
                    if controlling_signal == stop_signal:
                        break
                    if reflection > 0 and controlling_signal == stop_reflections_signal:
                        break

                # Stop reflections
                if controlling_signal == stop_signal:
                    break

                # Prepare request for new reflection from  results
                request = ''.join(resoults)

                # Reset reflections and turn back to initial request
                if reflection > 0 and controlling_signal == stop_reflections_signal:
                    request = initial_request
                    reflection = 0
                elif request == '':
                    request = initial_request
                    reflection = 0
                else:
                    reflection += 1
        return ''.join(resoults)

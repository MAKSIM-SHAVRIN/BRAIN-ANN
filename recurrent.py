from json import dump, load
from time import time

from perceptron import Perceptron
from utils import (conv_int_to_list, conv_list_to_int, dict_sum,
                   get_element_by_adress, get_index_by_adress,
                   split_by_volumes)


class Recurrent(Perceptron):
    # Signals
    STOP_SIGNAL = [0, 0, 1],
    SKIP_SIGNAL = [0, 1, 0],
    REPEAT_SIGNAL = [1, 0, 0],
    STOP_REFLECTIONS_SIGNAL = [1, 1, 1],

    ADD_NEURON_SIGNAL = [1, 0, 0],
    DELETE_NEURON_SIGNAL = [0, 1, 1],

    ADD_WRITTING_MEMORY_SIGNAL = [0, 0, 1],
    DELETE_WRITTING_MEMORY_SIGNAL = [1, 1, 0],
    ADD_READING_MEMORY_SIGNAL = [1, 0, 1],
    DELETE_READING_MEMORY_SIGNAL = [0, 1, 0],

    # Net structure
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
        layer = self.layers[layer_index]
        # get inputs number from previous layer neurons number
        inputs_number = self.structure[layer_index]
        layer._add_neuron(neuron_inputs_number=inputs_number)
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
        index = self.signifying_outputs_number\
            + self.SIGNAL_NEURONS_NUMBER\
            + dict_sum(self.REFORMING_NEURONS_STRUCTURE)
        for _ in dict_sum(self.MEMORY_CELL_STRUCTURE.values()):
            last_layer._insert_neuron(index, inputs_number)

    def _delete_writing_memory_neurons(self):
        if self.writing_memory_cells_number:
            last_layer = self.layers[-1]
            index = self.signifying_outputs_number\
                + self.SIGNAL_NEURONS_NUMBER\
                + dict_sum(self.REFORMING_NEURONS_STRUCTURE)
            for _ in dict_sum(self.MEMORY_CELL_STRUCTURE.values()):
                last_layer.neurons.pop(index)

    def _add_reading_memory_neurons(self):
        last_layer = self.layers[-1]
        inputs_number = len(self.layers[-2].neurons)
        for _ in dict_sum(self.MEMORY_CELL_STRUCTURE.values()):
            last_layer._add_neuron(inputs_number)
        # add reading memory input
        self.layers[0]._add_weights()

    def _delete_reading_memory_neurons(self):
        if self.reading_memory_cells_number:
            last_layer = self.layers[-1]
            for _ in dict_sum(self.MEMORY_CELL_STRUCTURE.values()):
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

    def __call__(self, inputs: list[list], time_limit=None):
        # Start of timer
        if time_limit:
            start_time = time()

        # Save initial request for use it if reflections stopped
        initial_request = inputs

        # Fill initial reading_memory_inputs by zero values
        reading_memory_inputs = [0,] * self.reading_memory_cells_number

        # Reflections loop
        while True:
            for reflection in range(2 ** self.REFLECTIONS_INPUTS_NUMBER - 1):
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
                            controlling_signal,
                            transforming,
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
                        # Transforming
                        self._transform(transforming_outputs=transforming)

                        # Convert binary signals list to character
                        if controlling_signal == self.SKIP_SIGNAL:
                            signifying_outputs = []
                        else:
                            raw_index = conv_list_to_int(character_outputs)
                            maximal_index = len(self.coding_string) - 1
                            if raw_index > maximal_index:
                                signifying_outputs = []
                            else:
                                signifying_outputs = self.coding_string[raw_index]

                        # Add character to list of resoults
                        resoults.append(signifying_outputs)

                        # Stop by time limit
                        if time_limit and time_limit < time() - start_time:
                            controlling_signal = stop_signal

                        # Stop repeating
                        if controlling_signal != self.REPEAT_SIGNAL:
                            break

                    # Stop iterations
                    if controlling_signal == self.STOP_SIGNAL:
                        break
                    if reflection > 0 and controlling_signal == self.STOP_REFLECTIONS_SIGNAL:
                        break

                # Stop reflections
                if controlling_signal == self.STOP_SIGNAL:
                    break

                # Prepare request for new reflection from  results
                request = [].extend(signifying_outputs)

                # Reset reflections and turn back to initial request
                if reflection > 0 and controlling_signal == self.STOP_REFLECTIONS_SIGNAL:
                    request = initial_request
                    reflection = 0
                elif request == []:
                    request = initial_request
                    reflection = 0
                else:
                    reflection += 1
        return [].extend(signifying_outputs)

from json import dump, load
from pathlib import Path
from time import time

from numpy import array

from perceptron import Perceptron
from utils import (
    check_dir_path_slash_ending,
    dict_sum,
    get_element_by_decimal,
    get_index_by_decimal,
    split_by_volumes,
)


class Brain(Perceptron):
    # Structure
    INITIAL_WRITING_MEMORY_CELLS_NUMBER: int = 5
    INITIAL_MIDDLE_LAYERS_STRUCTURE: list = 6*[10]
    INITIAL_READING_MEMORY_CELLS_NUMBER: int = 5

    ADRESS_POWER: int = 1

    MEMORY_CELL_STRUCTURE = dict(
        layer_adress=ADRESS_POWER,
        neuron_adress=ADRESS_POWER,
        weight_adress=ADRESS_POWER,
        new_walue=1,
    )
    REFORMING_NEURONS_STRUCTURE = dict(
        signal_neurons_number=1,
        layer_adress=ADRESS_POWER,
        neuron_adress=ADRESS_POWER,
    )
    SIGNAL_NEURONS_NUMBER: int = 1

    TIME_INPUTS_NUMBER: int = 1
    TIME_LIMIT_INPUTS_NUMBER: int = 1
    REFLECTIONS_INPUTS_NUMBER: int = 1

    # Signals
    CONTROLLING_SIGNALS = [
        'NOTHING_SIGNAL',
        'STOP_SIGNAL',
        'SKIP_SIGNAL',
        'REPEAT_SIGNAL',
        'STOP_REFLECTIONS_SIGNAL',
    ]

    TRANSFORMING_SIGNALS = [
        'NOTHING_SIGNAL',

        'ADD_NEURON_SIGNAL',
        'DELETE_NEURON_SIGNAL',

        'ADD_WRITTING_MEMORY_SIGNAL',
        'DELETE_WRITTING_MEMORY_SIGNAL',

        'ADD_READING_MEMORY_SIGNAL',
        'DELETE_READING_MEMORY_SIGNAL',
    ]

    # Methods
    def __init__(self, inputs_number: int = 1, outputs_number: int = 1):
        # Nested functions
        def _initial_inputs_number() -> int:
            perceptron_inputs_number = dict_sum(
                dict(
                    signifying=self.inputs_number,
                    time=self.TIME_INPUTS_NUMBER,
                    time_limit=self.TIME_LIMIT_INPUTS_NUMBER,
                    reflections_counter=self.REFLECTIONS_INPUTS_NUMBER,
                    reading_memory=self.INITIAL_READING_MEMORY_CELLS_NUMBER,
                ),
            )
            return perceptron_inputs_number

        def _initial_outputs_number() -> int:
            memory_cell_neurons_number = dict_sum(self.MEMORY_CELL_STRUCTURE)
            w_memory_neurons_number = self.INITIAL_WRITING_MEMORY_CELLS_NUMBER\
                * memory_cell_neurons_number
            r_memory_neurons_number = self.INITIAL_READING_MEMORY_CELLS_NUMBER\
                * memory_cell_neurons_number

            perceptron_outputs_number = dict_sum(
                dict(
                    signifying=self.outputs_number,
                    signal=self.SIGNAL_NEURONS_NUMBER,
                    reforming=dict_sum(self.REFORMING_NEURONS_STRUCTURE),
                    writing_memory=w_memory_neurons_number,
                    reading_memory=r_memory_neurons_number,
                ),
            )
            return perceptron_outputs_number

        def _initial_perceptron_structure():
            structure = [
                _initial_inputs_number(),
                *self.INITIAL_MIDDLE_LAYERS_STRUCTURE,
                _initial_outputs_number(),
            ]
            return structure

        ################################################################
        if inputs_number < 1:
            raise ValueError('Recurrent must have at least one input')
        if outputs_number < 1:
            raise ValueError('Recurrent must have at least one output')

        self.inputs_number = inputs_number
        self.outputs_number = outputs_number

        super().__init__(_initial_perceptron_structure())
        return None  # Added for visual end of the method

    def __call__(
        self, inputs: list[list[float]],
        time_limit: float = -1, steps_limit: int = None,
        transform=True, introspect=True,
        just_last_resoult=False, do_not_skip_and_repeat=False,
        verbalize=False,
    ) -> list[list[float]]:

        # Nested functions
        def verb(*args, **kwargs):
            if verbalize:
                print(*args, **kwargs)

        def _transform(transforming_outputs: list[float]):

            # Nested functions
            def _add_neuron(layer_adress: float):
                index = get_index_by_decimal(self.layers[:-1], layer_adress)
                layer = self.layers[index]
                layer._add_neuron()
                # Add due weights of next layer
                next_layer = self.layers[index + 1]
                next_layer._add_weights()

            def _delete_neuron(layer_adress: float, neuron_adress: float):
                index = get_index_by_decimal(self.layers[:-1], layer_adress)
                layer = self.layers[index]
                neurons_number = layer.neurons_number
                neuron_index = get_index_by_decimal(
                    sequence=list(range(neurons_number - 1)),
                    decimal=neuron_adress,
                )
                if neurons_number > 1:
                    layer._delete_neuron(neuron_index)
                    # Delete due weights of next layer
                    next_layer = self.layers[index + 1]
                    # `1 + neuron_index` cuz remember about bias weight
                    next_layer._delete_weights(1 + neuron_index)

            def _get_last_writting_memory_index():
                writting_memory_neurons = dict_sum(self.MEMORY_CELL_STRUCTURE)\
                    * self.writing_memory_cells_number
                last_writting_memory_index = dict_sum(
                    dict(
                        signifying=self.outputs_number,
                        signal=self.SIGNAL_NEURONS_NUMBER,
                        reforming=dict_sum(self.REFORMING_NEURONS_STRUCTURE),
                        writing_memory=writting_memory_neurons,
                    ),
                )
                return last_writting_memory_index

            def _add_writing_memory_neurons():
                index = _get_last_writting_memory_index()
                for number in range(dict_sum(self.MEMORY_CELL_STRUCTURE)):
                    self.layers[-1]._insert_neuron(number + index)

            def _delete_writing_memory_neurons():
                index = _get_last_writting_memory_index()
                if self.writing_memory_cells_number > 1:
                    for number in range(dict_sum(self.MEMORY_CELL_STRUCTURE)):
                        self.layers[-1]._delete_neuron(index - number - 1)

            ############################################################
            transform_signal, layer_adress, neuron_adress = split_by_volumes(
                list_for_split=transforming_outputs,
                volumes=self.REFORMING_NEURONS_STRUCTURE.values(),
                get_rest=False,
            )
            signal = get_element_by_decimal(
                self.TRANSFORMING_SIGNALS,
                transform_signal[0],
            )
            verb(f'Transforming signal: {signal}')

            # Add or delete neuron
            if signal == 'ADD_NEURON_SIGNAL':
                _add_neuron(layer_adress[0])

            elif signal == 'DELETE_NEURON_SIGNAL':
                _delete_neuron(layer_adress[0], neuron_adress[0])

            # Add or delete writting memory neurons
            elif signal == 'ADD_WRITTING_MEMORY_SIGNAL':
                _add_writing_memory_neurons()

            elif signal == 'DELETE_WRITTING_MEMORY_SIGNAL':
                _delete_writing_memory_neurons()

        def _reading_memory_transform(transforming_outputs: list[float]):
            # Nested functions
            def _add_reading_memory_neurons():
                for _ in range(dict_sum(self.MEMORY_CELL_STRUCTURE)):
                    self.layers[-1]._add_neuron()
                self.layers[0]._add_weights()

            def _delete_reading_memory_neurons():
                if self.reading_memory_cells_number > 1:
                    for _ in range(dict_sum(self.MEMORY_CELL_STRUCTURE)):
                        self.layers[-1]._delete_last_neuron()
                    self.layers[0]._delete_last_weights()

            ############################################################
            transform_signal, layer_adress, neuron_adress = split_by_volumes(
                list_for_split=transforming_outputs,
                volumes=self.REFORMING_NEURONS_STRUCTURE.values(),
                get_rest=False,
            )
            signal = get_element_by_decimal(
                self.TRANSFORMING_SIGNALS,
                transform_signal[0],
            )
            # Add or delete reading memory neurons
            if signal == 'ADD_READING_MEMORY_SIGNAL':
                _add_reading_memory_neurons()

            elif signal == 'DELETE_READING_MEMORY_SIGNAL':
                _delete_reading_memory_neurons()

        def _introspect(
            writting_memory: list[float], reading_memory: list[float],
        ) -> list[float]:

            # Nested functions
            def _write_weights(writing_memory: list[float]):
                for _ in range(self.writing_memory_cells_number):
                    (
                        layer_adress,
                        neuron_adress,
                        weight_adress,
                        new_walue,

                        writing_memory,

                    ) = split_by_volumes(
                        list_for_split=writing_memory,
                        volumes=self.MEMORY_CELL_STRUCTURE.values(),
                    )
                    layer = get_element_by_decimal(
                        sequence=self.layers, decimal=layer_adress[0],
                    )
                    neuron_index = get_index_by_decimal(
                        sequence=list(range(layer.neurons_number - 1)),
                        decimal=neuron_adress[0]
                    )
                    weight_index = get_index_by_decimal(
                        sequence=list(
                            range(layer.each_neuron_weights_number - 1),
                        ),
                        decimal=weight_adress[0],
                    )
                    layer._change_weight(
                        neuron_index=neuron_index,
                        weight_index=weight_index,
                        new_walue=new_walue[0],
                    )

            def _read_weights(reading_memory: list[float]) -> list[float]:
                reading_memory_inputs = list()
                for _ in range(self.reading_memory_cells_number):
                    (
                        layer_adress,
                        neuron_adress,
                        weight_adress,
                        new_walue,

                        reading_memory,

                    ) = split_by_volumes(
                        list_for_split=reading_memory,
                        volumes=self.MEMORY_CELL_STRUCTURE.values(),
                    )
                    layer = get_element_by_decimal(
                        sequence=self.layers, decimal=layer_adress[0],
                    )
                    neuron_index = get_index_by_decimal(
                        sequence=list(range(layer.neurons_number - 1)),
                        decimal=neuron_adress[0]
                    )
                    weight_index = get_index_by_decimal(
                        sequence=list(
                            range(layer.each_neuron_weights_number - 1),
                        ),
                        decimal=weight_adress[0],
                    )
                    reading_memory_inputs.append(
                        layer._read_weight(neuron_index, weight_index),
                    )
                return reading_memory_inputs

            ############################################################
            # Read weights
            reading_memory_inputs = _read_weights(reading_memory)
            verb('MEMORY IS READ')
            # Write weights
            _write_weights(writting_memory)
            verb('MEMORY IS WRITTEN')
            return reading_memory_inputs

        ################################################################
        # Start of timer
        if time_limit != -1:
            start_time = time()

        # Start of steps counting
        if steps_limit:
            steps_counter = 0

        # initial controlling signal is always do nothing
        controlling_signal = 'NOTHING_SIGNAL'

        # Save initial request for use it if reflections stopped
        initial_inputs = inputs

        # Fill initial reading_memory_inputs by zero values
        reading_memory_inputs = [0,] * self.reading_memory_cells_number

        # Reflections loop
        while True:
            inputs = initial_inputs

            # Reflections
            reflections_counter = 0
            while True:
                resoults = list()
                # Iterations
                verb(f'Feflection number: {reflections_counter}')
                for inputs_values in inputs:
                    if not do_not_skip_and_repeat:
                        if controlling_signal == 'SKIP_SIGNAL':
                            controlling_signal = 'NOTHING_SIGNAL'
                            continue
                    # Repeating
                    while True:
                        # Stop by time limit
                        if time_limit != -1:
                            if time_limit < time() - start_time:
                                controlling_signal = 'STOP_SIGNAL'
                                verb('STOPPED BY TIME LIMIT')
                                break

                        # Stop by steps limit
                        if steps_limit and steps_limit < steps_counter:
                            controlling_signal = 'STOP_SIGNAL'
                            verb('STOPPED BY STEP LIMIT')
                            break

                        # Get outputs as list of binary signals and
                        # Split binary list to valuable binary lists
                        inputs_list = [
                            *inputs_values,
                            time(),
                            time_limit,
                            reflections_counter,
                            *reading_memory_inputs,
                        ]
                        (
                            signifying_outputs,
                            controlling,
                            transforming,
                            writting_memory,
                            reading_memory,

                        ) = split_by_volumes(
                            list_for_split=super().__call__(array(inputs_list)),
                            volumes=self.outputs_structure.values(),
                            get_rest=False,
                        )

                        # Get controlling signal
                        controlling_signal = get_element_by_decimal(
                            self.CONTROLLING_SIGNALS, controlling[0],
                        )
                        verb(f'Controlling signal: {controlling_signal}')

                        # Reading memory transforming
                        if transform:
                            _reading_memory_transform(transforming)

                        # Introspecton
                        if introspect:
                            reading_memory_inputs = _introspect(
                                writting_memory, reading_memory,
                            )

                        # Transforming
                        if transform:
                            _transform(transforming)

                        # Add character to list of resoults
                        if just_last_resoult:
                            resoults = [signifying_outputs,]
                        else:
                            resoults.append(signifying_outputs)

                        # Increase steps counter
                        if steps_limit:
                            steps_counter += 1

                        # Stop repeating
                        if do_not_skip_and_repeat:
                            break
                        if controlling_signal != 'REPEAT_SIGNAL':
                            break

                    # Stop iterations
                    if controlling_signal == 'STOP_SIGNAL':
                        break
                    elif controlling_signal == 'STOP_REFLECTIONS_SIGNAL':
                        break

                # Stop reflection
                if controlling_signal == 'STOP_SIGNAL':
                    break
                elif controlling_signal == 'STOP_REFLECTIONS_SIGNAL':
                    break
                # Resoults are empty or not enough for the next reflection
                elif self.inputs_number != self.outputs_number:
                    break
                elif resoults == list():
                    break
                elif just_last_resoult:
                    break

                # Prepare request for new reflection from results
                inputs = resoults
                reflections_counter += 1

            # Stop reflections loop
            if controlling_signal != 'STOP_REFLECTIONS_SIGNAL':
                break
        return resoults

    @property
    def reading_memory_cells_number(self) -> int:
        return self.structure[0]\
            - self.inputs_number\
            - self.TIME_INPUTS_NUMBER\
            - self.TIME_LIMIT_INPUTS_NUMBER\
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

    def __repr__(self) -> str:
        repr_string = '< Brain:\n'\
            + f'structure: {self.structure}\n'\
            + f'{self.inputs_number} inputs\n'\
            + f'{self.outputs_number} outputs\n'\
            + f'{self.reading_memory_cells_number} reading memory cells\n'\
            + f'{self.writing_memory_cells_number} writting memory cells\n'\
            + '>'
        return repr_string

    def __eq__(self, o):
        return self.layers == o.layers\
            and self.inputs_number == o.inputs_number\
            and self.outputs_number == o.outputs_number


# Testing
if __name__ == '__main__':
    class BigBrain(Brain):
        INITIAL_MIDDLE_LAYERS_STRUCTURE = 6 * [10 ** 3,]

    print(BigBrain()([[789], [7], [8], [9], [1], [0], [5], [6]], verbalize=True))

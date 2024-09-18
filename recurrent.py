from time import time
from typing import Callable, Iterable
from types import NoneType
from json import dump, load
from pathlib import Path

from numpy import array
from numpy.typing import NDArray

from perceptron import Perceptron
from memory import ReadingMemory, WritingMemory
from utils import (
    check_dir_path_slash_ending,
    dict_sum, split_by_volumes,
    get_index_by_decimal,
    get_element_by_decimal,
)


class Brain(Perceptron):
    # Structure
    I_ML_S = INITIAL_MIDDLE_LAYERS_STRUCTURE = 6*[10]

    # Dictkeys are a comments-like to better understand brain structure
    T_O_BS = TRANSFORMING_OUTPUTS_BLOCK_STRUCTURE = dict(
        transforming_signal_ouputs_number=1,
        layer_adress_ouputs_number=1,
        output_adress_ouputs_number=1,
    )
    TES_I_N = TRANSFORMING_ERROR_SIGNAL_INPUTS_NUMBER = 1

    CS_O_N = CONTROLLING_SIGNAL_OUTPUTS_NUMBER = 1

    T_I_N = TIME_INPUTS_NUMBER = 1
    TL_I_N = TIME_LIMIT_INPUTS_NUMBER = 1

    R_I_N = REFLECTIONS_INPUTS_NUMBER = 1
    RL_I_N = REFLECTIONS_LIMIT_INPUTS_NUMBER = 1

    S_I_N = STEPS_INPUTS_NUMBER = 1
    SL_I_N = STEPS_LIMIT_INPUTS_NUMBER = 1

    # Signals
    CONTROLLING_SIGNALS = [
        'NOTHING',
        'SKIP',
        'REPEAT',
        'RESET_REFLECTIONS',
        'STOP',
    ]

    TRANSFORMING_SIGNALS = [
        'NOTHING',

        'APPEND_OUTPUT',
        'DELETE_OUTPUT',

        'APPEND_WRITING_MEMORY_BLOCK',
        'POP_WRITING_MEMORY_BLOCK',

        'APPEND_READING_MEMORY_BLOCK',
        'POP_READING_MEMORY_BLOCK',
    ]

    def verb(self, *args, **kwargs):
        if self._verbalize:
            print(*args, **kwargs)
        return None

    def if_transform(self, method: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # First argument is usualy class` instance
            if self.brain._transform:
                method(*args, **kwargs)
        return wrapper

    def if_introspect(self, method: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # First argument is usualy class` instance
            if self.brain._introspect:
                return method(*args, **kwargs)
        return wrapper

    def catch_error(self, method: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # First argument is usualy class` instance
            try:
                method(*args, **kwargs)
            except RuntimeError:
                self.brain._transforming_error_flag = 1
            else:
                self.brain._transforming_error_flag = 0
        return wrapper

    @property
    def initial_perceptron_inputs_number(self) -> int:
        perceptron_inputs_number = dict_sum(
            # Dictkeys are a comments-like...
            # ...to better understand brain structure
            dict(
                signifying_inputs_number=self.inputs_number,

                time_inputs_number=self.TIME_INPUTS_NUMBER,
                time_limit_inputs_number=self.TIME_LIMIT_INPUTS_NUMBER,

                transforming_error_signal_inpts_number=self.TES_I_N,

                reflections_counter_inputs_number=self.R_I_N,
                reflections_limit_inputs_number=self.RL_I_N,

                steps_counter_inputs_number=self.STEPS_INPUTS_NUMBER,
                steps_limit_inputs_number=self.STEPS_LIMIT_INPUTS_NUMBER,

                reading_memory_inputs_number=self.reading_memory.I_BN,
            ),
        )
        return perceptron_inputs_number

    @property
    def initial_perceptron_outputs_number(self) -> int:
        perceptron_outputs_number = dict_sum(
            # Dictkeys are a comments-like...
            # ...to better understand brain structure
            dict(
                signifying_outputs_number=self.outputs_number,

                controlling_signal_outputs_number=self.CS_O_N,

                transforming_outputs_number=dict_sum(self.T_O_BS),

                writing_memory_outputs_number=self.writing_memory.I_ON,
                reading_memory_outputs_number=self.reading_memory.I_ON,
            ),
        )
        return perceptron_outputs_number

    @property
    def initial_perceptron_structure(self) -> list[int]:
        perceptron_structure = [
            self.initial_perceptron_inputs_number,
            *self.INITIAL_MIDDLE_LAYERS_STRUCTURE,
            self.initial_perceptron_outputs_number,
        ]
        return perceptron_structure

    # Methods
    def __init__(self, inputs_number: int = 1, outputs_number: int = 1):
        self.brain = self

        self.inputs_number = inputs_number
        self.outputs_number = outputs_number

        self.reading_memory = ReadingMemory(brain=self)
        self.writing_memory = WritingMemory(brain=self)

        # decorating of methods by decorators from Brain class
        self.append_output_to_layer = self\
            .if_transform(self.append_output_to_layer)

        self.delete_output_from_layer = self\
            .catch_error(self.if_transform(self.delete_output_from_layer))

        super().__init__(self.initial_perceptron_structure)
        return None  # Added for visual end of the method

    def append_output_to_layer(self, layer_adress: float):
        # Can not append outputs to last layer...
        # ...cuz it has its own special structure
        layers_excluding_last = self.layers[:-1]
        index = get_index_by_decimal(layers_excluding_last, layer_adress)
        layer = self.layers[index]

        layer.append_output()
        # Add due inputs of next layer
        next_layer = self.layers[index + 1]
        next_layer.append_input()
        return None

    def delete_output_from_layer(
        self, layer_adress: float, output_adress: float,
    ):
        # Can not delete outputs from last layer...
        # ...cuz it has its own special structure
        layers_excluding_last = self.layers[:-1]

        index = get_index_by_decimal(layers_excluding_last, layer_adress)
        layer = self.layers[index]

        outputs_number = layer.outputs_number

        if outputs_number == 1:
            raise RuntimeError('Can not delete only output of layer')
        output_index = get_index_by_decimal(
            sequence=list(range(outputs_number)),
            decimal=output_adress,
        )
        layer.delete_output(output_index)

        # Delete due weights of next layer
        next_layer = self.layers[index + 1]
        next_layer.delete_input(output_index)
        return None

    def verbalize_input_values(
        self, signifying_inputs_values, passed_time, time_limit,
        reflections_counter, reflections_limit, steps_counter, steps_limit,
    ):
        self.verb(f'\nCURRENT INPUTS: {signifying_inputs_values}')

        self.verb(f'PASSED TIME: {passed_time}')
        self.verb(f'TIME LIMIT: {time_limit}')

        self.verb(
            'TRASFORMATION ERROR ON PREVIOUS STEP: ',
            self._transforming_error_flag,
        )
        self.verb(f'REFLECTION NUMBER: {reflections_counter}')
        self.verb(f'REFLECTIONS LIMIT: {reflections_limit}')

        self.verb(f'STEPS NUMBER: {steps_counter}')
        self.verb(f'STEPS LIMIT: {steps_limit}')

        return None

    def transform(
        self, transforming_outputs_values: list[float] | NDArray[float],
    ):
        (
            transform_signal_output_value,
            layer_adress_output_value,
            output_adress_output_value,

        ) = split_by_volumes(
            list_for_split=transforming_outputs_values,
            volumes=self.T_O_BS.values(),
            get_rest=False,
        )
        transforming_signal = get_element_by_decimal(
            self.TRANSFORMING_SIGNALS,
            transform_signal_output_value,
        )
        self.verb(f'TRANSFORMING SIGNAL: {transforming_signal}')

        # Add or delete reading memory neurons
        if transforming_signal == 'APPEND_READING_MEMORY_BLOCK':
            self.reading_memory.append_block()

        elif transforming_signal == 'POP_READING_MEMORY_BLOCK':
            self.reading_memory.pop_block()

        # Add or delete neuron
        if transforming_signal == 'APPEND_OUTPUT':
            self.append_output_to_layer(layer_adress_output_value)

        elif transforming_signal == 'DELETE_OUTPUT':
            self.delete_output_from_layer(
                layer_adress_output_value,
                output_adress_output_value,
            )

        # Add or delete writing memory neurons
        elif transforming_signal == 'APPEND_WRITING_MEMORY_BLOCK':
            self.writing_memory.append_block()

        elif transforming_signal == 'POP_WRITING_MEMORY_BLOCK':
            self.writing_memory.pop_block()

        elif transforming_signal == 'NOTHING':
            self._transforming_error_flag = 0

        return None

    def turn_Nones_negative_ones(self, iterable: list) -> NDArray[float]:
        resoult = list()
        for element in iterable:
            if element is None:
                resoult.append(-1)
                continue
            resoult.append(element)
        return array(resoult)

    def do_one_step(
        self, signifying_inputs_values, passed_time, time_limit,
        reflections_counter, reflections_limit, steps_counter, steps_limit,
    ):
        self.verbalize_input_values(
            signifying_inputs_values,
            passed_time, time_limit,
            reflections_counter, reflections_limit,
            steps_counter, steps_limit,
        )

        inputs_values_list = self.turn_Nones_negative_ones(
            [
                *signifying_inputs_values,
                passed_time, time_limit,
                self._transforming_error_flag,
                reflections_counter, reflections_limit,
                steps_counter, steps_limit,
                *self._rmi_values,
            ]
        )
        (
            signifying_outputs_values,
            controlling_signal_outputs_values,
            transforming_outputs_values,
            writing_memory_outputs_values,
            reading_memory_outputs_values,

        ) = split_by_volumes(
            list_for_split=super().__call__(
                array(inputs_values_list),
            ),
            volumes=self.outputs_structure.values(),
            extract_single_values=False,
            get_rest=False,
        )

        # Get controlling signal
        controlling_signal = get_element_by_decimal(
            self.CONTROLLING_SIGNALS,
            controlling_signal_outputs_values[-1],
        )
        self.verb(f'CONTROLLING SIGNAL: {controlling_signal}')

        # Introspection
        self.writing_memory\
            .write_weights(writing_memory_outputs_values)

        self._rmi_values = self.reading_memory\
            .read_weights(reading_memory_outputs_values)

        # Transforming
        self.transform(transforming_outputs_values)

        return signifying_outputs_values, controlling_signal

    def __call__(
        self, input_values: Iterable,
        time_limit: int | float = 60, steps_limit: int | NoneType = None,
        reflections_limit: int = 7, transform=True, introspect=True,
        verbalize=False,
    ) -> list[NDArray[float]] | list:
        """
        The method performs calculations
        using the recurrent generative neural network

        Args:
            input_values (Iterable):
            Input iterable by axis=1 must provide due number of cells
            to number of signifuing inputs of the net to provide reflections.

            time_limit (int | float | NoneType):
            Ensures that the calculation stops
            after the specified time has passed,
            without waiting a stopping signal,
            Positive,
            None for unlimited steps.
            Defaults to 60.

            steps_limit (int | NoneType):
            Ensures that the calculation stops
            after the specified number of steps has passed -
            single forvard propogations within the perceptron,
            Positive,
            None for unlimited steps.
            Defaults to None.

            reflections_limit (int | NoneType):
            Limitation on the number of reflection levels,
            i.e. recursion iterations,
            Warning - it doesen't stop after reaching of limit,
            turns back to reflection number 0 instead,
            Positive,
            None for unlimited reflections.
            Defaults to 7.

            transform (bool):
            Allows the network to change its own structure,
            thus changing its efficiency and its memory mechanisms.
            Defaults to True.

            introspect (bool):
            Allows the network to change its own weights,
            thus providing memorization and zero-shot learning.
            Defaults to True.

            verbalize (bool):
            Output data about the calculation process to the console.
            Defaults to False.

        Raises:
            RuntimeError: if it is impossible delede an input
            or a writing/reading memory outputs block

        Returns:
            list[NDArray[float]] | list[]: a resoulting list
        """
        stop_reset = ['STOP', 'RESET_REFLECTIONS', 'STOP_BY_LIMIT']

        # Atributes for working of decorators
        self._transforming_error_flag = 0

        self._transform: bool = transform
        self._introspect: bool = introspect
        self.__class__._verbalize: bool = verbalize

        # Start of timer
        start_time: float = time()

        # Start of steps counting
        steps_counter = 0

        # initial controlling signal is always do nothing
        controlling_signal = 'NOTHING'

        # Fill initial reading_memory_inputs_values by zero values
        self._rmi_values: list[int] | list[float] = [0,]\
            * self.reading_memory.blocks_number

        # Reflections loop
        while True:
            resoults = list()
            signifying_inputs_values_sqnce: Iterable = input_values

            # Reflections
            reflections_counter: int = 0
            while True:
                if reflections_limit:
                    if reflections_counter >= reflections_limit:
                        self.verb(
                            f'REFLECTIONS LIMIT {reflections_limit}',
                            'IS REACHED: RESET REFLECTIONS',
                        )
                        controlling_signal = 'RESET_REFLECTIONS'
                        break

                # Iterations
                resoults = list()
                for signifying_inputs_values in signifying_inputs_values_sqnce:
                    if controlling_signal == 'SKIP':
                        controlling_signal = 'NOTHING'
                        self.verb('SKIPPED')
                        continue

                    # Repeating
                    while True:
                        # Get passed time
                        passed_time: float = time() - start_time

                        (
                            signifying_outputs_values,
                            controlling_signal,

                        ) = self.do_one_step(
                            signifying_inputs_values,
                            passed_time, time_limit,
                            reflections_counter, reflections_limit,
                            steps_counter, steps_limit,
                        )

                        # Add character to list of resoults
                        resoults.append(signifying_outputs_values)
                        self.verb(f'RESOULTS: {resoults}')

                        # Increase steps counter
                        steps_counter += 1

                        # Stop by time limit
                        if time_limit and time_limit < passed_time:
                            controlling_signal = 'STOP_BY_LIMIT'
                            self.verb('STOPPED BY TIME LIMIT')
                            break

                        # Stop by steps limit
                        if steps_limit and steps_limit < steps_counter:
                            controlling_signal = 'STOP_BY_LIMIT'
                            self.verb('STOPPED BY STEP LIMIT')
                            break

                        # Stop repeating
                        if controlling_signal != 'REPEAT':
                            break
                        else:
                            self.verb('REPEATING OF STEP')

                    # Stop iterations
                    if controlling_signal in stop_reset:
                        self.verb('CONTROLLING SIGNAL IS ')
                        self.verb(f'"{controlling_signal}"')
                        self.verb('SO NEXT ITERATION IS IMPOSIBLE')
                        break

                    self.verb('NEXT ITERATION')

                # Stop reflection
                if controlling_signal in stop_reset:
                    self.verb('CONTROLLING SIGNAL IS ')
                    self.verb(f'"{controlling_signal}"')
                    self.verb('SO NEXT REFLECTION IS IMPOSIBLE')
                    break

                # Resoults are empty or not enough for the next reflect
                elif self.inputs_number != self.outputs_number:
                    self.verb('INPUTS AND OUTPUTS ARE INCOMPLETABLE')
                    self.verb('SO NEXT REFLECTION IS IMPOSIBLE')
                    break
                elif resoults == list():
                    self.verb('RESOULTS ARE EMPTY')
                    self.verb('SO NEXT REFLECTION IS IMPOSIBLE')
                    break

                # Prepare request for new reflection from results
                signifying_inputs_values_sqnce = resoults
                reflections_counter += 1
                self.verb('NEXT REFLECTION')

            # Stop reflections loop
            if controlling_signal != 'RESET_REFLECTIONS':
                self.verb('THE END.\n')
                break
            self.verb('REFLECTIONS ARE RESET')
        return resoults

    def calculate_with_due_resoults_length(
        self, input_values: Iterable,
        time_limit: int | float = 60, steps_limit: int | NoneType = None,
        reflections_limit: int = 7, transform=True, introspect=True,
        verbalize=False,
    ) -> list[NDArray[float]] | list:

        stop_reset = ['STOP', 'RESET_REFLECTIONS', 'STOP_BY_LIMIT']

        # Atributes for working of decorators
        self._transforming_error_flag = 0

        self._transform: bool = transform
        self._introspect: bool = introspect
        self.__class__._verbalize: bool = verbalize

        # Start of timer
        start_time: float = time()

        # Start of steps counting
        steps_counter = 0

        # initial controlling signal is always do nothing
        controlling_signal = 'NOTHING'

        # Fill initial reading_memory_inputs_values by zero values
        self._rmi_values: list[int] | list[float] = [0,]\
            * self.reading_memory.blocks_number

        # Reflections loop
        while True:
            resoults = list()
            signifying_inputs_values_sqnce: Iterable = input_values

            # Reflections
            reflections_counter = 0
            while True:
                if reflections_limit:
                    if reflections_counter >= reflections_limit:
                        self.verb(
                            f'REFLECTIONS LIMIT {reflections_limit}',
                            'IS REACHED: RESET REFLECTIONS',
                        )
                        controlling_signal = 'RESET_REFLECTIONS'
                        break

                # Iterations
                resoults = list()
                for signifying_inputs_values in signifying_inputs_values_sqnce:
                    # Get passed time
                    passed_time: float = time() - start_time

                    (
                        signifying_outputs_values,
                        controlling_signal,

                    ) = self.do_one_step(
                        signifying_inputs_values,
                        passed_time, time_limit,
                        reflections_counter, reflections_limit,
                        steps_counter, steps_limit,
                    )

                    if controlling_signal in ['SKIP', 'REPEAT']:
                        self.verb('SIGNAL OPERATION IS IMPOSSIBLE')
                        self.verb('CUZ DUE RESOULTS LENGTH IS ACTIVE')

                    # Add character to list of resoults
                    resoults.append(signifying_outputs_values)
                    self.verb(f'RESOULTS: {resoults}')

                    # Increase steps counter
                    steps_counter += 1

                    # Stop iterations
                    are_same_length = len(resoults) == len(input_values)

                    # Stop by time limit
                    if time_limit != -1 and time_limit < passed_time:
                        if are_same_length:
                            controlling_signal = 'STOP_BY_LIMIT'
                            self.verb('STOPPED BY TIME LIMIT')
                            break
                        self.verb('RESOULT AND INPUT AREN`T SAME LENGTH')
                        self.verb('CAN`T BE STOPPED BY TIME LIMIT')

                    # Stop by steps limit
                    elif steps_limit != -1 and steps_limit < steps_counter:
                        if are_same_length:
                            controlling_signal = 'STOP_BY_LIMIT'
                            self.verb('STOPPED BY STEP LIMIT')
                            break
                        self.verb('RESOULT AND INPUT AREN`T SAME LENGTH')
                        self.verb('CAN`T BE STOPPED BY STEP LIMIT')

                    # Stop by signals
                    elif controlling_signal in stop_reset:
                        if are_same_length:
                            self.verb(
                                'CONTROLLING SIGNAL IS',
                                f'"{controlling_signal}"',
                            )
                            self.verb('SO NEXT ITERATION IS IMPOSIBLE')
                            break
                        self.verb('RESOULT AND INPUT AREN`T SAME LENGTH')
                        self.verb('CAN`T BE STOPPED')

                    self.verb('NEXT ITERATION')

                # Stop reflection
                if controlling_signal in stop_reset:
                    self.verb(f'CONTROLLING SIGNAL IS "{controlling_signal}"')
                    self.verb('SO NEXT REFLECTION IS IMPOSIBLE')
                    break

                # Resoults are empty or not enough for the next reflect
                elif self.inputs_number != self.outputs_number:
                    self.verb('INPUTS AND OUTPUTS ARE INCOMPLETABLE')
                    self.verb('SO NEXT REFLECTION IS IMPOSIBLE')
                    break
                elif resoults == list():
                    self.verb('RESOULTS ARE EMPTY')
                    self.verb('SO NEXT REFLECTION IS IMPOSIBLE')
                    break

                # Prepare request for new reflection from results
                signifying_inputs_values_sqnce = resoults
                reflections_counter += 1
                self.verb('NEXT REFLECTION')

            # Stop reflections loop
            if controlling_signal != 'RESET_REFLECTIONS':
                self.verb('THE END.\n')
                break
            self.verb('REFLECTIONS ARE RESET')
        return resoults

    def calculate_with_just_last_resoult(
        self, input_values: Iterable,
        time_limit: int | float = 60, steps_limit: int | NoneType = None,
        reflections_limit: int = 7, transform=True, introspect=True,
        verbalize=False,
    ) -> list[NDArray[float]] | list:
        stop_reset = ['STOP', 'RESET_REFLECTIONS', 'STOP_BY_LIMIT']

        # Atributes for working of decorators
        self._transforming_error_flag = 0

        self._transform: bool = transform
        self._introspect: bool = introspect
        self.__class__._verbalize: bool = verbalize

        # Start of timer
        start_time: float = time()

        # Start of steps counting
        steps_counter = 0

        # initial controlling signal is always do nothing
        controlling_signal = 'NOTHING'

        # Fill initial reading_memory_inputs_values by zero values
        self._rmi_values: list[int] | list[float] = [0,]\
            * self.reading_memory.blocks_number

        # Reflections loop
        while True:
            resoults = list()
            signifying_inputs_values_sqnce: Iterable = input_values

            # Reflections
            reflections_counter: int = 0
            while True:
                if reflections_limit:
                    if reflections_counter >= reflections_limit:
                        self.verb(
                            f'REFLECTIONS LIMIT {reflections_limit}',
                            'IS REACHED: RESET REFLECTIONS',
                        )
                        controlling_signal = 'RESET_REFLECTIONS'
                        break

                # Iterations
                resoults = list()
                for signifying_inputs_values in signifying_inputs_values_sqnce:
                    if controlling_signal == 'SKIP':
                        controlling_signal = 'NOTHING'
                        self.verb('SKIPPED')
                        continue

                    # Repeating
                    while True:
                        # Get passed time
                        passed_time: float = time() - start_time

                        (
                            signifying_outputs_values,
                            controlling_signal,

                        ) = self.do_one_step(
                            signifying_inputs_values,
                            passed_time, time_limit,
                            reflections_counter, reflections_limit,
                            steps_counter, steps_limit,
                        )

                        # Add character to list of resoults
                        resoults = [signifying_outputs_values]
                        self.verb(f'RESOULTS: {resoults}')

                        # Increase steps counter
                        steps_counter += 1

                        # Stop by time limit
                        if time_limit and time_limit < passed_time:
                            controlling_signal = 'STOP_BY_LIMIT'
                            self.verb('STOPPED BY TIME LIMIT')
                            break

                        # Stop by steps limit
                        if steps_limit and steps_limit < steps_counter:
                            controlling_signal = 'STOP_BY_LIMIT'
                            self.verb('STOPPED BY STEP LIMIT')
                            break

                        # Stop repeating
                        if controlling_signal != 'REPEAT':
                            break
                        else:
                            self.verb('REPEATING OF STEP')

                    # Stop iterations
                    if controlling_signal in stop_reset:
                        self.verb('CONTROLLING SIGNAL IS ')
                        self.verb(f'"{controlling_signal}"')
                        self.verb('SO NEXT ITERATION IS IMPOSIBLE')
                        break

                    self.verb('NEXT ITERATION')

                # Stop reflection
                if controlling_signal in stop_reset:
                    self.verb('CONTROLLING SIGNAL IS ')
                    self.verb(f'"{controlling_signal}"')
                    self.verb('SO NEXT REFLECTION IS IMPOSIBLE')
                    break

                # Resoults are empty or not enough for the next reflect
                elif self.inputs_number != self.outputs_number:
                    self.verb('INPUTS AND OUTPUTS ARE INCOMPLETABLE')
                    self.verb('SO NEXT REFLECTION IS IMPOSIBLE')
                    break
                elif resoults == list():
                    self.verb('RESOULTS ARE EMPTY')
                    self.verb('SO NEXT REFLECTION IS IMPOSIBLE')
                    break

                # Prepare request for new reflection from results
                signifying_inputs_values_sqnce = resoults
                reflections_counter += 1
                self.verb('NEXT REFLECTION')

            # Stop reflections loop
            if controlling_signal == 'RESET_REFLECTIONS' or resoults == list():
                self.verb('REFLECTIONS ARE RESET')
            else:
                self.verb('THE END.\n')
                break
        return resoults

    @property
    def first_layer(self):
        return self.layers[0]

    @property
    def last_layer(self):
        return self.layers[-1]

    @property
    def outputs_structure(self) -> dict:
        # Dictkeys are a comments-like...
        # ...to better understand brain structure
        return dict(
            signifying_outputs_number=self.outputs_number,
            controlling_signal_outputs_number=self.CS_O_N,
            transforming_outputs_number=dict_sum(self.T_O_BS),
            writing_memory_outputs_number=self.writing_memory.outputs_number,
            reading_memory_outputs_number=self.reading_memory.outputs_number,
        )
    # aliasing outputs_structure
    O_S = outputs_structure

    def save(self, dir_path: str, file_name_without_extension: str) -> str:
        check_dir_path_slash_ending(dir_path)

        file = f'{dir_path}{file_name_without_extension}.recurrent'
        with open(file, mode='w', encoding='ascii') as filebuffer:
            dump(
                obj=dict(
                    inputs_number=self.inputs_number,
                    outputs_number=self.outputs_number,
                ),
                fp=filebuffer,
            )
        super().save(dir_path, file_name_without_extension)
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
            + f'{self.reading_memory.blocks_number}'\
            + 'reading memory outputs blocks number\n'\
            + f'{self.writing_memory.blocks_number}'\
            + 'writting memory outputs blocks number\n'\
            + '>'
        return repr_string

    def __eq__(self, o) -> bool:
        return self.layers == o.layers\
            and self.inputs_number == o.inputs_number\
            and self.outputs_number == o.outputs_number


# Testing
if __name__ == '__main__':
    class BigBrain(Brain):
        INITIAL_MIDDLE_LAYERS_STRUCTURE = 100 * [10 ** 3,]

    for iteration in range(100):
        print(f'\nITERATION NUMBER {iteration}\n')
        print(
            BigBrain()(
                [[789], [7], [8], [9], [1], [0], [5], [6]],
                verbalize=True,
            ),
        )

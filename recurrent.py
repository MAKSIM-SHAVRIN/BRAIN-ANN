from json import dump, load
from pathlib import Path
from time import time
from typing import Iterable

from numpy import array
from numpy.typing import NDArray

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
    I_WM_O_BN = INITIAL_WRITING_MEMORY_OUTPUTS_BLOCKS_NUMBER = 5
    I_ML_S = INITIAL_MIDDLE_LAYERS_STRUCTURE = 6*[10]
    I_RM_IO_BN = INITIAL_READING_MEMORY_INPUTS_OUTPUTS_BLOCKS_NUMBER = 5

    # Dictkeys are a comments-like to better understand brain structure
    WM_O_BS = WRITTING_MEMORY_OUTPUTS_BLOCK_STRUCTURE = dict(
        layer_adress_ouputs_number=1,
        output_adress_ouputs_number=1,
        input_adress_ouputs_number=1,
        new_value_sign_outputs_number=1,
        new_walue_ouputs_number=1,
    )

    # Dictkeys are a comments-like to better understand brain structure
    RM_O_BS = READING_MEMORY_OUTPUTS_BLOCK_STRUCTURE = dict(
        layer_adress_ouputs_number=1,
        output_adress_ouputs_number=1,
        input_adress_ouputs_number=1,
    )

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
        'STOP_REFLECTIONS',
        'STOP',
    ]

    TRANSFORMING_SIGNALS = [
        'NOTHING',

        'APPEND_OUTPUT',
        'DELETE_OUTPUT',

        'APPEND_WRITTING_MEMORY_OUTPUTS_BLOCK',
        'POP_WRITTING_MEMORY_OUTPUTS_BLOCK',

        'APPEND_READING_MEMORY_INPUTS_OUTPUTS_BLOCK',
        'POP_READING_MEMORY_INPUTS_OUTPUTS_BLOCK',
    ]
    APPEND_RM_IO_B_SIGNAL = TRANSFORMING_SIGNALS[5]
    POP_RM_IO_B_SIGNAL = TRANSFORMING_SIGNALS[6]

    # Methods
    def __init__(self, inputs_number: int = 1, outputs_number: int = 1):
        # Nested functions
        def _count_initial_perceptron_inputs_number() -> int:
            perceptron_inputs_number = dict_sum(
                # Dictkeys are a comments-like...
                # ...to better understand brain structure
                dict(
                    signifying_inputs_number=inputs_number,
                    time_inputs_number=self.T_I_N,
                    time_limit_inputs_number=self.TL_I_N,
                    transforming_error_signal_inpts_number=self.TES_I_N,
                    reflections_counter_inputs_number=self.R_I_N,
                    reflections_limit_inputs_number=self.RL_I_N,
                    steps_counter_inputs_number=self.S_I_N,
                    steps_limit_inputs_number=self.SL_I_N,
                    reading_memory_inputs_number=self.I_RM_IO_BN,
                ),
            )
            return perceptron_inputs_number

        def _count_initial_perceptron_outputs_number() -> int:
            w_memory_outputs_number = self.I_WM_O_BN * dict_sum(self.WM_O_BS)
            r_memory_outputs_number = self.I_RM_IO_BN * dict_sum(self.RM_O_BS)

            perceptron_outputs_number = dict_sum(
                # Dictkeys are a comments-like...
                # ...to better understand brain structure
                dict(
                    signifying_outputs_number=outputs_number,
                    controlling_signal_outputs_number=self.CS_O_N,
                    transforming_outputs_number=dict_sum(self.T_O_BS),
                    writing_memory_outputs_number=w_memory_outputs_number,
                    reading_memory_outputs_number=r_memory_outputs_number,
                ),
            )
            return perceptron_outputs_number

        def _count_initial_perceptron_structure():
            initial_perceptron_structure = [
                _count_initial_perceptron_inputs_number(),
                *self.INITIAL_MIDDLE_LAYERS_STRUCTURE,
                _count_initial_perceptron_outputs_number(),
            ]
            return initial_perceptron_structure

        ################################################################

        self.inputs_number = inputs_number
        self.outputs_number = outputs_number

        self._transforming_error_flag = 0

        super().__init__(_count_initial_perceptron_structure())
        return None  # Added for visual end of the method

    def __call__(
        self, input_values: Iterable,
        time_limit: int | float = 60, steps_limit: int = -1,
        reflections_limit: int = 7, transform=True, introspect=True,
        just_last_resoult=False, do_not_skip_repeat_and_stop=False,
        verbalize=False,
    ) -> list[NDArray[float]] | list:
        """
        The method performs calculations
        using the recurrent generative neural network

        Args:
            input_values (Iterable):
            Input iterable by axis=1 must provide due number of cells
            to number of signifuing inputs of the net to provide reflections.

            time_limit (int | float):
            Ensures that the calculation stops
            after the specified time has passed,
            without waiting a stopping signal,
            -1 for unlimited time.
            Defaults to 60.

            steps_limit (int | NoneType):
            Ensures that the calculation stops
            after the specified number of steps has passed -
            single forvard propogations within the perceptron,
            -1 for unlimited steps.
            Defaults to -1.

            reflections_limit (int):
            Limitation on the number of reflection levels,
            i.e. recursion iterations,
            Warning - it doesen't stop after reaching of limit,
            turns back to reflection number 0 instead,
            -1 for unlimited reflections.
            Defaults to 7.

            transform (bool):
            Allows the network to change its own structure,
            thus changing its efficiency and its memory mechanisms.
            Defaults to True.

            introspect (bool):
            Allows the network to change its own weights,
            thus providing memorization and zero-shot learning.
            Defaults to True.

            just_last_resoult (bool):
            Provides a result of only the last step of the calculation
            and not store the entire chain of results in memory.
            Defaults to False.

            do_not_skip_repeat_and_stop (bool):
            Provides a result equal in length to the input data.
            Defaults to False.

            verbalize (bool):
            Output data about the calculation process to the console.
            Defaults to False.

        Raises:
            RuntimeError: if it is impossible delede an input
            or a writing/reading memory outputs block

        Returns:
            list[NDArray[float]] | list[]: a resoulting list
        """
        # Nested functions
        def verb(*args, **kwargs):
            if verbalize:
                print(*args, **kwargs)

        def if_transform(func):
            def wrapper(*args, **kwargs):
                if transform:
                    verb('\nTRANSFORMING IS ENABLED SO RUN FUNCTION')
                    func(*args, **kwargs)
                else:
                    verb('\nTRANSFORMING IS DISABLED SO FUNCTION WON`T BE RUN')
            return wrapper

        def if_introspect(func):
            def wrapper(*args, **kwargs):
                if introspect:
                    verb('\nINTROSPECTION IS ENABLED SO RUN FUNCTION')
                    return func(*args, **kwargs)
                else:
                    verb('\nINTROSPECTION IS DISABLED - FUNCTION WON`T BE RUN')
            return wrapper

        def catch_error(func):
            def wrapper(*args, **kwargs):
                try:
                    func(*args, **kwargs)
                except RuntimeError:
                    verb('TRANSFORMING ERROR IS CAUGHT')
                    self._transforming_error_flag = 0
                else:
                    verb('NO TRANSFORMING ERROR IS CAUGHT')
                    self._transforming_error_flag = 1
            return wrapper

        ################################################################

        @if_transform
        def _append_output_to_layer(layer_adress: float):
            # Can not append outputs to last layer...
            # ...cuz it has its own special structure
            layers_excluding_last = self.layers[:-1]
            index = get_index_by_decimal(layers_excluding_last, layer_adress)
            layer = self.layers[index]

            verb(f'LAYER {index} CONTAINS {layer.outputs_number}')

            layer.append_output()
            # Add due inputs of next layer
            next_layer = self.layers[index + 1]
            next_layer.append_input()

            verb(f'LAYER {index} CONTAINS {layer.outputs_number}')
            verb(f'OUTPUT IS APPENDED TO LAYER {index}')

        @if_transform
        @catch_error
        def _delete_output_from_layer(
            layer_adress: float, output_adress: float,
        ):
            # Can not delete outputs from last layer...
            # ...cuz it has its own special structure
            layers_excluding_last = self.layers[:-1]
            index = get_index_by_decimal(layers_excluding_last, layer_adress)
            layer = self.layers[index]
            outputs_number = layer.outputs_number

            verb(f'LAYER {index} CONTAINS {outputs_number}')

            if outputs_number == 1:
                verb('OUTPUT CAN NOT BE DELETED')
                raise RuntimeError('Can not delete only output of layer')
            output_index = get_index_by_decimal(
                sequence=list(range(outputs_number)),
                decimal=output_adress,
            )
            layer.delete_output(output_index)
            # Delete due weights of next layer
            next_layer = self.layers[index + 1]
            # `1 + neuron_index` cuz remember about bias weight
            next_layer.delete_input(1 + output_index)

            verb(f'LAYER {index} CONTAINS {layer.outputs_number}')
            verb(f'OUTPUT {output_index} IS DELETED FROM LAYER {index}')

        def _count_index_after_last_writting_memory_output():
            w_memory_outputs_number = dict_sum(self.WM_O_BS) * self.WM_O_BN
            index_after_last_writting_memory_output = dict_sum(
                # Dictkeys are a comments-like...
                # ...to better understand brain structure
                dict(
                    signifying_outputs_number=self.outputs_number,
                    controlling_signal_outputs_number=self.CS_O_N,
                    transforming_outputs_number=dict_sum(self.T_O_BS),
                    writing_memory_outputs_number=w_memory_outputs_number,
                    # don` use final reading_memory_outputs_number...
                    # ...cuz we don` need it
                ),
            )
            return index_after_last_writting_memory_output

        @if_transform
        def _append_writing_memory_outputs_block():
            verb(f'WRITING MEMORY OUTPUTS BLOCKS {self.WM_O_BN}')

            index = _count_index_after_last_writting_memory_output()
            for number in range(dict_sum(self.WM_O_BS)):
                self.last_layer.insert_output(number + index)

            verb(f'WRITING MEMORY OUTPUTS BLOCKS {self.WM_O_BN}')
            verb('WRITING MEMORY OUTPUTS BLOCK IS APPENDED')

        @if_transform
        @catch_error
        def _pop_writing_memory_outputs_block():
            verb(f'WRITING MEMORY OUTPUTS BLOCKS {self.WM_O_BN}')

            if self.writing_memory_outputs_blocks_number == 1:
                verb('WRITING MEMORY OUTPUTS BLOCK CAN NOT BE POPPED')
                raise RuntimeError(
                    'Can not delete only writting memory outputs block',
                )
            index = _count_index_after_last_writting_memory_output()
            for number in range(dict_sum(self.WM_O_BS)):
                self.last_layer.delete_output(index - number - 1)

            verb(f'WRITING MEMORY OUTPUTS BLOCKS {self.WM_O_BN}')
            verb('WRITING MEMORY OUTPUTS BLOCK IS POPPED')

        @if_transform
        def _append_reading_memory_outputs_block():
            verb(f'READING MEMORY OUTPUTS BLOCKS {self.RM_O_BN}')

            for _ in range(dict_sum(self.RM_O_BS)):
                self.last_layer.append_output()
            # Add due reading memory input
            self.first_layer.append_input()

            verb(f'READING MEMORY OUTPUTS BLOCKS {self.RM_O_BN}')
            verb('READING MEMORY OUTPUTS BLOCK IS APPENDED')

        @if_transform
        @catch_error
        def _pop_reading_memory_outputs_block():
            verb(f'READING MEMORY OUTPUTS BLOCKS {self.RM_O_BN}')

            if self.reading_memory_outputs_blocks_number == 1:
                verb('READING MEMORY OUTPUTS BLOCK CAN NOT BE POPPED')
                raise RuntimeError(
                    'Can not delete only reading memory outputs block',
                )
            for _ in range(dict_sum(self.RM_O_BS)):
                self.last_layer.pop_output()
            # Pop due reading memory input
            self.first_layer.pop_input()

            verb(f'READING MEMORY OUTPUTS BLOCKS {self.RM_O_BN}')
            verb('READING MEMORY OUTPUTS BLOCK IS POPPED')

        @if_introspect
        def _write_weights(writing_memory_outputs_values: list[float]):
            writing_memory_outputs_blocks_number = self.WM_O_BN
            verb(
                'WRITTING MEMORY OUTPUTS VALUES:',
                writing_memory_outputs_values,
            )
            verb(
                'WRITTING MEMORY OUTPUTS BLOCKS NUMBER:',
                writing_memory_outputs_blocks_number,
            )
            for number in range(writing_memory_outputs_blocks_number):
                verb(
                    '\nWRITTING MEMORY OUTPUTS BLOCK:',
                    f'{number + 1} / {writing_memory_outputs_blocks_number}',
                )
                (
                    layer_adress_value,
                    output_adress_value,
                    input_adress_value,
                    new_value_sign,
                    new_value,

                    # rest values for next splitting by split_by_volumes
                    writing_memory_outputs_values,

                ) = split_by_volumes(
                    list_for_split=writing_memory_outputs_values,
                    volumes=self.WM_O_BS.values(),
                )
                verb('LAYER ADRESS VALUE:', layer_adress_value)
                verb('OUTPUT ADRESS VALUE:', output_adress_value)
                verb('INPUT ADRESS VALUE:', input_adress_value)
                verb('NEW WEIGHT VALUE SIGN:', new_value_sign)
                verb('NEW WEIGHT VALUE:', new_value)

                layer_index = get_index_by_decimal(
                    sequence=self.layers,
                    decimal=layer_adress_value,
                )
                layer = self.layers[layer_index]

                output_index = get_index_by_decimal(
                    sequence=list(range(layer.outputs_number)),
                    decimal=output_adress_value,
                )
                input_index = get_index_by_decimal(
                    # use inputs number + 1 cuz we need all outputs...
                    # ...including bias
                    sequence=list(range(layer.inputs_number + 1)),
                    decimal=input_adress_value,
                )
                if new_value_sign < 0.5:
                    new_value *= -1

                layer.write_weight(
                    input_index_with_bias=input_index,
                    output_index=output_index,
                    new_walue=new_value,
                )
                verb(
                    f'VALUE "{new_value}" IS WRITTEN TO THE',
                    f'"{input_index} -> {output_index} WEIGHT"',
                    f'OF THE LAYER {layer_index}',
                )
                if input_index == 0:
                    verb('BIAS VALUE IS CHANGED')

        @if_introspect
        def _read_weights(
            reading_memory_outputs_values: list[float],
        ) -> list[float]:
            verb(
                'READING MEMORY OUTPUTS VALUES:',
                reading_memory_outputs_values,
            )
            verb(
                'READING MEMORY OUTPUTS BLOCKS NUMBER:',
                self.reading_memory_outputs_blocks_number,
            )
            reading_memory_inputs_values = list()
            for number in range(self.reading_memory_outputs_blocks_number):
                verb(
                    f'\nREADING MEMORY OUTPUTS BLOCK: {number + 1} /',
                    self.reading_memory_outputs_blocks_number,
                )
                (
                    layer_adress_value,
                    output_adress_value,
                    input_adress_value,

                    # rest values for next splitting by split_by_volumes
                    reading_memory_outputs_values,

                ) = split_by_volumes(
                    list_for_split=reading_memory_outputs_values,
                    volumes=self.RM_O_BS.values(),
                )
                verb('LAYER ADRESS VALUE:', layer_adress_value)
                verb('OUTPUT ADRESS VALUE:', output_adress_value)
                verb('INPUT ADRESS VALUE:', input_adress_value)

                layer_index = get_index_by_decimal(
                    sequence=self.layers,
                    decimal=layer_adress_value,
                )
                layer = self.layers[layer_index]

                output_index = get_index_by_decimal(
                    sequence=list(range(layer.outputs_number)),
                    decimal=output_adress_value,
                )
                input_index = get_index_by_decimal(
                    # use inputs number + 1 cuz we need all outputs...
                    # ...including bias
                    sequence=list(range(layer.inputs_number + 1)),
                    decimal=input_adress_value,
                )
                weight_value = layer.read_weight(input_index, output_index)
                reading_memory_inputs_values.append(weight_value)
                verb(
                    f'VALUE "{weight_value}" IS READ FROM THE',
                    f'"{input_index} -> {output_index} WEIGHT"',
                    f'OF THE LAYER {layer_index}',
                )
            return reading_memory_inputs_values

        ################################################################

        # Start of timer
        if time_limit != -1:
            start_time = time()

        # Start of steps counting
        steps_counter = 0

        # initial controlling signal is always do nothing
        controlling_signal = 'NOTHING'

        # Fill initial reading_memory_inputs_values by zero values
        reading_memory_inputs_values = [0,] * self.RM_O_BN

        verb(f'\nTRANSFORMATION: {transform}')
        verb(f'INTROSPECTION: {introspect}')
        verb(f'JUST LAST RESOULT: {just_last_resoult}')
        verb(f'DO NOT SKIP, REPEAT, STOP: {do_not_skip_repeat_and_stop}')
        verb(f'STEPS LIMIT: {steps_limit}')

        # Reflections loop
        while True:
            signifying_inputs_values_sqnce = input_values_iterator
            verb(f'\nINPUTS VALUES: {signifying_inputs_values_sqnce}')
            # Reflections
            reflections_counter = 0
            while True:
                if reflections_limit != -1:
                    if reflections_counter >= reflections_limit:
                        verb(
                            f'\nREFLECTIONS LIMIT {reflections_limit}',
                            'IS REACHED: STOP REFLECTIONS',
                        )
                        controlling_signal = 'STOP_REFLECTIONS'
                        break

                resoults = list()
                # Iterations
                for signiying_inputs_values in signifying_inputs_values_sqnce:
                    if not do_not_skip_and_repeat:
                        if controlling_signal == 'SKIP':
                            controlling_signal = 'NOTHING'
                            verb('\nSKIPPED')
                            continue
                    # Repeating
                    while True:
                        current_time = time()

                        # Stop by time limit
                        if time_limit != -1:
                            if time_limit < current_time - start_time:
                                controlling_signal = 'STOP'
                                verb('\nSTOPPED BY TIME LIMIT')
                                break

                        # Stop by steps limit
                        if steps_limit != -1 and steps_limit < steps_counter:
                            controlling_signal = 'STOP'
                            verb('\nSTOPPED BY STEP LIMIT')
                            break

                        # Get outputs as list of binary signals and
                        # Split binary list to valuable binary lists
                        verb(
                            '\nREADING MEMORY INPUTS VALUES: ',
                            reading_memory_inputs_values,
                        )
                        inputs_values_list = [
                            *signiying_inputs_values,
                            current_time,
                            time_limit,
                            self._transforming_error_flag,
                            reflections_counter,
                            reflections_limit,
                            steps_counter,
                            steps_limit,
                            *reading_memory_inputs_values,
                        ]

                        verb(f'\nCURRENT INPUTS: {signifying_inputs_values}')
                        verb(f'CURRENT_TIME: {current_time}')
                        verb(f'TIME LIMIT: {time_limit}')
                        verb(
                            'TRASFORMATION ERROR ON PREVIOUS STEP: ',
                            self._transforming_error_flag,
                        )
                        verb(f'REFLECTION NUMBER: {reflections_counter}')
                        verb(f'REFLECTIONS LIMIT: {reflections_limit}')
                        verb(f'STEPS NUMBER: {steps_counter}')
                        verb(f'STEPS LIMIT: {steps_limit}')
                        verb(f'CONTROLLING SIGNAL: {controlling_signal}')
                        (
                            signifying_outputs_values,
                            controlling_signal_outputs_values,
                            transforming_outputs_values,
                            writting_memory_outputs_values,
                            reading_memory_outputs_values,

                        ) = split_by_volumes(
                            list_for_split=super().__call__(
                                array(inputs_values_list),
                            ),
                            volumes=self.outputs_structure.values(),
                            get_rest=False,
                        )

                        verb(
                            '\nSIGNIFYING OUTPUT:',
                            signifying_outputs_values,
                        )
                        verb(
                            'CONTROLLING SIGNAL OUTPUT: ',
                            controlling_signal_outputs_values,
                        )
                        verb(
                            'TRANSFORMING OUTPUTS: ',
                            transforming_outputs_values,
                        )
                        verb(
                            'WRITTING MEMORY OUTPUTS: ',
                            writting_memory_outputs_values,
                        )
                        verb(
                            'READING MEMORY OUTPUTS: ',
                            reading_memory_outputs_values,
                        )

                        # Get controlling signal
                        controlling_signal = get_element_by_decimal(
                            self.CONTROLLING_SIGNALS,
                            controlling_signal_outputs_values,
                        )
                        verb(f'\nCONTROLLING SIGNAL: {controlling_signal}')

                        # Introspection
                        reading_memory_inputs_values = _read_weights(
                            reading_memory_outputs_values,
                        )
                        _write_weights(writting_memory_outputs_values)

                        # Transforming
                        (
                            transform_signal_output_value,
                            layer_adress_output_value,
                            output_adress_output_value,

                        ) = split_by_volumes(
                            list_for_split=transforming_outputs_values,
                            volumes=self.T_O_BS.values(),
                            get_rest=False,
                        )
                        signal = get_element_by_decimal(
                            self.TRANSFORMING_SIGNALS,
                            transform_signal_output_value,
                        )
                        verb(
                            '\nTRANSFORMING SIGNAL VALUE: ',
                            transform_signal_output_value,
                        )
                        verb(
                            'LAYER ADRESS VALUE: ',
                            layer_adress_output_value,
                        )
                        verb(
                            'OUTPUT ADRESS VALUE: ',
                            output_adress_output_value,
                        )
                        verb(f'\nTRANSFORMING SIGNAL: {signal}')

                        # Add or delete reading memory neurons
                        if signal == self.APPEND_RM_IO_B_SIGNAL:
                            _append_reading_memory_outputs_block()

                        elif signal == self.POP_RM_IO_B_SIGNAL:
                            _pop_reading_memory_outputs_block()

                        # Introspection
                        reading_memory_inputs_values = _read_weights(
                            reading_memory_outputs_values,
                        )
                        verb(
                            '\nREADING MEMORY INPUTS VALUES: ',
                            reading_memory_inputs_values,
                        )

                        # Add or delete neuron
                        elif signal == 'APPEND_OUTPUT':
                            _append_output_to_layer(layer_adress_output_value)

                        elif signal == 'DELETE_OUTPUT':
                            _delete_output_from_layer(
                                layer_adress_output_value,
                                output_adress_output_value,
                            )

                        # Add or delete writting memory neurons
                        elif signal == 'APPEND_WRITTING_MEMORY_OUTPUTS_BLOCK':
                            _append_writing_memory_outputs_block()

                        elif signal == 'POP_WRITTING_MEMORY_OUTPUTS_BLOCK':
                            _pop_writing_memory_outputs_block()

                        # Add character to list of resoults
                        if just_last_resoult:
                            resoults = [signifying_outputs_values,]
                        else:
                            resoults.append(signifying_outputs_values)

                        # Increase steps counter
                        steps_counter += 1

                        # Stop repeating
                        if do_not_skip_and_repeat:
                            break
                        if controlling_signal != 'REPEAT':
                            break
                        else:
                            verb('\nREPEATING')

                    # Stop iterations
                    if controlling_signal == 'STOP':
                        break
                    elif controlling_signal == 'STOP_REFLECTIONS':
                        break
                    verb('\nNEXT ITERATION')

                # Stop reflection
                if controlling_signal == 'STOP':
                    break
                elif controlling_signal == 'STOP_REFLECTIONS':
                    break
                # Resoults are empty or not enough for the next reflection
                elif self.inputs_number != self.outputs_number:
                    break
                elif resoults == list():
                    verb('RESOULTS ARE EMPTY')
                    break
                elif just_last_resoult:
                    break

                # Prepare request for new reflection from results
                signifying_inputs_values_sqnce = resoults
                reflections_counter += 1
                verb('\nNEXT REFLECTION')

            # Stop reflections loop
            if controlling_signal != 'STOP_REFLECTIONS':
                break
            verb('\nREFLECTIONS ARE RESET')
        return resoults

    @property
    def first_layer(self):
        return self.layers[0]

    @property
    def last_layer(self):
        return self.layers[-1]

    @property
    def reading_memory_outputs_blocks_number(self) -> int:
        return self.first_layer.inputs_number\
            - self.inputs_number\
            - self.TIME_INPUTS_NUMBER\
            - self.TIME_LIMIT_INPUTS_NUMBER\
            - self.TRANSFORMING_ERROR_SIGNAL_INPUTS_NUMBER\
            - self.REFLECTIONS_INPUTS_NUMBER\
            - self.REFLECTIONS_LIMIT_INPUTS_NUMBER\
            - self.STEPS_INPUTS_NUMBER\
            - self.STEPS_LIMIT_INPUTS_NUMBER
    # aliasing reading_memory_outputs_blocks_number
    RM_O_BN = reading_memory_outputs_blocks_number

    @property
    def writing_memory_outputs_blocks_number(self) -> int:
        reading_memory_outputs_number = self.RM_O_BN * dict_sum(self.RM_O_BS)
        writing_memory_outputs_number = self.last_layer.outputs_number\
            - self.outputs_number\
            - self.CONTROLLING_SIGNAL_OUTPUTS_NUMBER\
            - dict_sum(self.TRANSFORMING_OUTPUTS_BLOCK_STRUCTURE)\
            - reading_memory_outputs_number
        resoult = writing_memory_outputs_number / dict_sum(self.WM_O_BS)
        return round(resoult)
    # aliasing writing_memory_outputs_blocks_number
    WM_O_BN = writing_memory_outputs_blocks_number

    @property
    def outputs_structure(self) -> dict:
        writing_memory_outputs_number = self.WM_O_BN * dict_sum(self.WM_O_BS)
        reading_memory_outputs_number = self.RM_O_BN * dict_sum(self.RM_O_BS)
        # Dictkeys are a comments-like...
        # ...to better understand brain structure
        return dict(
            signifying_outputs_number=self.outputs_number,
            controlling_signal_outputs_number=self.CS_O_N,
            transforming_outputs_number=dict_sum(self.T_O_BS),
            writing_memory_outputs_number=writing_memory_outputs_number,
            reading_memory_outputs_number=reading_memory_outputs_number,
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
            + f'{self.RM_O_BN} reading memory outputs blocks number\n'\
            + f'{self.WM_O_BN} writting memory outputs blocks number\n'\
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

    print(
        BigBrain()(
            [[789], [7], [8], [9], [1], [0], [5], [6]],
            verbalize=True
        ),
    )

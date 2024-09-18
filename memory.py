from typing import Callable
from numpy import append
from numpy.typing import NDArray

from utils import dict_sum, split_by_volumes, get_index_by_decimal


class Memory:
    # Structure
    I_BN = INITIAL_BLOCKS_NUMBER = 5
    BS = BLOCK_STRUCTURE = ...

    def align_walues_with_outputs(self, method: Callable) -> Callable:
        def wrapper(self, values: NDArray[float]):
            difference = self.outputs_number - len(values)
            if difference > 0:
                values = append(arr=values, values=difference*[0,], axis=0)
            elif difference < 0:
                values = values[:self.outputs_number]

            return method(values)
        return wrapper

    def __init__(self, brain):
        self.brain = brain

        # decorating of methods by decorators from Brain class
        self.append_block = self.brain.if_transform(self.append_block)

        self.pop_block = self.brain\
            .if_transform(self.brain.catch_error(self.pop_block))

        self.general_method = self.brain\
            .if_introspect(self.align_walues_with_outputs(self.general_method))

    @property
    def initial_outputs_number(self) -> int:
        return self.INITIAL_BLOCKS_NUMBER * dict_sum(self.BLOCK_STRUCTURE)
    # Alias
    I_ON = initial_outputs_number

    @property
    def outputs_number(self) -> int:
        return self.blocks_number * dict_sum(self.BLOCK_STRUCTURE)
    # Alias
    ON = outputs_number


class WritingMemory(Memory):
    # Dictkeys are a comments-like to better understand brain structure
    BS = BLOCK_STRUCTURE = dict(
        layer_adress_ouputs_number=1,
        output_adress_ouputs_number=1,
        input_adress_ouputs_number=1,
        new_value_sign_outputs_number=1,
        new_walue_ouputs_number=1,
    )

    def __init__(self, brain):
        self.general_method = self.write_weights
        super().__init__(brain)

    @property
    def blocks_number(self) -> int:
        writing_memory_outputs_number = self.brain.last_layer.outputs_number\
            - self.brain.outputs_number\
            - self.brain.CONTROLLING_SIGNAL_OUTPUTS_NUMBER\
            - dict_sum(self.brain.TRANSFORMING_OUTPUTS_BLOCK_STRUCTURE)\
            - self.brain.reading_memory.outputs_number
        resoult = writing_memory_outputs_number\
            / dict_sum(self.BLOCK_STRUCTURE)
        return round(resoult)

    @property
    def index_after_last_output(self) -> int:
        index = dict_sum(
            # Dictkeys are a comments-like...
            # ...to better understand brain structure
            dict(
                signifying_outputs_number=self.brain.outputs_number,
                controlling_signal_outputs_number=self.brain.CS_O_N,
                transforming_outputs_number=dict_sum(self.brain.T_O_BS),
                writing_memory_outputs_number=self.outputs_number,
                # don` use final reading_memory_outputs_number...
                # ...cuz we don` need it
            ),
        )
        return index

    def append_block(self):
        index = self.index_after_last_output
        for number in range(dict_sum(self.BLOCK_STRUCTURE)):
            self.brain.last_layer.insert_output(number + index)
        return None

    def pop_block(self):
        if self.blocks_number == 1:
            raise RuntimeError(
                'Can`t delete only writing memory outputs block',
            )
        index = self.index_after_last_output
        for number in range(dict_sum(self.BLOCK_STRUCTURE)):
            self.brain.last_layer.delete_output(index - number - 1)
        return None

    def write_weights(self, writing_memory_outputs_values: list[float]):
        for number in range(self.blocks_number):
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
                volumes=self.BLOCK_STRUCTURE.values(),
            )

            layer_index = get_index_by_decimal(
                sequence=self.brain.layers,
                decimal=layer_adress_value,
            )
            layer = self.brain.layers[layer_index]

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
        return None


class ReadingMemory(Memory):
    # Dictkeys are a comments-like to better understand brain structure
    BS = BLOCK_STRUCTURE = dict(
        layer_adress_ouputs_number=1,
        output_adress_ouputs_number=1,
        input_adress_ouputs_number=1,
    )

    def __init__(self, brain):
        self.general_method = self.read_weights
        super().__init__(brain)

        # decorating of methods
        self.append_block = self.align_walues_with_inputs(self.append_block)
        self.pop_block = self.align_walues_with_inputs(self.pop_block)

    def align_walues_with_inputs(self, method: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            method(*args, **kwargs)

            difference = self.blocks_number - len(self.brain._rmi_values)
            if difference > 0:
                self.brain._rmi_values = append(
                    arr=self.brain._rmi_values,
                    values=difference*[0,],
                    axis=0,
                )
            elif difference < 0:
                self.brain._rmi_values = self.brain\
                    ._rmi_values[:self.blocks_number]
        return wrapper

    @property
    def blocks_number(self) -> int:
        return self.brain.first_layer.inputs_number\
            - self.brain.inputs_number\
            - self.brain.TIME_INPUTS_NUMBER\
            - self.brain.TIME_LIMIT_INPUTS_NUMBER\
            - self.brain.TRANSFORMING_ERROR_SIGNAL_INPUTS_NUMBER\
            - self.brain.REFLECTIONS_INPUTS_NUMBER\
            - self.brain.REFLECTIONS_LIMIT_INPUTS_NUMBER\
            - self.brain.STEPS_INPUTS_NUMBER\
            - self.brain.STEPS_LIMIT_INPUTS_NUMBER

    def append_block(self):
        for _ in range(dict_sum(self.BLOCK_STRUCTURE)):
            self.brain.last_layer.append_output()
        # Add due reading memory input
        self.brain.first_layer.append_input()
        return None

    def pop_block(self):
        if self.blocks_number == 1:
            raise RuntimeError(
                'Can`t delete only reading memory outputs block',
            )
        for _ in range(dict_sum(self.BLOCK_STRUCTURE)):
            self.brain.last_layer.pop_output()
        # Pop due reading memory input
        self.brain.first_layer.pop_input()
        return None

    def read_weights(
        self, reading_memory_outputs_values: list[float],
    ) -> list[float]:
        rmi_values = list()
        for number in range(self.blocks_number):
            (
                layer_adress_value,
                output_adress_value,
                input_adress_value,

                # rest values for next splitting by split_by_volumes
                reading_memory_outputs_values,

            ) = split_by_volumes(
                list_for_split=reading_memory_outputs_values,
                volumes=self.BLOCK_STRUCTURE.values(),
            )
            layer_index = get_index_by_decimal(
                sequence=self.brain.layers,
                decimal=layer_adress_value,
            )
            layer = self.brain.layers[layer_index]

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
            rmi_values.append(weight_value)
        return rmi_values

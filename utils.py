from functools import wraps
from time import time


UNICODE_RANGES = [
    (0x0020, 0x007E),  # ASCII punctuation and symbols, digits, latin
    (0x0400, 0x045F),  # Cyrilic
]


def get_index_by_decimal(sequence, decimal: float):
    return round(decimal * (len(sequence) - 1))


def get_element_by_decimal(sequence, decimal: float):
    return sequence[get_index_by_decimal(sequence, decimal)]


def split_by_volumes(
    list_for_split: list, volumes: list[int], get_rest: bool = True,
) -> list[list]:
    if len(list_for_split) < 2:
        raise ValueError(
            'list_for_split must contain at least two element to be splittied',
        )
    if len(volumes) < 2:
        raise ValueError('volumes must contain at least two splitting values')
    if 0 in volumes:
        raise ValueError('Every resoult list must have an element at least')
    if len(list_for_split) < sum(volumes):
        # You can`t split a coin to two stacks
        raise ValueError(
            'list_for_split length must be longer or equal to volumes length',
        )
    resoult_lists = list()
    for volume in volumes:
        resoult_lists.append(list_for_split[:volume])
        list_for_split = list_for_split[volume:]
    # Add the rest of splited list if it is
    if get_rest:
        resoult_lists.append(list_for_split)
    return resoult_lists


def get_unicode_characters_by_ranges(ranges: list[tuple[int]]) -> str:
    string_characters = list()
    for start, finish in ranges:
        if finish < start:
            raise ValueError('Start value of range is bigger thn finish value')
        for number in range(start, finish + 1):
            string_characters.append(chr(number))
    return ''.join(string_characters)


def mean(sequence) -> float:
    if len(sequence) < 2:
        raise ValueError('Sequence must contain at least two elements')
    return sum(sequence) / len(sequence)


def dict_sum(dictionary_of_numbers: dict):
    if len(dictionary_of_numbers) < 2:
        raise ValueError('Dictionary must contain at least two elements')
    return sum(dictionary_of_numbers.values())


def conv_int_to_list(number: int, length: int) -> list[int]:
    resoult_list = list()
    while number != 0:
        number, modulo = divmod(number, 2)
        resoult_list.append(modulo)
    rjust_len = length - len(resoult_list)
    if rjust_len < 0:
        raise ValueError('Resoult list length is bigger than length argument')
    return rjust_len * [0] + list(reversed(resoult_list))


def conv_list_to_int(binaries_list: list[int]) -> int:
    for digit in binaries_list:
        if digit not in [0, 1]:
            raise ValueError('List must contain just zeros and ones')
    return sum([n*2**p for p, n in enumerate(reversed(binaries_list))])


def check_dir_path_slash_ending(dir_path: str):
    if dir_path[-1] != '/':
        raise ValueError("Directory path must ending with `/`")


def make_simple_structure(
    inputs_number: int, intermediate_layers_number: int,
    intermediate_layers_neurons_number: int, outputs_number: int,
) -> list[int]:
    resoult_structure = [inputs_number]
    for _item in range(intermediate_layers_number):
        resoult_structure.append(intermediate_layers_neurons_number)
    resoult_structure.append(outputs_number)
    return resoult_structure


def measure_execution_time(procedure):
    @wraps(procedure)
    def _wrapper(*args, **kwargs) -> float:
        start = time()
        procedure(*args, **kwargs)
        finish = time()
        return finish - start
    return _wrapper


def mix_in(*mixins: type):
    def decorator(class_be_decorated: type):
        for mixin in mixins:
            for key, value in mixin.__dict__.items():
                if callable(value) or isinstance(value, classmethod):
                    setattr(class_be_decorated, key, value)
        return class_be_decorated
    return decorator


def split_by_evenodd_position(sequence) -> tuple:
    if len(sequence) < 2:
        raise ValueError('Sequence must contain at least two elements')
    if len(sequence) % 2 != 0:
        raise ValueError('Sequence must contain even number of elements')
    return (sequence[::2], sequence[1::2])

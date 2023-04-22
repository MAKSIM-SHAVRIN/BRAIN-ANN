from functools import wraps
from random import choice
from time import time

from jaro import jaro_winkler_metric


def get_unicode_characters_by_ranges(ranges: list[tuple]) -> str:
    string_characters = list()
    for start, finish in ranges:
        for number in range(start, finish + 1):
            string_characters.append(chr(number))
    return ''.join(string_characters)


def jaro_loss(string1, string2):
    return 1 - jaro_winkler_metric(string1, string2)


def mean(sequence) -> float:
    return sum(sequence) / len(sequence)


def dict_sum(dictionary_of_numbers: dict):
    return sum(dictionary_of_numbers.values())


def get_element_by_fraction(sequence, fraction):
    return sequence[round(fraction * (len(sequence) - 1))]


def conv_int_to_list(number: int, width: int) -> list[int]:
    resoult_list = list()
    while number != 0:
        number, modulo = divmod(number, 2)
        resoult_list.append(modulo)
    rjust_len = width - len(resoult_list)
    return rjust_len * [0] + list(reversed(resoult_list))


def conv_list_to_int(binaries_list: list) -> int:
    return sum([n*2**p for p, n in enumerate(reversed(binaries_list))])


def generate_sign() -> int:
    return choice((-1, 1,))


def make_simple_structure(
    inputs_number: int, intermediate_layers_number: int,
    intermediate_layers_neurons_number: int, outputs_number: int,
) -> list:
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


def mix_in(*mixins):
    def decorator(class_be_decorated):
        for mixin in mixins:
            for key, value in mixin.__dict__.items():
                if callable(value) or isinstance(value, classmethod):
                    setattr(class_be_decorated, key, value)
        return class_be_decorated
    return decorator


def split_by_evenodd_position(sequence) -> tuple:
    return (sequence[::2], sequence[1::2])

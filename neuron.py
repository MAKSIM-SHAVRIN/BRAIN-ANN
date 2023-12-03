from random import choice


def heaviside(neuro_sum: float) -> int:
    if neuro_sum < 0:
        return 0
    return 1


def count_arrays_product(array_1: list[int], array_2: list[int]) -> list[int]:
    if len(array_1) != len(array_2):
        raise ValueError('The arguments have various lengths')
    return [num_1 * num_2 for num_1, num_2 in zip(array_1, array_2)]


class Neuron:
    def __init__(self, inputs_number: int):
        if inputs_number < 2:
            raise ValueError('Neuron must have at least two non-bias inputs')
        # Add the 1 to inputs number for bias weight creating
        self.weights = [choice((-1, 1,)) for _ in range(inputs_number + 1)]

    def __call__(self, inputs_values: list[int]) -> int:
        # Add bias is always 1
        inputs = [1,] + inputs_values

        # Product inputs values on due weights
        arrays_product = count_arrays_product(self.weights, inputs)

        # Sum all values
        weighted_sum = sum(arrays_product)

        # Use activation function to get neuron output
        return heaviside(weighted_sum)

    def __repr__(self):
        return f'\n< Neuron with weights: {self.weights} >'

    def _add_weight(self):
        self.weights.append(choice((-1, 1,)))

    def _delete_weight(self, index: int):
        if len(self.weights) <= 3:
            raise PermissionError(
                'Can`t delete, cuz every Neuron must have two non-bias inputs'
            )
        if index == 0:
            raise PermissionError('Can`t delete bias input')
        self.weights.pop(index)

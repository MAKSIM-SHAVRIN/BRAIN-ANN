from random import choice


def heaviside(neuro_sum: float) -> int:
    if neuro_sum >= 0:
        return 1
    return 0


def count_arrays_product(first_array: list, second_array: list) -> list:
    return [first * second for first, second in zip(first_array, second_array)]


class Neuron:
    def __init__(self, inputs_number: int):
        # Add the 1 to inputs number for bias weight creating
        self.weights = [choice((-1, 1,)) for _ in range(inputs_number + 1)]

    def __call__(self, inputs_values: list[int]) -> int:
        return heaviside(self._get_weighted_sum(inputs_values))

    def _get_weighted_sum(self, inputs_values: list) -> int:
        return sum(count_arrays_product(self.weights, inputs_values))

    def __repr__(self):
        return f'\n< Neuron with weights: {self.weights} >'

    def add_weight(self):
        self.weights.append(choice((-1, 1,)))

    def delete_weight(self, weight_number: int):
        self.weights.pop(index=weight_number)

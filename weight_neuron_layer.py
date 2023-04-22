from utils import generate_sign


def heaviside(neuro_sum: float) -> int:
    if neuro_sum >= 0:
        return 1
    return 0


def count_arrays_product(first_array: list, second_array: list) -> list:
    return [first * second for first, second in zip(first_array, second_array)]


class Weight:
    def __init__(self):
        self.value = generate_sign()

    def __repr__(self):
        return f'\n< Weight: {self.value} >'

    @classmethod
    def init_with_value(cls, value: int):
        new_weight = cls()
        new_weight.value = value
        return new_weight


class Neuron:
    def __init__(self, inputs_number: int):
        self.weights = [Weight() for _ in range(inputs_number)]

    def __call__(self, inputs_values: list[int]) -> int:
        return heaviside(self._get_weighted_sum(inputs_values))

    def __repr__(self):
        return f'\n< Neuron with {len(self.weights)} inputs >'

    @property
    def weights_values(self):
        return [weight.value for weight in self.weights]

    def _get_weighted_sum(self, inputs_values: list) -> int:
        return sum(count_arrays_product(self.weights_values, inputs_values))


class Bias(Neuron):
    def __init__(self):
        super().__init__(inputs_number=0)

    def __call__(self, *args) -> int:
        return 1

    def __repr__(self):
        return '\n< Bias >'


class Layer:
    def __init__(self, neuron_inputs_number: int, neurons_number: int):
        self.neurons = [Bias()]
        for _ in range(neurons_number):
            self.neurons.append(Neuron(neuron_inputs_number))

    def __call__(self, inputs_values: list[int]) -> list[int]:
        return [neuron(inputs_values) for neuron in self.neurons]

    def __repr__(self):
        return f'\n< Layer with {len(self.neurons)} neurons >'


# Testing:
if __name__ == '__main__':
    layer = Layer(neuron_inputs_number=4, neurons_number=3)
    print(layer.neurons)

from neuron import Neuron


class Layer:
    def __init__(self, neuron_inputs_number: int, neurons_number: int):
        self.neurons = []
        for _ in range(neurons_number):
            self.neurons.append(Neuron(neuron_inputs_number))

    def __call__(self, inputs_values: list[int]) -> list[int]:
        return [neuron(inputs_values) for neuron in self.neurons]

    def __repr__(self):
        return f'\n< Layer with {len(self.neurons)} neurons >'

    def _add_neuron(self, neuron_inputs_number: int):
        self.neurons.append(Neuron(neuron_inputs_number))

    def _insert_neuron(self, index: int, neuron_inputs_number: int):
        self.neurons.insert(index, Neuron(neuron_inputs_number))

    def _add_weights(self):
        for neuron in self.neurons:
            neuron._add_weight()

    def _delete_weights(self, weight_number: int):
        for neuron in self.neurons:
            neuron._delete_weight(weight_number)

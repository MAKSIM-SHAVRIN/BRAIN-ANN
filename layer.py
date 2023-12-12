from neuron import Neuron


class Layer:
    def __init__(self, neuron_inputs_number: int, neurons_number: int):
        if neuron_inputs_number < 2:
            raise ValueError('Neuron must have at least two non-bias inputs')
        if neurons_number < 1:
            raise ValueError('Layer must have at least one neurons')
        self.neurons = []
        for _ in range(neurons_number):
            self.neurons.append(Neuron(neuron_inputs_number))

    def __call__(self, inputs_values: list[int]) -> list[int]:
        return [neuron(inputs_values) for neuron in self.neurons]

    def __repr__(self):
        return f'\n< Layer with {len(self.neurons)} neurons >'

    def _add_neuron(self):
        neuron_inputs_number = len(self.neurons[0].weights) - 1
        self.neurons.append(Neuron(neuron_inputs_number))

    def _insert_neuron(self, index: int):
        neuron_inputs_number = len(self.neurons[0].weights) - 1
        self.neurons.insert(index, Neuron(neuron_inputs_number))

    def _add_weights(self):
        for neuron in self.neurons:
            neuron._add_weight()

    def _delete_weights(self, index: int):
        for neuron in self.neurons:
            neuron._delete_weight(index)

    def __eq__(self, o):
        return self.neurons == o.neurons

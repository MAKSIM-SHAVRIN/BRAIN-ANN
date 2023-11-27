from neuron import Neuron


class Layer:
    def __init__(self, neuron_inputs_number: int, neurons_number: int):
        self.neurons = []
        for _ in range(neurons_number):
            self.neurons.append(Neuron(neuron_inputs_number))

    def __call__(self, inputs_values: list[int]) -> list[int]:
        inputs_values.insert(0, 1)  # Add bias is always 1
        return [neuron(inputs_values) for neuron in self.neurons]

    def __repr__(self):
        return f'\n< Layer with {len(self.neurons)} neurons >'

    def add_neuron(self, neuron_inputs_number: int):
        self.neurons.append(Neuron(neuron_inputs_number))

    def add_weights(self):
        for neuron in self.neurons:
            neuron.add_weight()

    def delete_weights(self, weight_number: int):
        for neuron in self.neurons:
            neuron.delete_weight()

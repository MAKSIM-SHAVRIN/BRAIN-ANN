from weight_neuron_layer import Layer, Neuron, Weight
from json import dump, load
from utils import get_element_by_fraction


class Perceptron:
    def __init__(self, structure: list):
        self.layers = list()
        for layer_number, neurons_number in enumerate(structure):
            # first layer (index=0) of the structure isn't a true layer,
            # but the next layer neuron's inputs number.
            if layer_number == 0:
                continue
            # a last layer neurons` number consider bias adding
            last_layer_neurons_number = structure[layer_number - 1] + 1

            layer = Layer(
                neuron_inputs_number=last_layer_neurons_number,
                neurons_number=neurons_number,
            )
            self.layers.append(layer)

    def __repr__(self):
        return f'< Perceptron: {self.structure}> '

    def __call__(self, inputs_values: list[int]) -> list[int]:
        inputs_values.insert(0, 1)  # first layer bias output emitation
        resoults = inputs_values
        for layer in self.layers:
            resoults = layer(resoults)
        return resoults[1:]

    def save(self, file: str) -> str:
        with open(file=file, mode='w', encoding='ascii') as filebuffer:
            dump(
                obj=dict(
                    structure=self.structure,
                    weights=[weight.value for weight in self.all_weights],
                ),
                fp=filebuffer,
            )
        return file

    @classmethod
    def load(cls, file: str):
        with open(file=file, mode='r', encoding='ascii') as filebuffer:
            dictionary = load(fp=filebuffer)
        perceptron = Perceptron(structure=dictionary['structure'])
        for position, weight in enumerate(perceptron.all_weights):
            weight.value = dictionary['weights'][position]
        return perceptron

    def add_neuron(self, layer_number: int):
        self.layers[layer_number - 1].neurons.append(
            Neuron(inputs_number=self.structure[layer_number - 1] + 1),
        )
        if layer_number != len(self.structure) - 1:
            for neuron in self.layers[layer_number].neurons[1:]:
                neuron.weights.append(Weight())

    def delete_neuron(self, layer_number: int):
        neurons = self.layers[layer_number - 1].neurons
        if len(neurons) != 2:
            neurons.pop()
            if layer_number != len(self.structure) - 1:
                for neuron in self.layers[layer_number].neurons[1:]:
                    neuron.weights = neuron.weights[:-1]

    def change_weight(
        self, layer_fraction: float, neuron_fraction: float,
        weight_fraction: float, new_value: int,
    ):
        layer = get_element_by_fraction(self.layers, layer_fraction)
        neuron = get_element_by_fraction(layer.neurons[1:], neuron_fraction)
        weight = get_element_by_fraction(neuron.weights, weight_fraction)
        weight.value = new_value

    @property
    def weights_number(self) -> int:
        return len(self.all_weights)

    @property
    def structure(self) -> list:
        structure = list()
        for number, layer in enumerate(self.layers):
            if number == 0:
                structure.append(len(layer.neurons[-1].weights) - 1)
            structure.append(len(layer.neurons) - 1)
        return structure

    @property
    def all_weights(self) -> list:
        weights = list()
        for neuron in self.all_neurons:
            for weight in neuron.weights:
                weights.append(weight)
        return weights

    @property
    def all_neurons(self) -> list:
        neurons = list()
        for layer in self.layers:
            for neuron in layer.neurons:
                neurons.append(neuron)
        return neurons


# Testing;
if __name__ == '__main__':
    neuronet = Perceptron(structure=[3, 5, 4, 100, 6])
    print(neuronet.structure)

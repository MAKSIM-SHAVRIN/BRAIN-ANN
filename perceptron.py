from json import dump, load

from layer import Layer


class Perceptron:
    def __init__(self, structure: list):
        self.layers = list()
        # first layer (index=0) of the structure isn't a true layer,
        # but the next layer neuron's inputs number.
        for layer_number, neurons_number in enumerate(structure[1:]):
            self.layers.append(
                Layer(
                    neuron_inputs_number=structure[layer_number],
                    neurons_number=neurons_number,
                ),
            )

    @property
    def structure(self) -> list:
        structure = list()
        for number, layer in enumerate(self.layers):
            if number == 0:
                structure.append(len(layer.neurons[0].weights) - 1)
            structure.append(len(layer.neurons))
        return structure

    def __repr__(self):
        return f'< Perceptron: {self.structure} >'

    def __call__(self, inputs_values: list[int]) -> list[int]:
        resoults = inputs_values
        for layer in self.layers:
            resoults = layer(resoults)
        return resoults

    def save(self, file: str) -> str:
        with open(file=file, mode='w', encoding='ascii') as filebuffer:
            dump(
                obj=dict(structure=self.structure, weights=self.all_weights),
                fp=filebuffer,
            )
        return file

    @classmethod
    def load(cls, file: str):
        with open(file=file, mode='r', encoding='ascii') as filebuffer:
            dictionary = load(fp=filebuffer)
        perceptron = cls(structure=dictionary['structure'])
        weights = dictionary['weights']
        for neuron in perceptron.all_neurons:
            weights_number = len(neuron.weights)
            neuron.weights = weights[:weights_number]
            weights = weights[weights_number:]
        return perceptron

    @property
    def weights_number(self) -> int:
        return len(self.all_weights)

    @property
    def all_weights(self) -> list:
        weights = list()
        for neuron in self.all_neurons:
            weights.extend(neuron.weights)
        return weights

    @property
    def all_neurons(self) -> list:
        neurons = list()
        for layer in self.layers:
            neurons.extend(layer.neurons)
        return neurons


# Testing;
if __name__ == '__main__':
    neuronet = Perceptron([2, 2])
    print(neuronet.weights_number)
    print(neuronet([1, 0]))

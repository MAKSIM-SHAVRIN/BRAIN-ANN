from itertools import chain
from json import dump, load
from pathlib import Path

from layer import Layer
from neuron import Neuron  # Imported for annotation
from utils import check_dir_path_slash_ending


class Perceptron:
    def __init__(self, structure: list[int]):
        if len(structure) <= 2:
            raise ValueError('Perceptron must have at least 2 layers')
        # first layer (index=0) of the structure isn't a true layer,
        # but the next layer neuron's inputs number.
        self.layers = list()
        for layer_number, neurons_number in enumerate(structure[1:]):
            self.layers.append(Layer(structure[layer_number], neurons_number))

    @classmethod
    def init_with_weights(cls, structure: list[int], weights: list[int]):
        perceptron = cls(structure)
        if len(perceptron.all_weights) != len(weights):
            raise ValueError('Wrong weights number')
        for neuron in perceptron.all_neurons:
            weights_number = len(neuron.weights)
            neuron.weights = weights[:weights_number]
            weights = weights[weights_number:]
        return perceptron

    @property
    def structure(self) -> list[int]:
        structure = list()
        for number, layer in enumerate(self.layers):
            if number == 0:
                structure.append(len(layer.neurons[0].weights) - 1)
            structure.append(len(layer.neurons))
        return structure

    def __repr__(self) -> str:
        return f'< Perceptron: {self.structure} >'

    def __call__(self, inputs_values: list[int]) -> list[int]:
        resoults = inputs_values
        for layer in self.layers:
            resoults = layer(resoults)
        return resoults

    def save(self, dir_path: str, file_name: str) -> str:
        check_dir_path_slash_ending(dir_path)
        file = dir_path + file_name + '.perceptron'
        with open(file, mode='w', encoding='ascii') as filebuffer:
            dump(
                obj=dict(structure=self.structure, weights=self.all_weights),
                fp=filebuffer,
            )
        return file

    @classmethod
    def load(cls, file: str):
        with open(file, mode='r', encoding='ascii') as filebuffer:
            dictionary = load(fp=filebuffer)
        structure = dictionary['structure']
        weights = dictionary['weights']
        return cls.init_with_weights(structure, weights)

    @property
    def all_neurons(self) -> list[Neuron]:
        return list(chain(*[layer.neurons for layer in self.layers]))

    @property
    def all_weights(self) -> list[int]:
        return list(chain(*[neuron.weights for neuron in self.all_neurons]))

    def __eq__(self, o):
        return self.layers == o.layers

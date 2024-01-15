from json import dump, load

from numpy import array, ndarray

from layer import Layer
from utils import check_dir_path_slash_ending


class Perceptron:
    def __init__(self, structure: list[int]):
        if len(structure) <= 2:
            raise ValueError('Perceptron must have at least 2 layers')
        # first layer (index=0) of the structure isn't a true layer,
        # but the next layer neuron's inputs number.
        self.layers = list()
        for layer_number, neurons_number in enumerate(structure[1:]):
            self.layers.append(Layer(neurons_number, structure[layer_number]))

    def __call__(self, inputs_values: ndarray) -> ndarray:
        resoults = inputs_values
        for layer in self.layers:
            resoults = layer(resoults)
        return resoults

    def __repr__(self) -> str:
        return f'< Perceptron: {self.structure} >'

    @property
    def structure(self) -> list[int]:
        structure = list()
        for number, layer in enumerate(self.layers):
            if number == 0:
                structure.append(layer.matrix.shape[1] - 1)
            structure.append(layer.matrix.shape[0])
        return structure

    def save(self, dir_path, file_name) -> str:
        check_dir_path_slash_ending(dir_path)
        file = dir_path + file_name + '.perceptron'
        with open(file, mode='w', encoding='ascii') as filebuffer:
            dump(
                obj=[layer.matrix.tolist() for layer in self.layers],
                fp=filebuffer,
            )
        return file

    @classmethod
    def load(cls, file: str):
        with open(file, mode='r', encoding='ascii') as filebuffer:
            loaded: list = load(fp=filebuffer)
        perceptron = object.__new__(cls)
        perceptron.layers = list()
        for matrix in loaded:
            layer = object.__new__(Layer)
            layer.matrix = array(matrix)
            perceptron.layers.append(layer)
        return perceptron

    def __eq__(self, o) -> bool:
        return self.layers == o.layers


# Testing
if __name__ == '__main__':
    perceptron = Perceptron([3, *6*[10**4], 10])
    print(perceptron(array([0, 1, 1])))

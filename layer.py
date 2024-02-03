from numpy import array, array_equal, delete, exp, insert, ndarray, sum
from numpy.random import uniform


def sigmoid(x):
    return 1 / (1 + exp(-x))


class Layer:
    def __init__(self, neurons_number: int, neuron_inputs_number: int):
        if not neurons_number:
            raise ValueError('Layer must have at least a neuron')
        if neuron_inputs_number < 2:
            raise ValueError('Neuron must have at least two non-bias inputs')
        self.matrix = uniform(
            low=-1,
            high=1,
            # `neuron_inputs_number + 1` is adding of weight for bias input
            size=(neurons_number, neuron_inputs_number + 1,),
        )

    def __call__(self, inputs_values: ndarray) -> ndarray:
        # Insert bias input
        inputs_values = insert(arr=inputs_values, obj=0, values=[1])
        weighted_inputs = inputs_values * self.matrix
        return sigmoid(sum(weighted_inputs, axis=1))

    def __repr__(self):
        return f'\n< Layer with {self.neurons_number} neurons >'

    @property
    def neurons_number(self):
        return self.matrix.shape[0]

    @property
    def each_neuron_weights_number(self):
        return self.matrix.shape[1]

    def _insert_neuron(self, index: int):
        neuron_inputs_number = self.matrix.shape[1]
        self.matrix = insert(
            arr=self.matrix,
            obj=index,
            axis=0,
            values=uniform(low=-1, high=1, size=(neuron_inputs_number,)),
        )

    def _add_neuron(self):
        self._insert_neuron(index=self.matrix.shape[0])

    def _delete_neuron(self, index: int):
        self.matrix = delete(arr=self.matrix, obj=index, axis=0)

    def _delete_last_neuron(self):
        self._delete_neuron(self.neurons_number - 1)

    def _add_weights(self):
        self.matrix = insert(
            arr=self.matrix,
            obj=self.matrix.shape[1],
            axis=1,
            values=uniform(low=-1, high=1, size=(self.matrix.shape[0])),
        )

    def _delete_weights(self, index: int):
        self.matrix = delete(arr=self.matrix, obj=index, axis=1)

    def _delete_last_weights(self):
        self._delete_weights(self.each_neuron_weights_number - 1)

    def _change_weight(
        self, neuron_index: int, weight_index: int, new_walue: float,
    ):
        self.matrix[neuron_index: weight_index] = new_walue

    def _read_weight(self, neuron_index: int, weight_index: int):
        return self.matrix[neuron_index][weight_index]

    def __eq__(self, o):
        return array_equal(self.matrix, o.matrix)


if __name__ == '__main__':
    layer = Layer(5, 3)
    print(layer(array([789, 90, 6])))

from numpy import array, array_equal, delete, exp, insert, ndarray, sum
from numpy.random import uniform


VALUES_RANGE = (-1, 1,)


def sigmoid(x):
    return 1 / (1 + exp(-x))


class Layer:
    def __init__(self, inputs_number: int = 1, outputs_number: int = 1):
        self.matrix = uniform(
            *VALUES_RANGE,
            # `neuron_inputs_number + 1` is adding of weight for bias
            size=(outputs_number, inputs_number + 1,),
        )

    def __call__(self, inputs_values: ndarray) -> ndarray:
        # Insert bias input
        inputs_values = insert(arr=inputs_values, obj=0, values=[1])

        weighted_inputs = inputs_values * self.matrix
        return sigmoid(sum(weighted_inputs, axis=1))

    def __repr__(self):
        return f'< Layer {self.inputs_number} -> {self.outputs_number} >'

    @property
    def outputs_number(self):
        return self.matrix.shape[0]

    @property
    def inputs_number(self):
        return self.matrix.shape[1] - 1

    def insert_output(self, index: int):
        inputs_number = self.matrix.shape[1]
        self.matrix = insert(
            arr=self.matrix,
            obj=index,
            axis=0,
            values=uniform(*VALUES_RANGE, size=(inputs_number,)),
        )

    def append_output(self):
        self.insert_output(index=self.matrix.shape[0])

    def delete_output(self, index: int):
        if self.outputs_number == 1:
            raise Exception('Can not delete last output')
        self.matrix = delete(arr=self.matrix, obj=index, axis=0)

    def _pop_output(self):
        if self.outputs_number == 1:
            raise Exception('Can not delete last output')
        self.delete_output(self.outputs_number - 1)

    def append_input(self):
        self.matrix = insert(
            arr=self.matrix,
            obj=self.matrix.shape[1],
            axis=1,
            values=uniform(*VALUES_RANGE, size=(self.matrix.shape[0])),
        )

    def delete_input(self, index: int):
        # `index+1` for we never could delete the bias
        if self.inputs_number == 1:
            raise Exception('Can not delete last non-bias input')
        self.matrix = delete(arr=self.matrix, obj=index+1, axis=1)

    def pop_input(self):
        if self.inputs_number == 1:
            raise Exception('Can not delete last non-bias input')
        self.delete_input(self.matrix.shape[1] - 1)

    def write_weight(
        self, input_index_with_bias: int, output_index: int, new_walue: float,
    ):
        self.matrix[output_index: input_index_with_bias] = new_walue

    def read_weight(
        self, input_index_with_bias: int, output_index: int,
    ) -> float:
        return self.matrix[output_index][input_index_with_bias]

    def __eq__(self, o) -> bool:
        return array_equal(self.matrix, o.matrix)


if __name__ == '__main__':
    layer = Layer(3, 5)
    print(layer)
    print(layer(array([789, 90, 6])))

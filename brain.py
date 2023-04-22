from json import dump, load
from random import choice
from time import time

from perceptron import Perceptron
from utils import (conv_int_to_list, conv_list_to_int, dict_sum,
                   get_unicode_characters_by_ranges)


STANDDARD_STRUCTURE = dict(
    inputs_structure=dict(
        character_inputs=8,
        time=6*8,
        next_reflection=1,
    ),
    outputs_structure=dict(
        controlling_signal=3,
        character_outputs=8,
        add_delete_neuron_signal=2,
        add_delete_memory_signal=2,
    ),
    memory_cell_structure=dict(
        layer_outputs=10,
        neuron_outputs=10,
        weight_outputs=10,
        new_value_outputs=1,
    )
)

UNICODE_RANGES = [
    (0x0020, 0x007E),  # ASCII punctuation and symbols, digits, latin
    (0x0400, 0x045F),  # Cyrilic
]


class Brain:
    def __init__(
        self, 
        saving_file_path: str, middle_layers_structure: list = 6*[10],
        memory_cells_number: int = 1, structure: dict = STANDDARD_STRUCTURE,
        coding_string=get_unicode_characters_by_ranges(UNICODE_RANGES),
    ):
        self.saving_file_path = saving_file_path

        self.inputs_structure = structure['inputs_structure']
        self.outputs_structure = structure['outputs_structure']
        self.memory_cell_structure = structure['memory_cell_structure']

        self.coding_string = coding_string
        perceptron = Perceptron(
            structure=[
                dict_sum(self.inputs_structure),
                *middle_layers_structure,
                memory_cells_number * dict_sum(self.memory_cell_structure)\
                + dict_sum(self.outputs_structure),
            ],
        )
        with open(
            file=self.saving_file_path, mode='w', encoding='ascii',
        ) as filebuffer:
            dump(
                obj=dict(
                    middle_layers_structure=middle_layers_structure,
                    memory_cells_number=memory_cells_number,
                    weights=[wgt.value for wgt in perceptron.all_weights],
                ),
                fp=filebuffer,
            )

    @property
    def perceptron(self):
        with open(
            file=self.saving_file_path, mode='r', encoding='ascii',
        ) as filebuffer:
            dictionary = load(fp=filebuffer)
        perceptron = Perceptron(
            structure=[
                dict_sum(self.inputs_structure),
                *dictionary['middle_layers_structure'],
                dictionary['memory_cells_number'] * dict_sum(self.memory_cell_structure)\
                + dict_sum(self.outputs_structure),
            ],
        )
        for position, weight in enumerate(perceptron.all_weights):
            weight.value = dictionary['weights'][position]
        return perceptron

    def save(self, file: str) -> str:
        with open(file=file, mode='w', encoding='ascii') as filebuffer:
            dump(
                obj=dict(
                    middle_layers_structure=self.middle_layers_structure,
                    memory_cells_number=self.memory_cells_number,
                    weights=[wgt.value for wgt in self.perceptron.all_weights],
                ),
                fp=filebuffer,
            )
        return file

    @classmethod
    def load(cls, file: str):
        with open(file=file, mode='r', encoding='ascii') as filebuffer:
            dictionary = load(fp=filebuffer)
        brain = cls(
            middle_layers_structure=dictionary['middle_layers_structure'],
            memory_cells_number=dictionary['memory_cells_number']
        )
        for position, weight in enumerate(brain.perceptron.all_weights):
            weight.value = dictionary['weights'][position]
        return brain
    
    def count_loss(self, loss_function, dataset):
        shuffle(dataset)
        losses, speeds, lengths = list(), list(), list()
        for case in dataset:
            time_limit = case['T']
            request = case['Q'].replace('%T%', str(time_limit))
            waited_response = case['A']

            start = time()
            real_response = self(request, time_limit=time_limit)
            finish = time()
            real_time = int(finish - start)

            try:
                print(f'RESPONSE {real_time}/{time_limit} sec: {real_response}')
            except UnicodeEncodeError:
                pass
            responses_lengths = [len(waited_response), len(real_response)]

            losses.append(loss_function(waited_response, real_response))
            speeds.append(real_time / time_limit)
            lengths.append(
                max(responses_lengths) / (min(responses_lengths) + 1),
            )
        self.loss = mean(losses)
        print(f'LOSS: {self.loss} ', end='')
        self.speed = mean(speeds)
        print(f'SPEED: {self.speed} ', end='')
        self.length = mean(lengths)
        print(f'LENGTH: {self.length} ')

    def __call__(self, request: str, time_limit=None, verbose: bool = False):
        # signals
        stop_signal = [0, 0, 1]
        skip_signal = [0, 1, 0]
        repeat_signal = [1, 0, 0]
        stop_reflections_signal = [1, 1, 1]
        add_neuron = add_memory = [0, 0]
        delete_neuron = delete_memory = [1, 1]

        def memory_cells_number():
            memory_outputs = perceptron.structure[-1]\
            - dict_sum(self.outputs_structure)
            cells_number = memory_outputs / dict_sum(self.memory_cell_structure)
            return int(cells_number)

        def verbose(string: str, verbose=verbose):
            if verbose:
                print(string)
            else:
                pass

        def split_by_volumes(list_for_split: list, volumes: list):
            resoult_lists = list()
            for volume in volumes:
                resoult_lists.append(list_for_split[:volume])
                list_for_split = list_for_split[volume:]
            resoult_lists.append(list_for_split)
            return resoult_lists

        def get_raw_outputs(request_character: str, next_reflection: bool):
            char_binary_list = conv_int_to_list(
                number=self.coding_string.find(request_character),
                width=self.inputs_structure['character_inputs'],
            )
            time_binary_list = conv_int_to_list(
                number=int(time()),
                width=self.inputs_structure['time'],
            )
            return perceptron(
                [*char_binary_list, *time_binary_list, int(next_reflection)],
            )

        def change_weights(memory_outputs: list):
            for _ in range(memory_cells_number()):
                (layer_outputs, neuron_outputs, weight_outputs,
                    new_value_outputs, memory_outputs) = split_by_volumes(
                        list_for_split=memory_outputs,
                        volumes=self.memory_cell_structure.values(),
                )
                self.perceptron.change_weight(
                    layer_fraction=1/(conv_list_to_int(layer_outputs) + 1),
                    neuron_fraction=1/(conv_list_to_int(neuron_outputs) + 1),
                    weight_fraction=1/(conv_list_to_int(weight_outputs) + 1),
                    new_value=[-1, 1][new_value_outputs.pop()],
                )
                verbose('Weight is changed')

        def random_middle_layer_index() -> int:
            return choice(list(range(1, len(self.layers))))

        def add_or_delete_neuron(add_delete_neuron):
            if add_delete_neuron == add_neuron:
                self.perceptron.add_neuron(random_middle_layer_index())
                verbose('Neuron is added')

            if add_delete_neuron == delete_neuron:
                self.perceptron.delete_neuron(random_middle_layer_index())
                verbose('Neuron is deleted')

        def add_or_delete_memory(add_delete_memory):
            last_layer_index = len(self.layers)

            if add_delete_memory == add_memory:
                for _ in range(dict_sum(self.memory_cell_structure)):
                    self.perceptron.add_neuron(last_layer_index)
                verbose('Memory is added')

            if add_delete_memory == delete_memory and memory_cells_number()>1:
                for _ in range(dict_sum(self.memory_cell_structure)):
                    self.perceptron.delete_neuron(last_layer_index)
                verbose('Memory is deleted')

        def one_step(request_character: str, next_reflection: bool):
            outputs = get_raw_outputs(request_character, next_reflection)

            controlling_signal, character_outputs, add_delete_neuron,\
                add_delete_memory, memory_outputs = split_by_volumes(
                    list_for_split=outputs,
                    volumes=self.outputs_structure.values(),
                )
            change_weights(memory_outputs)
            add_or_delete_neuron(add_delete_neuron)
            add_or_delete_memory(add_delete_memory)

            response_character = ''
            if controlling_signal != skip_signal:  # Not skip
                raw_index = conv_list_to_int(character_outputs)
                maximal_index = len(self.coding_string) - 1
                if raw_index <= maximal_index:
                    response_character = self.coding_string[raw_index]
            else:
                verbose('Skipped')
                response_character = ''
            return response_character, controlling_signal

        if time_limit:
            start_time = time()
        initial_request = request
        reflection = False
        r = 0
        while True:  # Reflections
            if r > 0:
                verbose('Next Reflection')
            resoults = list()
            for request_character in request:
                repeat = True
                while repeat:
                    verbose(f'Request caracter now is {request_character}')
                    step_resoults = one_step(request_character, reflection)
                    response_character, controlling_signal = step_resoults

                    resoults.append(response_character)

                    if time_limit and time_limit < time() - start_time:
                        verbose('Stopped py time limit')
                        controlling_signal = stop_signal

                    if controlling_signal == stop_signal:
                        verbose('Stop')
                        break

                    if controlling_signal == skip_signal:
                        break

                    if controlling_signal != repeat_signal:
                        repeat = False
                    else:
                        verbose('Repeated')

                    if r > 0 and controlling_signal == stop_reflections_signal:
                        verbose('Stop Reflections')
                        break
                if controlling_signal == stop_signal:
                    break
                if r > 0 and controlling_signal == stop_reflections_signal:
                    break
            if controlling_signal == stop_signal:
                break
            request = ''.join(resoults)
            if r > 0 and controlling_signal == stop_reflections_signal:
                request = initial_request
                r = 0
            elif request == '':
                verbose('Request is empty')
                request = initial_request
                r = 0
            else:
                r += 1
            reflection = not reflection
        return ''.join(resoults)

    def copy_by_structure(self, saving_file_path):
        return self.__class__(
            saving_file_path=saving_file_path,
            middle_layers_structure=self.middle_layers_structure,
            memory_cells_number=self.memory_cells_number,
        )

    @property
    def all_weights(self):
        return self.perceptron.all_weights

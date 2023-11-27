from json import dump, load
from operator import attrgetter
from pathlib import Path
from random import choice, choices, shuffle, uniform
from string import ascii_letters
from time import time

from recurent import Brain
from utils import mean, split_by_evenodd_position


class BrainsPopulation:
    def __init__(self, size: int, saving_dir: str):
        def random_token():
            return ''.join(choices(list(ascii_letters), k=10))

        def path(token):
            return f'{saving_dir}save_{token}.json'

        self.brains = list()
        for _ in range(size):
            token = random_token()
            saving_file_path = path(token)
            while Path(saving_file_path).exists():
                token = random_token()
                saving_file_path = path(token)
            new_brain = Brain(saving_file_path)
            self.brains.append(new_brain)
            print(f'\nPOPULATION MEMBER {token} IS CREATED')

    @classmethod
    def load(cls, saving_dir: str):
        print('\nLOADING:')
        brains = list()
        for filepath in Path(saving_dir).iterdir():
            brain = Brain.load(saving_file_path=filepath)
            brains.append(brain)
        population = cls.__new__(cls)
        population.brains = brains
        print('POPULATION LOADED')
        return population

    def tich(
        self, loss_function, dataset: list,
        mortality: float=0.4, loss: float=0.25, mutability: float=0.2,
        with_cross_over=False, save_population_path=None,
    ):
        while True:
            # count errors and sort neuronets by the errors
            self.count_losses(loss_function, dataset)

            # sort all population neuronrts after its losses counting
            self.sort()

            # stop the tiching and return resouult if goal loss reached
            if self.best_neuronet.loss < loss:
                return self.best_neuronet

            # Killing worst neuronets
            dead_nets_number = round(mortality * self.size)
            survived_nets_number = self.size - dead_nets_number
            self.change_size_to(survived_nets_number)

            # breeding
            if with_cross_over:
                couples = self._form_couples(dead_nets_number)
                children = self._make_children(couples, mutability)
            else:
                children = self._make_mutants(dead_nets_number, mutability)

            # children adding to population
            self.brains += children

            if save_population_path:
                self.save(file=save_population_path)

    def change_size_to(self, neuronets_number):
        if self.size > neuronets_number:
            print('\nPOPULATION REDUCTION')
            self.brains = self.brains[:neuronets_number]

        if self.size < neuronets_number:
            print('\nPOPULATION ADDITION')
            size = neuronets_number - self.size
            neuronet = self.best_neuronet
            additional_neuronets = self.__class__(size, neuronet).brains
            self.brains.extend(additional_neuronets)

    @property
    def size(self) -> int:
        return len(self.brains)

    @property
    def best_neuronet(self) -> object:
        # best resoult is first element in sorted neuronets list:
        return self.brains[0]

    def _make_children(self, couples: list, mutability: float) -> list:
        children = list()
        for net_1, net_2 in couples:
            children.append(self._cross_over(net_1, net_2, mutability))
        return children

    def _cross_over(self, neuronet_1, neuronet_2, mutability: float):
        child = neuronet_1.copy_by_structure()
        for position, weight in enumerate(child.all_weights):
            if mutability < uniform(0, 1):
                continue
            else:
                weight.value = choice([neuronet_1, neuronet_2])[position].value
        return child

    def _form_couples(self, couples_number: int) -> list:
        couples_list = list(zip(*split_by_evenodd_position(self.neuronets)))
        return couples_list[:couples_number]

    def _make_mutants(self, number: int, mutability: float):
        print('\nCHILDREN MAKING:')
        children = list()
        while len(children) != number:
            for neuronet in self.neuronets:
                child = neuronet.copy_by_structure()
                nuronet_weights = neuronet.all_weights
                for position, weight in enumerate(child.all_weights):
                    if mutability < uniform(0, 1):
                        continue
                    else:
                        weight.value = nuronet_weights[position].value
                children.append(child)
                print(f'CHILD: {len(children)}/{number} ', end='')
                print(f'PARENT LOSS: {neuronet.loss} ', end='')
                print(f'PARENT SPEED: {neuronet.speed} ', end='')
                print(f'PARENT LENGTH: {neuronet.length} ')
                if len(children) == number:
                    break
        return children

    def sort(self):
        self.neuronets.sort(key=attrgetter('length'))
        self.neuronets.sort(key=attrgetter('speed'))
        self.neuronets.sort(key=attrgetter('loss'))

    def count_losses(self, loss_function, dataset):
        print('\nLOSSES COUNTING:')
        for number, brain in enumerate(self.brains):
            shuffle(dataset)
            brain.count_loss(loss_function, dataset)
            print(f'LOSSES COUNTING: {number + 1}/{self.size} ', end='')

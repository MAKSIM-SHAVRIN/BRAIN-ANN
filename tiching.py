from population import BrainsPopulation
from utils import jaro_loss
from pathlib import Path
from json import load


# Testing;
if __name__ == '__main__':
    project_path = Path(__file__).parent.as_posix()

    dataset_path = f'{project_path}/dataset_1.json'
    with open(dataset_path, mode='r') as filebuffer:
        dataset = load(fp=filebuffer)

    saving_file = f'{project_path}/save.json'

    population = BrainsPopulation.load(saving_file)
    population.change_size_to(neuronets_number=100)
    population.tich(
        loss_function=jaro_loss,
        dataset=dataset,
        mortality=0.7,
        mutability=0.05,
        save_population_path=saving_file,
    )

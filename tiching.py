from json import load
from pathlib import Path

from population import BrainsPopulation
from utils import jaro_loss


# Testing;
if __name__ == '__main__':
    project_path = Path(__file__).parent.as_posix()

    dataset_path = f'{project_path}/dataset_1.json'
    with open(dataset_path, mode='r') as filebuffer:
        dataset = load(fp=filebuffer)

    saving_dir = f'{project_path}/population_save'

    population = BrainsPopulation.load(saving_dir)
    population.tich(
        loss_function=jaro_loss,
        dataset=dataset,
        mortality=0.7,
        mutability=0.05,
        save_population_path=saving_dir,
    )

import copy

import numpy as np
import pandas as pd
import logging
import random
import time
import matplotlib.pyplot as plt

# Item Class
class Item:
    value = 0
    weight = 0
    ratio = 0

    def __init__(self, value, weight):
        self.value = value
        self.weight = weight
        self.ratio = value / weight

    def __repr__(self):
        return f"Item Value {self.value}, Item Weight: {self.weight}"


# The capacity of the knapsack must be kept at 1500.
class Knapsack:
    capacity = 1500
    content = []
    value = 0

    def __init__(self):
        self.capacity = 1500
        self.content = []
        self.value = 0

    def put(self, item: Item):
        if item.weight > self.capacity:
            logging.error("Capacity exceeded!")
            return False
        else:
            self.content.append(item)
            self.capacity -= item.weight
            self.value += item.value
            return True

    def pop(self, item):
        self.content.remove(item)
        self.capacity += item.weight
        self.value -= item.value
        assert (self.capacity >= 0)

    def pop_random(self):
        if len(self.content) < 1:
            print("Something is wrong here!")
        random_item = random.choice(self.content)
        self.pop(random_item)
        return random_item

    def __copy__(self):
        ks = Knapsack()
        ks.capacity = self.capacity
        ks.value = self.value
        ks.content = list(self.content)
        return ks

    def __repr__(self):
        return f"Remaining Capacity: {self.capacity}, Total Value: {self.value}, Number of Items: {len(self.content)}"


def random_init(sack: Knapsack, items: list):
    logging.info("Initializing Knapsack with Random Items")
    has_capacity = True
    while has_capacity:
        random_item = random.choice(items)
        if random_item.weight < sack.capacity:
            logging.debug(f"Adding {random_item}")
            sack.put(item=random_item)
            items.remove(random_item)
        else:
            has_capacity = False
    logging.info(f"Initial Solution: {sack}")
    return sack


def greedy_init(sack: Knapsack, items: list):
    logging.info("Initializing Knapsack with Greedy Algorithm")
    sorted_items = sorted(items, key=lambda item: item.ratio, reverse=True)
    for item in sorted_items:
        if item.weight < sack.capacity:
            sack.put(item)
            items.remove(item)
    logging.info(f"Initial Solution: {sack}")
    return sack


def steepest_assent_hill_climb(sack: Knapsack, items: list, iter=1000, cutoff=250):
    # Remove a random item from the knapsack
    iteration = 0
    for i in range(0, iter):
        if iteration < cutoff:
            new_solution = copy.copy(sack)
            new_solution.pop_random()
            new_items = copy.copy(items)
            filtered_items = list(filter(lambda item: item.weight < new_solution.capacity, items))
            sorted_items = sorted(filtered_items, key=lambda item: item.ratio, reverse=True)
            # Add items with best ratio
            for item in sorted_items:
                if item.weight < new_solution.capacity:
                    new_solution.put(item)
                    new_items.remove(item)
            if new_solution.value > sack.value:
                logging.debug(f"Better solution found, Old value: {sack.value}, new value: {new_solution.value}")
                sack = new_solution
                items = new_items
                iteration = 0
            else:
                # Number of iterations without any improvement
                iteration = iteration + 1
        else:
            logging.info(f"No better solutions found for {cutoff} iterations")
            break
        cost_history.append(sack.value)
    return sack, items


def simulated_annealing(sack: Knapsack, items: list, tmax, tmin, cooling_rate):
    t = tmax
    best_solution = copy.copy(sack)
    best_items = list(items)
    while t >= tmin:
        iteration = 0
        while iteration < 10:
            # Generate new solution
            new_solution = copy.copy(sack)
            new_items = list(items)
            removed_item = new_solution.pop_random()
            # Generate Neighbour Solution
            filtered_items = list(filter(lambda item: item.weight < new_solution.capacity, new_items))
            while len(filtered_items) > 0:
                random_item = random.choice(filtered_items)
                assert (new_solution.put(random_item))
                new_items.remove(random_item)
                filtered_items = list(filter(lambda item: item.weight < new_solution.capacity, new_items))
            new_items.append(removed_item)
            assert (len(new_solution.content) + len(new_items) == len(dataset))
            # Cost Difference
            dE = sack.value - new_solution.value
            # if new solution is better
            if dE < 0:
                logging.debug(f"Better Solution: Iteration {iteration}, Sack: {new_solution}, Items: {len(new_items)}")
                sack = new_solution
                items = new_items
                if new_solution.value > best_solution.value:
                    best_solution = new_solution
            # if the new solution is worse
            else:
                # Probability of accepting worse results
                probability = np.exp(-dE / t)
                randomity = random.random()
                if randomity < probability:
                    logging.debug(
                        f"Worse Solution: Iteration {iteration}, Sack: {new_solution}, Items: {len(new_items)}")
                    sack = new_solution
                    items = new_items
            iteration += 1
        t *= cooling_rate
        cost_history.append(new_solution.value)
    logging.info(f"Best solution {best_solution}")
    return sack, items


if __name__ == '__main__':
    # Settings
    logging.basicConfig(level=logging.WARNING)
    dataset = pd.read_csv("knapsack.csv")
    values = dataset.get("values")
    weights = dataset.get("weights")
    # List to hold all items outside Knapsack
    item_list = []

    # Add all items to item_list
    for i in range(len(dataset)):
        item_list.append(Item(values[i], weights[i]))

    items = list(item_list)
    for optimiser in ["sahc", "sa"]:
        for init in ["random", "greedy"]:

            for seed in [0, 3, 26, 32, 65, 33, 27, 12, 42]:
                logging.info("----------------------")
                random.seed(seed)
                logging.info(f"Random seed: {seed}")
                cost_history = []
                item_list = list(items)
                # Create a Knapsack object
                knapsack = Knapsack()

                start_time = time.perf_counter()
                if init == "random":
                    knapsack = random_init(knapsack, item_list)
                elif init == "greedy":
                    knapsack = greedy_init(knapsack, item_list)
                else:
                    logging.error("Wrong Initialiser")

                cost_history.append(knapsack.value)

                if optimiser == "sahc":
                    knapsack, item_list = steepest_assent_hill_climb(knapsack, item_list, iter=1000, cutoff=250)
                elif optimiser == "sa":
                    knapsack, item_list = simulated_annealing(knapsack, item_list, tmax=1000, tmin=0.001, cooling_rate=0.995)
                else:
                    logging.error("Wrong Optimiser")

                logging.info(f"Final Solution: {knapsack}")
                end_time = time.perf_counter()
                # Calculate the elapsed time
                elapsed_time = end_time - start_time
                logging.info(f"Total time elapsed: {elapsed_time * 1000} (ms)")
                plt.plot(cost_history, label=f'Seed {seed}')
                logging.info(f"Summary: ")
                print(f"{seed}\t{init}\t{optimiser}\t{elapsed_time}\t{len(cost_history)}\t{cost_history[0]}\t{knapsack.value}")
            plt.xlabel('Iterations')
            plt.ylabel('Knapsack Value')
            plt.legend()
            # plt.show()
            plt.savefig(f"{init}_init_{optimiser}.png", dpi=300)
            plt.clf()

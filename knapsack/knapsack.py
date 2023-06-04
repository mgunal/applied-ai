import copy

import numpy as np
import pandas as pd
import logging
import random
import time
import matplotlib.pyplot as plt
import hashlib

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
    items = []

    def __init__(self, item_list):
        self.capacity = 1500
        self.content = []
        self.value = 0
        self.items = list(item_list)

    def put(self, item: Item):
        if item.weight > self.capacity:
            logging.error("Capacity exceeded!")
            return False
        else:
            self.content.append(item)
            self.capacity -= item.weight
            self.value += item.value
            self.items.remove(item)
            return True

    def pop(self, item):
        self.content.remove(item)
        self.capacity += item.weight
        self.value -= item.value
        self.items.append(item)
        assert (self.capacity >= 0)

    def generate_neighbourhood(self, num=10):
        neighborhood = []
        for _ in range(num):
            neighbor = copy.copy(self)
            neighbor.pop_random()
            filtered_items = list(filter(lambda item: item.weight < neighbor.capacity, neighbor.items))
            while len(filtered_items) > 0:
                random_item = random.choice(filtered_items)
                assert (neighbor.put(random_item))
                filtered_items = list(filter(lambda item: item.weight < neighbor.capacity, neighbor.items))
            neighborhood.append(neighbor)
        return neighborhood

    def pop_random(self):
        if len(self.content) < 1:
            print("Something is wrong here!")
        random_item = random.choice(self.content)
        self.pop(random_item)
        return random_item

    def __copy__(self):
        ks = Knapsack(self.items)
        ks.capacity = self.capacity
        ks.value = self.value
        ks.content = list(self.content)
        return ks

    def __repr__(self):
        return f"Remaining Capacity: {self.capacity}, Total Value: {self.value}, Number of Items: {len(self.content)}"

    def __hash__(self):
        string = ''
        string += f"{self.capacity}"
        string += f"{self.value}"
        string += f"{self.content}"
        hash_object = hashlib.sha256(string.encode())
        hash_value = hash_object.hexdigest()
        return hash_value


def random_init(sack: Knapsack):
    logging.info("Initializing Knapsack with Random Items")
    has_capacity = True
    while has_capacity:
        random_item = random.choice(knapsack.items)
        if random_item.weight < sack.capacity:
            logging.debug(f"Adding {random_item}")
            sack.put(item=random_item)
        else:
            has_capacity = False
    logging.info(f"Initial Solution: {sack}")
    return sack


def greedy_init(sack: Knapsack):
    logging.info("Initializing Knapsack with Greedy Algorithm")
    sorted_items = sorted(sack.items, key=lambda item: item.ratio, reverse=True)
    for item in sorted_items:
        if item.weight < sack.capacity:
            sack.put(item)
    logging.info(f"Initial Solution: {sack}")
    return sack


def steepest_assent_hill_climb(sack: Knapsack,iter=1000, cutoff=250):
    # Remove a random item from the knapsack
    iteration = 0
    for i in range(0, iter):
        if iteration < cutoff:
            new_solution = copy.copy(sack)
            new_solution.pop_random()
            filtered_items = list(filter(lambda item: item.weight < new_solution.capacity, new_solution.items))
            sorted_items = sorted(filtered_items, key=lambda item: item.ratio, reverse=True)
            # Add items with best ratio
            for item in sorted_items:
                if item.weight < new_solution.capacity:
                    new_solution.put(item)
            if new_solution.value > sack.value:
                logging.debug(f"Better solution found, Old value: {sack.value}, new value: {new_solution.value}")
                sack = new_solution
                iteration = 0
            else:
                # Number of iterations without any improvement
                iteration = iteration + 1
        else:
            logging.info(f"No better solutions found for {cutoff} iterations")
            break
        cost_history.append(sack.value)
    return sack


def simulated_annealing(sack: Knapsack, tmax, tmin, cooling_rate):
    t = tmax
    best_solution = copy.copy(sack)
    while t >= tmin:
        iteration = 0
        while iteration < 20:
            # Generate neighbourhood solution
            new_solution = sack.generate_neighbourhood(1)[0]
            # Cost Difference
            dE = sack.value - new_solution.value
            # if new solution is better
            if dE < 0:
                logging.debug(f"Better Solution: Iteration {iteration}, Sack: {new_solution}, Items: {len(new_solution.items)}")
                sack = new_solution
                if new_solution.value > best_solution.value:
                    best_solution = new_solution
            # if the new solution is worse
            else:
                # Probability of accepting worse results
                probability = np.exp(-dE / t)
                randomity = random.random()
                if randomity < probability:
                    logging.debug(
                        f"Worse Solution: Iteration {iteration}, Sack: {new_solution}, Items: {len(new_solution.items)}")
                    sack = new_solution
            iteration += 1
        t *= cooling_rate
        cost_history.append(new_solution.value)
    logging.info(f"Best solution {best_solution}")
    return sack


def tabu_search(sack: Knapsack, tabu_size=10, iter=1000, cutoff=250):
    tabu_list = []
    global_best_solution = sack
    local_best_solution = global_best_solution

    iteration = 0
    for _ in range(iter):
        if iteration < cutoff:
            # Generate new solutions and select the best one
            neighbourhood = local_best_solution.generate_neighbourhood(10)
            local_best_solution = select_best_neighbor(neighbourhood, tabu_list)

            # Update the best solution if a better neighbor is found
            if local_best_solution.value > global_best_solution.value:
                global_best_solution = local_best_solution
                iteration = 0

            # Add the best neighbor to the tabu list
            tabu_list.append(local_best_solution.__hash__())
            if len(tabu_list) > tabu_size:
                tabu_list.pop(0)

            iteration += 1
            cost_history.append(local_best_solution.value)
        else:
            logging.info(f"No better solutions found for {cutoff} iterations")
            break
        #cost_history.append(local_best_solution.value)

    return global_best_solution


def select_best_neighbor(neighborhood: list, tabu_list: list):
    best_neighbor = None
    best_value = float('-inf')
    for neighbor in neighborhood:
        if neighbor.__hash__() in tabu_list:
            logging.debug(f"Item is in tabu list, skipping")
        elif neighbor.value > best_value:
            best_neighbor = neighbor
            best_value = neighbor.value
    return best_neighbor


def get_remaining_items(items: list, sack: Knapsack):
    remaining_items = []
    for item in items:
        if item.weight <= sack.capacity:
            remaining_items.append(item)
    return remaining_items



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

    for optimiser in ["ts", "sahc", "sa"]:
        for init in ["random"]: #, "greedy"]:
            for seed in [0, 3, 26, 32, 65, 33, 27, 12, 42]:
                logging.info("----------------------")
                random.seed(seed)
                logging.info(f"Random seed: {seed}")
                cost_history = []
                # Create a Knapsack object
                assert(len(item_list) == len(dataset))
                knapsack = Knapsack(item_list)

                start_time = time.perf_counter()
                if init == "random":
                    knapsack = random_init(knapsack)
                elif init == "greedy":
                    knapsack = greedy_init(knapsack)
                else:
                    logging.error("Wrong Initialiser")

                cost_history.append(knapsack.value)

                if optimiser == "sahc":
                    knapsack = steepest_assent_hill_climb(knapsack, iter=1000, cutoff=250)
                elif optimiser == "sa":
                    knapsack = simulated_annealing(knapsack, tmax=100, tmin=0.001, cooling_rate=0.995)
                elif optimiser == "ts":
                    knapsack = tabu_search(knapsack, tabu_size=50, iter=1000, cutoff=250)
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

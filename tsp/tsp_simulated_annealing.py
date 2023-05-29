import numpy as np
import random
import logging
import matplotlib.pyplot as plt

# Load Cost Matrix
cost_matrix = np.loadtxt('tsp_matrix.txt', delimiter=',')
cost_history = []

# Settings
random.seed(42)
logging.basicConfig(level=logging.ERROR)


def calculate_cost(path):
    cost = 0
    for i in range(0, len(path) - 1):
        start = path[i]
        end = path[i + 1]
        cost += cost_matrix[start, end]
    # Add cost from last node to the starting node
    cost += cost_matrix[path[-1], path[0]]
    return cost


def random_init():
    init_route = list(range(50))
    random.shuffle(init_route)
    return init_route


def greedy_init():
    matrix = np.copy(cost_matrix)
    init_route = []
    np.fill_diagonal(matrix, np.inf)
    start = random.randint(0, 49)
    matrix[:, start] = np.inf
    init_route.append(start)
    current = start
    for i in range(len(matrix) - 1):
        # Find the nearest city
        nearest = np.argmin(matrix[current, :])
        # Add it to the route
        init_route.append(nearest)
        current = nearest
        # matrix[start, :] = np.inf
        matrix[:, current] = np.inf
    unique_elements = np.unique(init_route)
    assert (len(init_route) == len(unique_elements))
    return init_route


def simple_hill_climb(route):
    current_cost = calculate_cost(route)
    swap_id = random.randint(0, len(route) - 2)
    # print(f"Swapping: {swap_id}, {swap_id+1}")
    alt_route = list(route)
    alt_route[swap_id], alt_route[swap_id + 1] = alt_route[swap_id + 1], alt_route[swap_id]
    new_cost = calculate_cost(alt_route)
    if new_cost < current_cost:
        route = list(alt_route)
        assert (len(route) == len(np.unique(route)))
    return route, new_cost


def steepest_hill_climb(route):
    current_cost = calculate_cost(route)
    for i in range(len(route) - 1):
        swap_id = i
        alt_route = list(route)
        alt_route[swap_id], alt_route[swap_id + 1] = alt_route[swap_id + 1], alt_route[swap_id]
        new_cost = calculate_cost(alt_route)
        if new_cost < current_cost:
            route = list(alt_route)
            assert (len(route) == len(np.unique(route)))
            current_cost = new_cost
    return route, current_cost


def simulated_annealing(route, tmax, tmin, cooling_rate):
    t = tmax
    current_cost = calculate_cost(route)
    while t >= tmin:
        iteration = 0
        while iteration < 10:
            # Generate new solution
            swap_id = random.randint(0, len(route) - 2)
            # Generate Neighbour Solution
            alt_route = list(route)
            alt_route[swap_id], alt_route[swap_id + 1] = alt_route[swap_id + 1], alt_route[swap_id]
            assert (len(alt_route) == len(np.unique(alt_route)))
            assert (len(route) == len(alt_route))
            new_cost = calculate_cost(alt_route)
            # Cost Difference
            dE = new_cost - current_cost
            # Probability of accepting worse results
            probability = np.exp(-dE / t)
            randomity = random.random()
            # if new solution is better
            if dE < 0:
                logging.debug(f"Iteration {iteration}, Cost: {new_cost}, route change: {alt_route}")
                route = alt_route
                current_cost = new_cost
            elif randomity < probability:
                logging.debug(f"Iteration {iteration}, Cost: {new_cost}, route change: {alt_route}")
                route = alt_route
                current_cost = new_cost
            iteration += 1
        t *= cooling_rate
        cost_history.append(current_cost)
        logging.info(f"T: {t} cost: {current_cost}")
    return route, current_cost



# Generate Initial Condition
#route = random_init()
route = greedy_init()
cost = calculate_cost(route)
cost_history.append(cost)
iteration = 0
print(f"Initial Cost: {cost}, Route {route}")

shc_route, shc_cost = steepest_hill_climb(route)
print(f"Steepest Hill Climb Cost: {shc_cost}, Route {shc_route}")

route, cost = simulated_annealing(route, tmax=10.0, tmin=0.0005, cooling_rate=0.995)
plt.plot(cost_history)
print(f"Final Cost: {cost}, Final Route {route}")

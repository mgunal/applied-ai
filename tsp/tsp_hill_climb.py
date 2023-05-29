import numpy as np
import random

# Load Cost Matrix
cost_matrix = np.loadtxt('tsp_matrix.txt', delimiter=',')
random.seed(42)


def calculate_cost(path):
    cost = 0
    for i in range(0, len(path) - 1):
        start = path[i]
        end = path[i + 1]
        cost += cost_matrix[start, end]
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
    # Add starting point as final destination
    return init_route


# Generate Initial Condition
route = random_init()
#route = greedy_init()
current_cost = calculate_cost(route)
iteration = 0


def simple_hill_climb(route):
    current_cost = calculate_cost(route)
    swap_id = random.randint(0, len(route) - 2)
    #print(f"Swapping: {swap_id}, {swap_id+1}")
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
    return route, new_cost

print(f"Initial Cost: {current_cost}, Route {route}")
for i in range(0, 1000):
    if iteration < 250:
        #new_route, new_cost = simple_hill_climb(route)
        new_route, new_cost = steepest_hill_climb(route)
        if new_cost < current_cost:
            print(f"Iteration {i}, Cost: {new_cost}, route change: {new_route}")
            route = list(new_route)
            current_cost = new_cost
            iteration = 0
        else:
            # Number of iterations without any improvement
            iteration = iteration + 1
    else:
        print("No improvement for 250 iterations")
        break

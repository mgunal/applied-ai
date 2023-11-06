import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor

import models as models

from dataset import X_train, Y_train, X_validation, Y_validation

# Define the size of the population and number of generations for the GA
population_size = 10
num_generations = 5
num_features = X_train.shape[1]

# Initialize a population with random feature selections
np.random.seed(1)  # for reproducibility
population = np.random.randint(2, size=(population_size, num_features))


# Evaluate the population: train a model on the selected features and calculate the fitness (R^2 score)
def evaluate_population(population, X_train, y_train):
    scores = []
    for individual in population:
        selected_features = X_train.columns[individual.astype(bool)]
        if len(selected_features) == 0:
            scores.append(-np.inf)  # penalty for selecting no features
        else:
            X_train_subset = X_train[selected_features]
            X_validation_subset = X_validation[selected_features]
            model = LinearRegression()
            #model = MLPRegressor(hidden_layer_sizes=(20, 40, 40, 20), activation='relu', solver='adam', random_state=1, learning_rate='adaptive')
            model.fit(X_train_subset, Y_train)
            prediction = model.predict(X_validation_subset)
            mse = mean_squared_error(Y_validation, prediction)
            scores.append(mse)
    return scores


# Run the GA
best_individual = None
best_score = np.inf

for generation in range(num_generations):
    # Evaluate the current population
    scores = evaluate_population(population, X_train, Y_train)
    best_idx = np.argmin(scores)
    if scores[best_idx] < best_score:
        best_individual = population[best_idx]
        best_score = scores[best_idx]

    # Select the top performers to become parents of the next generation
    sorted_idx = np.argsort(scores)
    top_performers = population[sorted_idx[:population_size // 2]]

    # Create the next generation through crossover and mutation
    next_generation = []
    while len(next_generation) < population_size:
        parent1, parent2 = top_performers[np.random.choice(len(top_performers), 2, replace=False)]
        # Crossover
        crossover_point = np.random.randint(num_features)
        child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        # Mutation
        mutation_idx = np.random.randint(num_features)
        child[mutation_idx] = 1 - child[mutation_idx]  # flip the bit
        next_generation.append(child)

    population = np.array(next_generation)

# The best individual and their corresponding R^2 score
selected_features_ga = X_train.columns[best_individual.astype(bool)].tolist(), best_score
print(selected_features_ga)

if __name__ == '__main__':
    # Run models with the selected features
    for model_name, model in models.models:
        print(f"{model_name} Model")
        X_train_selected = X_train[selected_features_ga[0]]
        X_validate_selected = X_validation[selected_features_ga[0]]
        model.fit(X_train_selected, Y_train)
        prediction = model.predict(X_validate_selected)
        mse = mean_squared_error(Y_validation, prediction)
        mae = mean_absolute_error(Y_validation, prediction)
        r2 = r2_score(Y_validation, prediction)
        print("Mean Squared Error:", mse)
        print("Mean Absolute Error:", mae)
        print("R2 Score:", r2)
        print()



import random

class Item:
    def __init__(self, weight, value):
        self.weight = weight
        self.value = value

def initialize_population(num_solutions, num_items):
    # Initialize a population of solutions with random binary values
    population = []
    for _ in range(num_solutions):
        solution = []
        for _ in range(num_items):
            solution.append(random.randint(0, 1))
        population.append(solution)
    return population

def evaluate_solution(solution, items, max_weights):
    # Evaluate the total value of a solution and check if it exceeds the maximum weights
    total_value = [0, 0]  # Initialize the total value for each knapsack
    total_weights = [0, 0]  # Initialize the total weights for each knapsack
    for i in range(len(solution)):
        if solution[i] == 1:
            # Add the value and weight of the item to the respective knapsack
            total_value[items[i].knapsack] += items[i].value
            total_weights[items[i].knapsack] += items[i].weight
    # Check if the total weights exceed the maximum weights for any knapsack
    if any(weight > max_weight for weight in total_weights):
        total_value = [0, 0]  # Set total value to 0 for invalid solutions
    return total_value

def local_search(solution, items, max_weights):
    # Perform a local search by flipping each bit in the solution and checking for improvement
    improved_solution = solution[:]
    for i in range(len(solution)):
        improved_solution[i] = 1 - solution[i]
        if evaluate_solution(improved_solution, items, max_weights) > evaluate_solution(solution, items, max_weights):
            solution = improved_solution[:]
        else:
            improved_solution[i] = solution[i]
    return solution

def roulette_wheel_selection(population, fitness_values):
    # Calculate the total fitness by summing all the values in fitness_values
    total_fitness = sum(sum(fitness) for fitness in fitness_values)

    # Calculate the probabilities of selection for each solution
    probabilities = [sum(fitness) / total_fitness for fitness in fitness_values]

    # Calculate the cumulative probabilities for selection
    cumulative_probabilities = [sum(probabilities[:i+1]) for i in range(len(probabilities))]

    # Generate a random value and select a solution based on the cumulative probabilities
    random_value = random.random()
    for i in range(len(cumulative_probabilities)):
        if random_value <= cumulative_probabilities[i]:
            return population[i]

    # If no solution is selected, return the last solution in the population
    return population[-1]

def bee_colony(num_solutions, num_items, max_weights, num_iterations):
    items = [Item(weight, value, knapsack) for weight, value, knapsack in zip(weights, values, knapsacks)]
    population = initialize_population(num_solutions, num_items)
    best_solution = None
    for _ in range(num_iterations):
        fitness_values = [evaluate_solution(solution, items, max_weights) for solution in population]
        for i in range(num_solutions):
            solution = population[i]
            neighbor = local_search(solution, items, max_weights)
            if evaluate_solution(neighbor, items, max_weights) > evaluate_solution(solution, items, max_weights):
                population[i] = neighbor
            if best_solution is None or evaluate_solution(population[i], items, max_weights) > evaluate_solution(best_solution, items, max_weights):
                best_solution = population[i][:]
        roulette_selection = roulette_wheel_selection(population, fitness_values)
        for i in range(num_solutions):
            if population[i] != roulette_selection:
                population[i] = roulette_selection[:]
    return best_solution

# Example usage
weights = [2, 3, 5, 7, 9]
values = [4, 6, 10, 12, 15]
knapsacks = [0, 1, 1, 0, 1]
max_weights = [15, 10]  # Maximum weights for each knapsack
num_solutions = 10
num_items = len(weights)
num_iterations = 100

best_solution = bee_colony(num_solutions, num_items, max_weights, num_iterations)
total_values = evaluate_solution(best_solution, items, max_weights)

print("Best solution:", best_solution)
print("Total values:", total_values)


import random

class Item:
    def __init__(self, weight, value):
        self.weight = weight
        self.value = value

def initialize_population(num_solutions, num_items):
    population = []
    for _ in range(num_solutions):
        solution = []
        for _ in range(num_items):
            solution.append([random.randint(0, 1), random.randint(0, 1)])
        population.append(solution)
    return population

def evaluate_solution(solution, items, max_weights):
    total_values = [0] * len(max_weights)
    total_weights = [0] * len(max_weights)
    for i in range(len(solution)):
        for j in range(len(max_weights)):
            if solution[i][j] == 1:
                total_values[j] += items[i].value
                total_weights[j] += items[i].weight
    for j in range(len(max_weights)):
        if total_weights[j] > max_weights[j]:
            total_values[j] = 0
    return total_values

def local_search(solution, items, max_weights):
    improved_solution = solution[:]
    for i in range(len(solution)):
        for j in range(len(max_weights)):
            improved_solution[i][j] = 1 - solution[i][j]
            if evaluate_solution(improved_solution, items, max_weights)[j] > evaluate_solution(solution, items, max_weights)[j]:
                solution = improved_solution[:]
            else:
                improved_solution[i][j] = solution[i][j]
    return solution

def roulette_wheel_selection(population, fitness_values):
    total_fitness = sum(sum(fitness) for fitness in fitness_values)
    probabilities = [sum(fitness) / total_fitness for fitness in fitness_values]
    cumulative_probabilities = [sum(probabilities[:i+1]) for i in range(len(probabilities))]
    random_value = random.random()
    for i in range(len(cumulative_probabilities)):
        if random_value <= cumulative_probabilities[i]:
            return population[i]
    return population[-1]

def bee_colony(num_solutions, num_items, max_weights, num_iterations, items):
    population = initialize_population(num_solutions, num_items)
    best_solution = None
    for _ in range(num_iterations):
        fitness_values = [evaluate_solution(solution, items, max_weights) for solution in population]
        for i in range(num_solutions):
            solution = population[i]
            neighbor = local_search(solution, items, max_weights)
            if evaluate_solution(neighbor, items, max_weights)[0] > evaluate_solution(solution, items, max_weights)[0]:
                population[i] = neighbor
            if best_solution is None or evaluate_solution(population[i], items, max_weights)[0] > evaluate_solution(best_solution, items, max_weights)[0]:
                best_solution = population[i][:]
        roulette_selection = roulette_wheel_selection(population, fitness_values)
        for i in range(num_solutions):
            if population[i] != roulette_selection:
                population[i] = roulette_selection[:]
    return best_solution

# Example usage
weights = [2, 3, 5, 7, 9]
values = [4, 6, 10, 12, 15]
max_weights = [15, 10]
num_solutions = 10
num_items = len(weights)
num_iterations = 100
items = [Item(weight, value) for weight, value in zip(weights, values)]

best_solution = bee_colony(num_solutions, num_items, max_weights, num_iterations, items)
total_values = evaluate_solution(best_solution, items, max_weights)

print("Best solution:", best_solution)
print("Total values:", total_values)

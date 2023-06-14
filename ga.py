import numpy as np
import random

# Problem data
jobs = [i for i in range(10)]  # 10 jobs
machines = [i for i in range(3)]  # 3 machines
time_matrix = np.random.randint(1, 10, size=(len(jobs), len(machines)))  # Processing times

# Parameters
population_size = 50
mutation_rate = 0.1
generations = 1000

# Initialize population
population = [random.sample(jobs, len(jobs)) for _ in range(population_size)]


def fitness(individual):
    # Calculate the total processing time of the jobs in the given order
    total_time = np.zeros(len(machines))
    for job in individual:
        job_time = time_matrix[job]
        total_time[0] += job_time[0]
        for m in range(1, len(machines)):
            total_time[m] = max(total_time[m], total_time[m-1]) + job_time[m]
    return total_time[-1]  # Return time when last machine finishes


def crossover(parent1, parent2):
    # Order crossover
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    offspring = [None]*size
    offspring[start:end] = parent1[start:end]
    for gene in parent2:
        if gene not in offspring:
            for i in range(size):
                if offspring[i] is None:
                    offspring[i] = gene
                    break
    return offspring


def mutate(individual):
    # Swap mutation
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual


# Main loop
for gen in range(generations):
    # Evaluate fitness
    scores = [fitness(ind) for ind in population]
    # Select parents (roulette wheel selection)
    # Select parents (roulette wheel selection)
    total_fit = sum(scores)
    selection_probs = np.array(scores) / total_fit
    parent_indices = np.random.choice(range(len(population)), size=population_size, p=selection_probs)
    parents = [population[i] for i in parent_indices]
    # Create next generation
    next_population = []
    for i in range(0, population_size, 2):
        offspring1 = crossover(parents[i], parents[i+1])
        offspring2 = crossover(parents[i+1], parents[i])
        next_population.append(mutate(offspring1))
        next_population.append(mutate(offspring2))
    population = next_population

# Return best solution
best_solution = min(population, key=fitness)
print('Best solution: ', best_solution)

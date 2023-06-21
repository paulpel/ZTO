import numpy as np
import random


def fitness(individual, machines, time_matrix):
    total_time = np.zeros(len(machines))
    for job in individual:
        job_time = time_matrix[job]
        total_time[0] += job_time[0]
        for m in range(1, len(machines)):
            total_time[m] = max(total_time[m], total_time[m - 1]) + job_time[m]
    return total_time[-1]


def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    offspring = [None] * size
    offspring[start:end] = parent1[start:end]
    for gene in parent2:
        if gene not in offspring:
            for i in range(size):
                if offspring[i] is None:
                    offspring[i] = gene
                    break
    return offspring


def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual


if __name__ == "__main__":
    jobs = [i for i in range(10)]
    machines = [i for i in range(3)]
    time_matrix = np.random.randint(1, 10, size=(len(jobs), len(machines)))

    population_size = 50
    mutation_rate = 0.1
    generations = 1000
    elitism_size = 5  # Top 5 individuals pass directly to the next generation

    population = [random.sample(jobs, len(jobs)) for _ in range(population_size)]

    for gen in range(generations):
        scores = [fitness(ind, machines, time_matrix) for ind in population]

        sorted_population = [x for _, x in sorted(zip(scores, population))]

        next_population = sorted_population[
            :elitism_size
        ]  # Add elite individuals to next generation

        scores = scores[elitism_size:]  # Remove elite individuals from scores
        sorted_population = sorted_population[
            elitism_size:
        ]  # Remove elite individuals from population

        inv_scores = 1.0 / np.array(scores)
        total_inv_fit = sum(inv_scores)
        selection_probs = inv_scores / total_inv_fit

        parent_indices = np.random.choice(
            range(len(sorted_population)),
            size=(population_size - elitism_size) // 2 * 2,
            p=selection_probs,
        )

        parents = [sorted_population[i] for i in parent_indices]

        for i in range(0, len(parents) - 1, 2):  # Modify loop range due to elitism
            offspring1 = crossover(parents[i], parents[i + 1])
            offspring2 = crossover(parents[i + 1], parents[i])
            next_population.append(mutate(offspring1, mutation_rate))
            next_population.append(mutate(offspring2, mutation_rate))
        population = next_population

    best_solution = min(population, key=lambda x: fitness(x, machines, time_matrix))

    print("Best solution: ", best_solution)

import numpy as np
import random as rd
import math
import matplotlib.pyplot as plt
from itertools import combinations

def makespan(individual, machines, time_matrix):
    total_time = np.zeros(len(machines))
    for job in individual:
        job_time = time_matrix[job]
        total_time[0] += job_time[0]
        for m in range(1, len(machines)):
            total_time[m] = max(total_time[m], total_time[m - 1]) + job_time[m]
    return total_time[-1]


def total_flowtime(individual, machines, time_matrix):
    total_time = np.zeros(len(machines))
    flow_time = 0
    for job in individual:
        job_time = time_matrix[job]
        total_time[0] += job_time[0]
        for m in range(1, len(machines)):
            total_time[m] = max(total_time[m], total_time[m - 1]) + job_time[m]
        flow_time += total_time[-1]
    return flow_time


def max_tardiness(individual, machines, time_matrix, due_dates):
    total_time = np.zeros(len(machines))
    max_tard = 0
    for job in individual:
        job_time = time_matrix[job]
        total_time[0] += job_time[0]
        for m in range(1, len(machines)):
            total_time[m] = max(total_time[m], total_time[m - 1]) + job_time[m]
        tard = max(0, total_time[-1] - due_dates[job])
        max_tard = max(max_tard, tard)
    return max_tard


def total_tardiness(individual, machines, time_matrix, due_dates):
    total_time = np.zeros(len(machines))
    total_tard = 0
    for job in individual:
        job_time = time_matrix[job]
        total_time[0] += job_time[0]
        for m in range(1, len(machines)):
            total_time[m] = max(total_time[m], total_time[m - 1]) + job_time[m]
        tard = max(0, total_time[-1] - due_dates[job])
        total_tard += tard
    return total_tard


def max_lateness(individual, machines, time_matrix, due_dates):
    total_time = np.zeros(len(machines))
    max_late = 0
    for job in individual:
        job_time = time_matrix[job]
        total_time[0] += job_time[0]
        for m in range(1, len(machines)):
            total_time[m] = max(total_time[m], total_time[m - 1]) + job_time[m]
        late = total_time[-1] - due_dates[job]
        max_late = max(max_late, late)
    return max_late


def total_lateness(individual, machines, time_matrix, due_dates):
    total_time = np.zeros(len(machines))
    total_late = 0
    for job in individual:
        job_time = time_matrix[job]
        total_time[0] += job_time[0]
        for m in range(1, len(machines)):
            total_time[m] = max(total_time[m], total_time[m - 1]) + job_time[m]
        late = total_time[-1] - due_dates[job]
        total_late += late
    return total_late


def simulated_annealing(jobs, machines, time_matrix, due_dates, objective_functions, T=10000, alpha=0.995, T_min=0.01):
    current_schedule = jobs.copy()
    rd.shuffle(current_schedule)
    scores = []

    while T > T_min:
        new_schedule = current_schedule.copy()
        i, j = rd.sample(range(len(new_schedule)), 2)
        new_schedule[i], new_schedule[j] = new_schedule[j], new_schedule[i]

        current_score = [f(current_schedule, machines, time_matrix) if f.__name__ in ['makespan', 'total_flowtime'] 
                         else f(current_schedule, machines, time_matrix, due_dates) for f in objective_functions]
        
        new_score = [f(new_schedule, machines, time_matrix) if f.__name__ in ['makespan', 'total_flowtime'] 
                     else f(new_schedule, machines, time_matrix, due_dates) for f in objective_functions]

        if math.exp((sum(current_score) - sum(new_score)) / T) > rd.random():
            current_schedule = new_schedule
            current_score = new_score
        
        scores.append(current_score)
        T *= alpha

    return current_schedule, scores


def pareto_dominance(objectives):
    """
    Given a list of objectives, this function finds the Pareto optimal ones.
    """
    num_objectives = len(objectives[0])
    pareto_optimal = np.ones(len(objectives), dtype=bool)
    for i, objective in enumerate(objectives):
        for j, candidate in enumerate(objectives):
            if all(candidate[k] <= objective[k] for k in range(num_objectives)) and \
                any(candidate[k] < objective[k] for k in range(num_objectives)):
                pareto_optimal[i] = 0
                break
    return np.array(objectives)[pareto_optimal]


def plot_pareto(scores, objective_functions):
    plt.figure(figsize=(10, 7))

    # Filter out the Pareto optimal points
    pareto_scores = pareto_dominance(scores)
    pareto_scores = pareto_scores[np.argsort(pareto_scores[:, 0])]

    # Plot all points
    scores = np.array(scores)
    plt.scatter(scores[:,0], scores[:,1], alpha=0.5)
    
    # Plot Pareto front
    plt.plot(pareto_scores[:,0], pareto_scores[:,1], 'r-')

    plt.xlabel(objective_functions[0].__name__)
    plt.ylabel(objective_functions[1].__name__)
    plt.title('Pareto Front')
    plt.show()


if __name__ == "__main__":
    # Przykładowe użycie:
    jobs = [i for i in range(10)]
    machines = [i for i in range(3)]
    time_matrix = np.random.randint(1, 10, size=(len(jobs), len(machines)))
    due_dates = np.random.randint(1, 100, size=len(jobs))
    all_functions = [makespan, total_flowtime, max_tardiness, total_tardiness, max_lateness, total_lateness]
    all_pairs = list(combinations(all_functions, 2))
    for pair in all_pairs:
        best_schedule, scores = simulated_annealing(jobs, machines, time_matrix, due_dates, pair)
        plot_pareto(scores, pair)
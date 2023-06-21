import random
import numpy as np

n = 10
# dane problemu
c = np.array([random.randint(1, 10) for _ in range(n)])  # wartości przedmiotów
w = np.array([random.randint(1, 10) for _ in range(n)])  # wagi dla plecaka pierwszego
v = np.array([random.randint(1, 10) for _ in range(n)])  # wagi dla plecaka drugiego
B = 10  # pojemność plecaków

n_particles = 100
n_iterations = 1000
n_dimensions = 2 * len(c)  # dla xi i yi każdego przedmiotu

# inicjalizacja cząstek
particles = np.random.rand(n_particles, n_dimensions)  # pozycje
velocities = np.random.rand(n_particles, n_dimensions)  # prędkości
personal_best_positions = particles.copy()
personal_best_values = np.full((n_particles,), float('-inf'))

# inicjalizacja globalnego optimum
global_best_position = particles[0].copy()
global_best_value = float('-inf')

for iteration in range(n_iterations):
    # obliczenie wartości funkcji celu dla każdej cząstki
    for i in range(n_particles):
        x = particles[i, :len(c)]
        y = particles[i, len(c):]
        value = np.sum((x + y) * c)

        # sprawdzenie, czy cząstka jest w granicach plecaków i spełnia ograniczenia
        if np.sum(x * w) <= B and np.sum(y * v) <= B and np.all(x + y <= 1):
            if value > personal_best_values[i]:
                personal_best_values[i] = value
                personal_best_positions[i] = particles[i]

            if value > global_best_value:
                global_best_value = value
                global_best_position = particles[i]

    # aktualizacja prędkości i pozycji
    for i in range(n_particles):
        inertia = velocities[i]
        cognitive = 0.5 * random.random() * (personal_best_positions[i] - particles[i])
        social = 0.5 * random.random() * (global_best_position - particles[i])
        velocities[i] = inertia + cognitive + social
        particles[i] += velocities[i]
        
        # Tutaj sprawdzamy, czy po aktualizacji nasze pozycje są w granicach [0,1]. 
        # Jeśli nie, to ustawiamy je na najbliższy dopuszczalny kraniec.
        particles[i] = np.clip(particles[i], 0, 1)
        
        # Tutaj sprawdzamy, czy suma x[i] i y[i] nie przekracza 1. 
        # Jeśli tak, normalizujemy je, dzieląc przez ich sumę, co gwarantuje, że ich suma wynosi 1.
        for j in range(len(c)):
            if particles[i,j] + particles[i,j+len(c)] > 1:
                total = particles[i,j] + particles[i,j+len(c)]
                particles[i,j] /= total
                particles[i,j+len(c)] /= total

# wydrukowanie wyników
x_opt = global_best_position[:len(c)]
y_opt = global_best_position[len(c):]
print(f"Optimum: x = {x_opt}, y = {y_opt}, value = {global_best_value}")

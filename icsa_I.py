import numpy as np

def fitness_function(solution, expectation):
    return np.sum(solution * expectation)

def icsa_optimization(expectation, assignment_matrix, max_iters=100, population_size=30):
    """
    ICSA-I Optimization: Evolve a population of solutions to minimize the energy consumption.
    """
    population = np.random.rand(population_size, assignment_matrix.shape[1])
    
    for _ in range(max_iters):
        # Evaluate fitness
        fitness_scores = np.array([fitness_function(ind, expectation) for ind in population])
        
        # Sort population based on fitness
        sorted_indices = np.argsort(fitness_scores)
        population = population[sorted_indices]
        
        # Select best chickens (20% of population)
        best_chickens = population[:int(0.2 * population_size)]
        
        # Evolve the rest of the population
        for i in range(int(0.2 * population_size), population_size):
            # Randomly select two best chickens to influence the current chicken
            idx1, idx2 = np.random.choice(best_chickens.shape[0], 2, replace=False)
            r1, r2 = np.random.random(2)
            population[i] = population[i] + r1 * (best_chickens[idx1] - population[i]) + r2 * (best_chickens[idx2] - population[i])
        
        # Apply constraints: Ensure solutions are within bounds
        population = np.clip(population, 0, 1)
    
    best_solution = population[0]
    best_fitness = fitness_function(best_solution, expectation)
    
    return best_solution, best_fitness

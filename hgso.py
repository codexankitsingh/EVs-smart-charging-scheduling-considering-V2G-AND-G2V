def hgso_optimization(initial_solution, expectation, max_iters=50):
    """
    Henry Gas Solubility Optimization: Fine-tune the solution to further minimize energy consumption.
    """
    best_solution = initial_solution
    best_fitness = fitness_function(initial_solution, expectation)
    
    for _ in range(max_iters):
        # Generate a new solution by perturbing the current best solution
        perturbation = np.random.normal(0, 0.1, best_solution.shape)
        new_solution = best_solution + perturbation
        
        # Apply constraints
        new_solution = np.clip(new_solution, 0, 1)
        
        new_fitness = fitness_function(new_solution, expectation)
        
        # Update best solution if new fitness is better
        if new_fitness < best_fitness:
            best_solution = new_solution
            best_fitness = new_fitness
            
    return best_solution, best_fitness

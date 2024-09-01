mean_values = [55, 128, 12]  # Example mean values for distance, arrival time, and departure time
std_dev_values = [10, 10, 2]  # Example standard deviations

# Assignment matrix based on the paper
assignment_matrix = np.array([
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
])

final_solution, final_fitness = integrate_modules(mean_values, std_dev_values, assignment_matrix)
print(f'Optimal solution: {final_solution}')
print(f'Minimized cost: {final_fitness}')

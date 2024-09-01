import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.optimize import differential_evolution

# 1. Data Preprocessing Module
def load_data():
    """Simulate EV and CS data."""
    # EV Data: [ev_id, location (x, y), t_arr, t_dep, soc]
    ev_data = pd.DataFrame({
        'ev_id': range(1, 16),
        'location_x': np.random.randint(0, 20, size=15),
        'location_y': np.random.randint(0, 20, size=15),
        'arrival_time': np.random.randint(0, 12, size=15),
        'departure_time': np.random.randint(12, 24, size=15),
        'soc': np.random.uniform(30, 80, size=15)
    })
    
    # CS Data: [cs_id, location (x, y), capacity]
    cs_data = pd.DataFrame({
        'cs_id': [1, 2, 3],
        'location_x': [5, 15, 10],
        'location_y': [10, 20, 5],
        'capacity': [5, 4, 3]  # Number of charging plugs
    })
    
    return ev_data, cs_data

def preprocess_data(ev_data, cs_data):
    """Normalize and clean the data for further processing."""
    ev_data = ev_data.dropna()
    cs_data = cs_data.dropna()
    return ev_data, cs_data

# 2. Handling Uncertainties with Hong's 2m-PEM
def pem_method(data, order=2):
    """Apply Hong's 2m-PEM method to handle uncertainties."""
    mean = np.mean(data)
    std_dev = np.std(data)
    pem_value = mean + order * std_dev
    return pem_value

def handle_uncertainties(ev_data):
    """Apply 2m-PEM to uncertain variables in the EV dataset."""
    uncertainty_range = np.random.uniform(8, 35, size=len(ev_data))
    ev_data['uncertain_soc'] = np.clip(ev_data['soc'] + (ev_data['soc'] * uncertainty_range / 100), 0, 100)
    ev_data['uncertain_arrival'] = pem_method(ev_data['arrival_time'])
    ev_data['uncertain_departure'] = pem_method(ev_data['departure_time'])
    return ev_data

# 3. ILP for Initial Assignment
def ilp_assignment(ev_data, cs_data):
    """Solve the assignment problem using ILP."""
    num_evs = len(ev_data)
    num_css = len(cs_data)
    
    # Cost matrix: energy consumption or distance
    cost_matrix = np.zeros((num_evs, num_css))
    
    for i, ev in ev_data.iterrows():
        for j, cs in cs_data.iterrows():
            distance = calculate_distance((ev['location_x'], ev['location_y']),
                                          (cs['location_x'], cs['location_y']))
            energy_consumption = calculate_energy_consumption(distance)
            cost_matrix[i, j] = energy_consumption
    
    # Bounds and constraints for ILP
    c = cost_matrix.flatten()
    A_eq = np.zeros((num_evs, num_evs * num_css)) # equity consrtaints assuring each ev is assign to only one charging station
    b_eq = np.ones(num_evs)
    
    for i in range(num_evs):
        A_eq[i, i*num_css:(i+1)*num_css] = 1
    
    bounds = [(0, 1) for _ in range(num_evs * num_css)]
    
    # Linear programming to minimize cost
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    assignments = np.round(result.x).reshape((num_evs, num_css))
    return assignments

def calculate_distance(loc1, loc2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

def calculate_energy_consumption(distance):
    """Calculate energy consumption based on distance."""
    return distance * 0.2  # Example factor for energy consumption

# 4. HGSO for Optimization and refine the optimal assignment of each ev to a specific charging station
def objective_function(params, ev_data, cs_data): # compute the total energy  consumption for a given set of assignments
    """Objective function for HGSO to optimize assignment."""
    total_energy = 0
    for i, ev in ev_data.iterrows():
        assigned_cs = int(params[i])
        distance = calculate_distance((ev['location_x'], ev['location_y']),
                                      (cs_data.loc[assigned_cs, 'location_x'],
                                       cs_data.loc[assigned_cs, 'location_y']))
        energy_consumption = calculate_energy_consumption(distance)
        total_energy += energy_consumption
    return total_energy

def hgso_optimization(assignments, ev_data, cs_data):
    """Use HGSO to optimize the initial ILP assignment."""
    bounds = [(0, len(cs_data)-1) for _ in range(len(ev_data))]
    result = differential_evolution(objective_function, bounds, 
                                    args=(ev_data, cs_data), strategy='best1bin',
                                    maxiter=1000, popsize=25, tol=1e-6)
    optimized_assignments = np.round(result.x).astype(int)
    return optimized_assignments  # further reduces energy consumption

# 5. Final Assignment Process and Output
def assign_ev_to_cs(ev_data, cs_data):
    """Complete the assignment process with ILP and HGSO."""
    # Handle uncertainties in EV data
    ev_data = handle_uncertainties(ev_data)
    
    # Select only 12 EVs for assignment (since we have 12 plugs)
    ev_data = ev_data.head(12)
    
    # ILP for initial assignment
    initial_assignments = ilp_assignment(ev_data, cs_data)
    
    # HGSO for optimization
    optimized_assignments = hgso_optimization(initial_assignments, ev_data, cs_data)
    
    # Output assignments
    for i, ev in ev_data.iterrows():
        print(f"EV {ev['ev_id']} assigned to CS: {optimized_assignments[i]} with uncertain SOC: {ev['uncertain_soc']:.2f}%")

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.optimize import differential_evolution

# 1. Data Preprocessing Module
def load_data():
    """Simulate EV and CS data."""
    # EV Data: [ev_id, location (x, y), t_arr, t_dep, soc]
    ev_data = pd.DataFrame({
        'ev_id': range(1, 16),
        'location_x': np.random.randint(0, 20, size=15),
        'location_y': np.random.randint(0, 20, size=15),
        'arrival_time': np.random.randint(0, 12, size=15),
        'departure_time': np.random.randint(12, 24, size=15),
        'soc': np.random.uniform(30, 80, size=15)
    })
    
    # CS Data: [cs_id, location (x, y), capacity]
    cs_data = pd.DataFrame({
        'cs_id': [1, 2, 3],
        'location_x': [5, 15, 10],
        'location_y': [10, 20, 5],
        'capacity': [5, 4, 3]  # Number of charging plugs
    })
    
    return ev_data, cs_data

def preprocess_data(ev_data, cs_data):
    """Normalize and clean the data for further processing."""
    ev_data = ev_data.dropna()
    cs_data = cs_data.dropna()
    return ev_data, cs_data

# 2. Handling Uncertainties with Hong's 2m-PEM
def pem_method(data, order=2):
    """Apply Hong's 2m-PEM method to handle uncertainties."""
    mean = np.mean(data)
    std_dev = np.std(data)
    pem_value = mean + order * std_dev
    return pem_value

def handle_uncertainties(ev_data):
    """Apply 2m-PEM to uncertain variables in the EV dataset."""
    uncertainty_range = np.random.uniform(8, 35, size=len(ev_data))
    ev_data['uncertain_soc'] = np.clip(ev_data['soc'] + (ev_data['soc'] * uncertainty_range / 100), 0, 100)
    ev_data['uncertain_arrival'] = pem_method(ev_data['arrival_time'])
    ev_data['uncertain_departure'] = pem_method(ev_data['departure_time'])
    return ev_data

# 3. ILP for Initial Assignment
def ilp_assignment(ev_data, cs_data):
    """Solve the assignment problem using ILP."""
    num_evs = len(ev_data)
    num_css = len(cs_data)
    
    # Cost matrix: energy consumption or distance
    cost_matrix = np.zeros((num_evs, num_css))
    
    for i, ev in ev_data.iterrows():
        for j, cs in cs_data.iterrows():
            distance = calculate_distance((ev['location_x'], ev['location_y']),
                                          (cs['location_x'], cs['location_y']))
            energy_consumption = calculate_energy_consumption(distance)
            cost_matrix[i, j] = energy_consumption
    
    # Bounds and constraints for ILP
    c = cost_matrix.flatten()
    A_eq = np.zeros((num_evs, num_evs * num_css))
    b_eq = np.ones(num_evs)
    
    for i in range(num_evs):
        A_eq[i, i*num_css:(i+1)*num_css] = 1
    
    bounds = [(0, 1) for _ in range(num_evs * num_css)]
    
    # Linear programming to minimize cost
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    assignments = np.round(result.x).reshape((num_evs, num_css))
    return assignments

def calculate_distance(loc1, loc2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

def calculate_energy_consumption(distance):
    """Calculate energy consumption based on distance."""
    return distance * 0.2  # Example factor for energy consumption

# 4. HGSO for Optimization
def objective_function(params, ev_data, cs_data):
    """Objective function for HGSO to optimize assignment."""
    total_energy = 0
    for i, ev in ev_data.iterrows():
        assigned_cs = int(params[i])
        distance = calculate_distance((ev['location_x'], ev['location_y']),
                                      (cs_data.loc[assigned_cs, 'location_x'],
                                       cs_data.loc[assigned_cs, 'location_y']))
        energy_consumption = calculate_energy_consumption(distance)
        total_energy += energy_consumption
    return total_energy

def hgso_optimization(assignments, ev_data, cs_data):
    """Use HGSO to optimize the initial ILP assignment."""
    bounds = [(0, len(cs_data)-1) for _ in range(len(ev_data))]
    result = differential_evolution(objective_function, bounds, 
                                    args=(ev_data, cs_data), strategy='best1bin',
                                    maxiter=1000, popsize=25, tol=1e-6)
    optimized_assignments = np.round(result.x).astype(int)
    return optimized_assignments

# 5. Final Assignment Process and Output
def assign_ev_to_cs(ev_data, cs_data):
    """Complete the assignment process with ILP and HGSO."""
    # Handle uncertainties in EV data
    ev_data = handle_uncertainties(ev_data)
    
    # Select only 12 EVs for assignment (since we have 12 plugs)
    ev_data = ev_data.head(12)
    
    # ILP for initial assignment
    initial_assignments = ilp_assignment(ev_data, cs_data)
    
    # HGSO for optimization
    optimized_assignments = hgso_optimization(initial_assignments, ev_data, cs_data)
    
    # Output assignments
    for i, ev in ev_data.iterrows():
        print(f"EV {ev['ev_id']} assigned to CS {optimized_assignments[i]} with uncertain SOC: {ev['uncertain_soc']:.2f}%")

# MAIN FUNCTION CALL
if __name__ == "__main__":
    # Load and preprocess data
    ev_data, cs_data = load_data()
    ev_data, cs_data = preprocess_data(ev_data, cs_data)
    
    # Assign EVs to Charging Stations
    assign_ev_to_cs(ev_data, cs_data)
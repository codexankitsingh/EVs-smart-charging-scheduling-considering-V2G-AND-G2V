import numpy as np

def calculate_2m_pem(mean, std_dev):
    """
    Calculate deterministic points using 2m-PEM method.
    """
    m3 = np.mean([((x - mean) ** 3) for x in std_dev])
    m4 = np.mean([((x - mean) ** 4) for x in std_dev])

    # Lambda calculations
    lambda_z_3 = m3 / (std_dev ** 3)
    lambda_z_4 = m4 / (std_dev ** 4)

    # Deterministic points calculation
    z_pos_1 = mean + std_dev * np.sqrt(lambda_z_4 / (3 * lambda_z_3))
    z_pos_2 = mean - std_dev * np.sqrt(lambda_z_4 / (3 * lambda_z_3))
    
    return z_pos_1, z_pos_2

def process_input_data(mean_values, std_dev_values):
    """
    Process input data using 2m-PEM to generate deterministic points.
    """
    deterministic_points = []
    for mean, std_dev in zip(mean_values, std_dev_values):
        z_pos_1, z_pos_2 = calculate_2m_pem(mean, std_dev)
        deterministic_points.append((z_pos_1, z_pos_2))
    return deterministic_points

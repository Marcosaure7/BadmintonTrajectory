from scipy.optimize import differential_evolution, minimize
import numpy as np
import matplotlib.pyplot as plt
import time

dt_DE=0.01
dt_SLSQP=0.001
co_fraine=0.22


# Start timer for the entire program
program_start_time = time.time()

class BadmintonTrajectory:
    def __init__(self, vitesse_ini, angle, dt):
        self.pos_x_list = []
        self.pos_y_list = []
        self.affichage_des_coo(vitesse_ini, angle, dt)

    def affichage_des_coo(self, vitesse_ini, angle, dt):
        angle_rad = np.radians(angle)
        vitesse_x = np.cos(angle_rad) * vitesse_ini
        vitesse_y = np.sin(angle_rad) * vitesse_ini
        pos_x = -1.98
        pos_y = 1.55
        vitesse_x_prec = vitesse_x

        self.pos_x_list = [pos_x]
        self.pos_y_list = [pos_y]

        while pos_y >= 0:
            vitesse_x += acceleration_x(vitesse_x, vitesse_y, dt)
            vitesse_y += acceleration_y(vitesse_x_prec, vitesse_y, dt)
            vitesse_x_prec = vitesse_x

            pos_x += vitesse_x * dt
            pos_y += vitesse_y * dt

            self.pos_x_list.append(pos_x)
            self.pos_y_list.append(pos_y)

    def distance_to_points(self, points, weights=None):
        # Ignore points with weight 0
        if weights is None:
            weights = [1] * len(points)  # Default to weight 1 if none provided
        filtered_points = [(px, py, w) for (px, py), w in zip(points, weights) if w > 0]

        distances = []
        for px, py, weight in filtered_points:
            min_distance = min(
                np.sqrt((px - x)**2 + (py - y)**2)
                for x, y in zip(self.pos_x_list, self.pos_y_list)
            )
            distances.append(weight * min_distance)
        return sum(distances)

def acceleration_x(vitesse_x, vitesse_y, increment):
    return -co_fraine * vitesse_x * np.sqrt(vitesse_x**2 + vitesse_y**2) * increment

def acceleration_y(vitesse_x, vitesse_y, increment):
    return -(9.81 + co_fraine * vitesse_y * np.sqrt(vitesse_x**2 + vitesse_y**2)) * increment

def optimize_trajectory(points, weights=None):
    start_time = time.time()  # Start timer for optimization

    # Differential Evolution Optimization (coarse dt)
    def objective_de(params):
        vitesse_ini, angle = params
        trajet = BadmintonTrajectory(vitesse_ini, angle, dt=dt_DE)
        return trajet.distance_to_points(points, weights)

    bounds = [(7, 35), (15, 75)]
    result_de = differential_evolution(objective_de, bounds)
    vitesse_de, angle_de = result_de.x

    # Sequential Least Squares Programming Optimization (fine dt)
    def objective_slsqp(params):
        vitesse_ini, angle = params
        trajet = BadmintonTrajectory(vitesse_ini, angle, dt=dt_SLSQP)
        return trajet.distance_to_points(points, weights)

    initial_guess = [vitesse_de, angle_de]
    result_slsqp = minimize(objective_slsqp, initial_guess, bounds=bounds, method='SLSQP')
    vitesse_slsqp, angle_slsqp = result_slsqp.x

    end_time = time.time()  # End timer for optimization
    elapsed_time = end_time - start_time
    print(f"Optimization time: {elapsed_time:.2f} seconds")

    return vitesse_slsqp, angle_slsqp  # Final optimized speed and angle


# Example usage with weights for each target point
distance_atterissage=7
hauteur_au_filet=0.05*(distance_atterissage-0.5)**2+1.6
importance_hauteur_filet= 5/(distance_atterissage+0.5)-0.3
importance_distance_atterissage=distance_atterissage+0.5

points = [(0,hauteur_au_filet), (distance_atterissage,0)]
weights = [importance_hauteur_filet,importance_distance_atterissage]  # Second point will be ignored
vitesse_opt, angle_opt = optimize_trajectory(points, weights)
print(f"Optimized initial conditions: speed = {vitesse_opt:.2f} m/s, angle = {angle_opt:.2f}°")

# Plotting the trajectory with optimized conditions
trajet = BadmintonTrajectory(vitesse_opt, angle_opt, dt=0.001)

# Stop timer for the entire program
program_end_time = time.time()
total_program_time = program_end_time - program_start_time
print(f"Total program time: {total_program_time:.2f} seconds")

def plot_trajectory(trajet):
    plt.figure(figsize=(12, 6))
    plt.plot(trajet.pos_x_list, trajet.pos_y_list, color='blue')
    plt.title('Badminton Shuttlecock Trajectory')
    plt.xlabel('Position X (meters)')
    plt.ylabel('Position Y (meters)')
    plt.grid(True)
    plt.show()

plot_trajectory(trajet)

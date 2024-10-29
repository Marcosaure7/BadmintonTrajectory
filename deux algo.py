from scipy.optimize import differential_evolution, minimize
import numpy as np
import matplotlib.pyplot as plt
import time

class BadmintonTrajectory:
    def __init__(self, vitesse_ini, angle, dt):
        self.pos_x_list = []
        self.pos_y_list = []
        self.affichage_des_coo(vitesse_ini, angle, dt)

    def affichage_des_coo(self, vitesse_ini, angle, dt):
        angle_rad = np.radians(angle)
        vitesse_x = np.cos(angle_rad) * vitesse_ini
        vitesse_y = np.sin(angle_rad) * vitesse_ini
        pos_x = 0
        pos_y = 1
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

    def distance_to_points(self, points):
        distances = []
        for px, py in points:
            min_distance = min(
                np.sqrt((px - x)**2 + (py - y)**2)
                for x, y in zip(self.pos_x_list, self.pos_y_list)
            )
            distances.append(min_distance)
        return sum(distances)

def acceleration_x(vitesse_x, vitesse_y, increment):
    return -0.22 * vitesse_x * np.sqrt(vitesse_x**2 + vitesse_y**2) * increment

def acceleration_y(vitesse_x, vitesse_y, increment):
    return -(9.81 + 0.22 * vitesse_y * np.sqrt(vitesse_x**2 + vitesse_y**2)) * increment

def optimize_trajectory(points):
    start_time = time.time()  # Start timer

    # Differential Evolution Optimization (coarse dt)
    def objective_de(params):
        vitesse_ini, angle = params
        trajet = BadmintonTrajectory(vitesse_ini, angle, dt=0.01)
        return trajet.distance_to_points(points)

    bounds = [(7, 35), (15, 45)]
    result_de = differential_evolution(objective_de, bounds)
    vitesse_de, angle_de = result_de.x

    # Sequential Least Squares Programming Optimization (fine dt)
    def objective_slsqp(params):
        vitesse_ini, angle = params
        trajet = BadmintonTrajectory(vitesse_ini, angle, dt=0.001)
        return trajet.distance_to_points(points)

    initial_guess = [vitesse_de, angle_de]
    result_slsqp = minimize(objective_slsqp, initial_guess, bounds=bounds, method='SLSQP')
    vitesse_slsqp, angle_slsqp = result_slsqp.x

    end_time = time.time()  # End timer
    elapsed_time = end_time - start_time
    print(f"Optimization time: {elapsed_time:.2f} seconds")

    return vitesse_slsqp, angle_slsqp  # Final optimized speed and angle

# Example usage
points = [(2, 1.6), (4, 0)]
vitesse_opt, angle_opt = optimize_trajectory(points)
print(f"Optimized initial conditions: speed = {vitesse_opt:.2f} m/s, angle = {angle_opt:.2f}°")

# Plotting the trajectory with optimized conditions
trajet = BadmintonTrajectory(vitesse_opt, angle_opt, dt=0.001)

def plot_trajectory(trajet):
    plt.figure(figsize=(12, 6))
    plt.plot(trajet.pos_x_list, trajet.pos_y_list, color='blue')
    plt.title('Badminton Shuttlecock Trajectory')
    plt.xlabel('Position X (meters)')
    plt.ylabel('Position Y (meters)')
    plt.grid(True)
    plt.show()

plot_trajectory(trajet)

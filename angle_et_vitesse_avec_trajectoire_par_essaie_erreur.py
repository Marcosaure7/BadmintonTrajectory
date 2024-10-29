from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

class BadmintonTrajectory:
    def __init__(self, vitesse_ini, angle):
        self.pos_x_list = []
        self.pos_y_list = []
        self.affichage_des_coo(vitesse_ini, angle)

    def affichage_des_coo(self, vitesse_ini, angle):
        angle_rad = np.radians(angle)
        vitesse_x = np.cos(angle_rad) * vitesse_ini
        vitesse_y = np.sin(angle_rad) * vitesse_ini
        pos_x = 0
        pos_y = 1
        dt = 0.001
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
    def objective(params):
        vitesse_ini, angle = params
        trajet = BadmintonTrajectory(vitesse_ini, angle)
        return trajet.distance_to_points(points)

    bounds = [(7, 35), (15, 45)]  # Vos conditions initiales
    initial_guess = [30, 30]

    result = minimize(objective, initial_guess, bounds=bounds)
    return result.x  # Renvoie la vitesse et l'angle optimaux

# Exemple d'utilisation
points = [(7,7 ), (0, 1), (10, 0)]
vitesse_opt, angle_opt = optimize_trajectory(points)
print(f"Conditions initiales optimisées: vitesse = {vitesse_opt:.2f} m/s, angle = {angle_opt:.2f}°")

# Tracer la trajectoire avec les conditions optimisées
trajet = BadmintonTrajectory(vitesse_opt, angle_opt)


def plot_trajectory(trajet):
    plt.figure(figsize=(12, 6))
plt.plot(trajet.pos_x_list, trajet.pos_y_list, color='blue')
plt.title('Trajectoire du volant de badminton')
plt.xlabel('Position X (mètres)')
plt.ylabel('Position Y (mètres)')
plt.grid(True)
plt.show()


plot_trajectory(trajet)


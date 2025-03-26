from scipy.optimize import differential_evolution, minimize
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools

dt_DE = 0.01
dt_SLSQP = 0.001
co_fraine = 0.25

class BadmintonTrajectory:
    def __init__(self, vitesse_ini, angle, dt):
        self.affichage_des_coo(vitesse_ini, angle, dt)

    def affichage_des_coo(self, vitesse_ini, angle, dt):
        angle_rad = np.radians(angle)
        v_x = np.cos(angle_rad) * vitesse_ini
        v_y = np.sin(angle_rad) * vitesse_ini
        pos_x, pos_y = -1.98, 1.55

        # Utilisation de listes puis conversion en np.array pour éviter des redimensionnements répétés
        pos_x_list = [pos_x]
        pos_y_list = [pos_y]

        while pos_y >= 0:
            # Calcul unique de la norme de la vitesse
            v_norm = np.hypot(v_x, v_y)
            # Mise à jour des vitesses avec accélération combinée
            v_x += -co_fraine * v_x * v_norm * dt
            v_y += -(9.81 + co_fraine * v_y * v_norm) * dt

            pos_x += v_x * dt
            pos_y += v_y * dt

            pos_x_list.append(pos_x)
            pos_y_list.append(pos_y)

        self.pos_x_list = np.array(pos_x_list)
        self.pos_y_list = np.array(pos_y_list)

    def get_y_at_x0(self):
        # Conversion déjà effectuée dans affichage_des_coo
        idx = np.searchsorted(self.pos_x_list, 0)
        if idx == 0 or idx >= len(self.pos_x_list):
            return 0
        x1, x2 = self.pos_x_list[idx - 1], self.pos_x_list[idx]
        y1, y2 = self.pos_y_list[idx - 1], self.pos_y_list[idx]
        return y1 + (-x1) * (y2 - y1) / (x2 - x1)

    def distance_to_points(self, points, weights=None):
        filtered_points = [(px, py, w) for (px, py), w in zip(points, weights) if w > 0]

        distances = []
        for px, py, weight in filtered_points:
            min_distance = min(
                np.sqrt((px - x)**2 + (py - y)**2)
                for x, y in zip(self.pos_x_list, self.pos_y_list)
            )
            distances.append(weight * min_distance)
        total_distance = sum(distances)

        # Pénalité ajustée
        y_at_x0 = self.get_y_at_x0()
        hauteur_au_filet = points[0][1]
        if y_at_x0 <= 1.55:
            penalty = (1.55 - y_at_x0) * 1000
            total_distance += penalty
        elif y_at_x0 < hauteur_au_filet:
            penalty = (hauteur_au_filet - y_at_x0) * 500 * weights[0]
            total_distance += penalty

        return total_distance

def optimize_trajectory(distance_atterissage, hauteur_au_filet=None, liste_Points=None):
    start_time = time.time()
    distance_atterissage = min(distance_atterissage, 7.36)

    if hauteur_au_filet is not None:
        hauteur_au_filet = max(hauteur_au_filet, 1.6)
        importance_hauteur_filet = -np.log(hauteur_au_filet - 1.55) + 1
    else:
        hauteur_au_filet = 0.05 * (distance_atterissage - 0.5)**2 + 1.6
        importance_hauteur_filet = -np.log(hauteur_au_filet - 1.55) + 1
        if hauteur_au_filet >= 3:
            importance_hauteur_filet = 0

    importance_distance_atterissage = -2 * np.sqrt(distance_atterissage) + 0.3 * distance_atterissage + 4 #valeur arbitraire

    points = [(0, hauteur_au_filet), (distance_atterissage, 0)] + (liste_Points or [])
    # Attribution des poids : le premier et le deuxième point ont une importance spécifique
    weights = [importance_hauteur_filet, importance_distance_atterissage] + [1] * (len(points) - 2)

    def objective_de(params):
        vitesse_ini, angle = params
        trajet = BadmintonTrajectory(vitesse_ini, angle, dt=dt_DE)
        return trajet.distance_to_points(points, weights)

    bounds = [(7, 35), (0, 85)]
    result_de = differential_evolution(objective_de, bounds)
    vitesse_de, angle_de = result_de.x

    def objective_slsqp(params):
        vitesse_ini, angle = params
        trajet = BadmintonTrajectory(vitesse_ini, angle, dt=dt_SLSQP)
        return trajet.distance_to_points(points, weights)

    initial_guess = [vitesse_de, angle_de]
    result_slsqp = minimize(objective_slsqp, initial_guess, bounds=bounds, method='SLSQP')
    vitesse_slsqp, angle_slsqp = result_slsqp.x

    elapsed_time = time.time() - start_time
    print(f"Temps d’optimisation : {elapsed_time:.2f} secondes")
    return vitesse_slsqp, angle_slsqp

# Exemple d’utilisation
distance_atterissage = 0.1
hauteur_au_filet = 1.6
points = None

vitesse_opt, angle_opt = optimize_trajectory(distance_atterissage, hauteur_au_filet, points)
print(f"Conditions initiales optimisées : vitesse = {vitesse_opt:.2f} m/s, angle = {angle_opt:.2f}°")

trajet = BadmintonTrajectory(vitesse_opt, angle_opt, dt=0.001)

def plot_trajectory(trajet):
    plt.figure(figsize=(12, 6))
    plt.plot(trajet.pos_x_list, trajet.pos_y_list, color='blue')
    plt.title('Trajectoire du volant de badminton')
    plt.xlabel('Position X (m)')
    plt.ylabel('Position Y (m)')
    plt.grid(True)
    plt.show()

plot_trajectory(trajet)

from scipy.optimize import differential_evolution, minimize
import numpy as np
import time

# Constantes globales
dt_DE = 0.01  # Pas de temps pour l'évolution différentielle
dt_SLSQP = 0.001  # Pas de temps pour SLSQP
co_fraine = 0.25  # Coefficient de frottement


class BadmintonTrajectory:
    """Classe interne pour simuler et évaluer une trajectoire de volant de badminton."""

    def __init__(self, vitesse_ini, angle_verticale, position_filet, dt):
        """
        Initialise la trajectoire avec les paramètres donnés et calcule immédiatement les positions.

        Args:
            vitesse_ini (float): Vitesse initiale en m/s
            angle_verticale (float): Angle de tir en degrés
            dt (float): Pas de temps pour la simulation
        """
        self.calculer_traj(vitesse_ini, angle_verticale, position_filet, dt)

    def calculer_traj(self, vitesse_ini, angle_verticale, position_filet, dt):
        """Calcule les positions x et y de la trajectoire jusqu'à ce que le volant touche le sol."""
        angle_rad = np.radians(angle_verticale)
        v_x = np.cos(angle_rad) * vitesse_ini
        v_y = np.sin(angle_rad) * vitesse_ini
        pos_x, pos_y = position_filet, 1.55  # Position initiale (côté joueur, hauteur standard)

        pos_x_list = [pos_x]
        pos_y_list = [pos_y]

        while pos_y >= 0:
            v_norm = np.hypot(v_x, v_y)
            v_x += -co_fraine * v_x * v_norm * dt
            v_y += -(9.81 + co_fraine * v_y * v_norm) * dt
            pos_x += v_x * dt
            pos_y += v_y * dt
            pos_x_list.append(pos_x)
            pos_y_list.append(pos_y)

        self.pos_x_list = np.array(pos_x_list)
        self.pos_y_list = np.array(pos_y_list)

    def get_y_at_x0(self):
        """Calcule la hauteur y au point x=0 (filet) par interpolation linéaire."""
        idx = np.searchsorted(self.pos_x_list, 0)
        if idx == 0 or idx >= len(self.pos_x_list):
            return 0
        x1, x2 = self.pos_x_list[idx - 1], self.pos_x_list[idx]
        y1, y2 = self.pos_y_list[idx - 1], self.pos_y_list[idx]
        return y1 + (-x1) * (y2 - y1) / (x2 - x1)

    def distance_to_points(self, points, weights=None):
        """Calcule la distance totale pondérée entre la trajectoire et les points cibles."""
        filtered_points = [(px, py, w) for (px, py), w in zip(points, weights) if w > 0]
        distances = []
        for px, py, weight in filtered_points:
            min_distance = min(
                np.sqrt((px - x) ** 2 + (py - y) ** 2)
                for x, y in zip(self.pos_x_list, self.pos_y_list)
            )
            distances.append(weight * min_distance)
        total_distance = sum(distances)

        # Pénalité pour la hauteur au filet
        y_at_x0 = self.get_y_at_x0()
        hauteur_au_filet = points[0][1]
        if y_at_x0 <= 1.55:
            penalty = (1.55 - y_at_x0) * 1000
            total_distance += penalty
        elif y_at_x0 < hauteur_au_filet:
            penalty = (hauteur_au_filet - y_at_x0) * 500 * weights[0]
            total_distance += penalty

        return total_distance


def optimize_trajectory(distance_atterissage_axe_x, hauteur_au_filet=None, liste_Points_2D=None, angle_horizontale_rad=None):
    """
    Optimise la trajectoire d'un volant de badminton pour passer par des points cibles.

    Args:
        distance_atterissage (float): Distance cible d'atterrissage en mètres selon l'axe des x a partir du filet
        hauteur_au_filet (float, optional): Hauteur cible au filet en mètres
        liste_Points_2D (list, optional): Liste de points 2D [(x1, y1), (x2, y2), ...]

    Returns:
        tuple: (vitesse_initiale, angle) optimaux en m/s et degrés
    """

    distance_filet = 1.98 / np.cos(angle_horizontale_rad)
    distance_atterissage = distance_atterissage_axe_x / np.cos(angle_horizontale_rad)

    if np.isclose(np.sin(angle_horizontale_rad), 0):
        terme1 = float('inf')  # Évite la division par zéro
    else:
        terme1 = 3.05 / np.abs(np.sin(angle_horizontale_rad)) - distance_filet

    if np.isclose(np.cos(angle_horizontale_rad), 0):
        terme2 = float('inf')
    else:
        terme2 = 6.7 / np.cos(angle_horizontale_rad)

    distance_atterissage_max = min(terme1, terme2)

    start_time = time.time()
    distance_atterissage = min(distance_atterissage, distance_atterissage_max)  # Limite à la longueur du court

    # Gestion de la hauteur au filet
    if hauteur_au_filet is not None:
        hauteur_au_filet = max(hauteur_au_filet, 1.6)
        importance_hauteur_filet = -np.log(hauteur_au_filet - 1.55) + 1
    else:
        hauteur_au_filet = 0.05 * (distance_atterissage - 0.5) ** 2 + 1.6
        importance_hauteur_filet = -np.log(hauteur_au_filet - 1.55) + 1
        if hauteur_au_filet >= 3:
            importance_hauteur_filet = 0

    importance_distance_atterissage = -2 * np.sqrt(distance_atterissage) + 0.3 * distance_atterissage + 4

    # Définition des points et poids
    points = [(0, hauteur_au_filet), (distance_atterissage, 0)] + (liste_Points_2D or [])
    weights = [importance_hauteur_filet, importance_distance_atterissage] + [1] * (len(points) - 2)

    # Fonction objective pour l'évolution différentielle
    def objective_de(params):
        vitesse_ini, angle = params
        trajet = BadmintonTrajectory(vitesse_ini, angle, -distance_filet , dt=dt_DE)
        return trajet.distance_to_points(points, weights)

    # Optimisation globale avec évolution différentielle
    bounds = [(7, 35), (0, 85)]  # Vitesse: 7-35 m/s, Angle: 0-85°
    result_de = differential_evolution(objective_de, bounds)
    vitesse_de, angle_de = result_de.x

    # Fonction objective pour SLSQP
    def objective_slsqp(params):
        vitesse_ini, angle = params
        trajet = BadmintonTrajectory(vitesse_ini, angle, -distance_filet, dt=dt_SLSQP)
        return trajet.distance_to_points(points, weights)

    # Optimisation locale avec SLSQP
    initial_guess = [vitesse_de, angle_de]
    result_slsqp = minimize(objective_slsqp, initial_guess, bounds=bounds, method='SLSQP')
    vitesse_slsqp, angle_slsqp = result_slsqp.x

    elapsed_time = time.time() - start_time
    print(f"Temps d’optimisation : {elapsed_time:.2f} secondes")
    return vitesse_slsqp, angle_slsqp

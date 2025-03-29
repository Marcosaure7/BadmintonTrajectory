import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

from SimulateurTrajPourOpti import SimulateurTrajPourOpti

DT_DE = 0.01
DT_SLSQP = 0.001

class OptimiseurTrajectoire:
    @staticmethod
    def calculer_distance_max(angle_horizontal_rad):
        terme1 = (3.05 / np.abs(np.sin(angle_horizontal_rad)) - 1.98 / np.cos(angle_horizontal_rad)
                  if not np.isclose(np.sin(angle_horizontal_rad), 0) else float('inf'))
        terme2 = (6.7 / np.cos(angle_horizontal_rad)
                  if not np.isclose(np.cos(angle_horizontal_rad), 0) else float('inf'))
        return min(terme1, terme2)

    @staticmethod
    def determiner_hauteur_net_cible(distance_atterrissage):
        return 0.05 * (distance_atterrissage - 0.5) ** 2 + 1.6

    @staticmethod
    def objectif(params, points_cibles, poids, position_net, dt):
        sim = SimulateurTrajPourOpti(params[0], params[1], position_net, dt)
        hauteur_net = sim.get_hauteur_net()
        distance_ponderee = sim.calculer_distance_ponderee(points_cibles, poids)
        if hauteur_net <= 1.55:
            distance_ponderee += (1.55 - hauteur_net) * 1000
        return distance_ponderee

    @staticmethod
    def optimiser_trajectoire(distance_atterrissage_x, angle_horizontal_rad,
                              hauteur_net_cible=None, liste_points_2D=None):

        distance_net = 1.98 / np.cos(angle_horizontal_rad)
        distance_atterrissage = distance_atterrissage_x / np.cos(angle_horizontal_rad)
        distance_max = OptimiseurTrajectoire.calculer_distance_max(angle_horizontal_rad)
        distance_atterrissage = min(distance_atterrissage, distance_max)

        if hauteur_net_cible is None:
            hauteur_net_cible = OptimiseurTrajectoire.determiner_hauteur_net_cible(distance_atterrissage)
        else:
            hauteur_net_cible = max(hauteur_net_cible, 1.6)

        importance_net = -np.log(hauteur_net_cible - 1.55) + 1 if hauteur_net_cible >= 1.55 else 0
        importance_atterrissage = -2 * np.sqrt(distance_atterrissage) + 0.3 * distance_atterrissage + 4

        points_cibles = [(0, hauteur_net_cible), (distance_atterrissage, 0)]
        poids = [importance_net, importance_atterrissage]

        if liste_points_2D:
            points_cibles.extend([(x / np.cos(angle_horizontal_rad), y) for x, y in liste_points_2D])
            poids.extend([1] * len(liste_points_2D))

        start_time = time.time()
        bounds = [(7, 35), (0, 85)]
        result_de = differential_evolution(OptimiseurTrajectoire.objectif, bounds,
                                           args=(points_cibles, poids, -distance_net, DT_DE))
        result_slsqp = minimize(OptimiseurTrajectoire.objectif, result_de.x, bounds=bounds, method='SLSQP',
                                args=(points_cibles, poids, -distance_net, DT_SLSQP))
        elapsed_time = time.time() - start_time
        print(f"Temps d’optimisation : {elapsed_time:.2f} secondes")
        return result_slsqp.x


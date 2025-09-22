import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

from SimulateurTrajPourOpti import SimulateurTrajPourOptiGPU

DT_DE = 0.0005

class OptimiseurTrajectoire:
    @staticmethod
    def calculer_distance_max(angle_horizontal_rad):
        s, c = np.sin(angle_horizontal_rad), np.cos(angle_horizontal_rad)
        terme1 = (3.05 / np.abs(s) - 1.98 / c) if not np.isclose(s, 0) else float("inf")
        terme2 = (6.7 / c) if not np.isclose(c, 0) else float("inf")
        return min(terme1, terme2)

    @staticmethod
    def determiner_hauteur_net_cible(distance_atterrissage):
        return 0.05 * (distance_atterrissage - 0.5) ** 2 + 1.6

    @staticmethod
    def optimiser_trajectoire(distance_atterrissage_x, angle_horizontal_rad,
                              hauteur_net_cible=None, liste_points_2D=None,
                              popsize=128, maxiter=120, t_max=8.0, device=None):
        """
        Renvoie [vitesse_initiale, angle_vertical_deg].
        """
        c = np.cos(angle_horizontal_rad)
        distance_net = 1.98 / c
        distance_atterrissage = distance_atterrissage_x / c
        distance_max = OptimiseurTrajectoire.calculer_distance_max(angle_horizontal_rad)
        distance_atterrissage = min(distance_atterrissage, distance_max)

        if hauteur_net_cible is None:
            hauteur_net_cible = OptimiseurTrajectoire.determiner_hauteur_net_cible(distance_atterrissage)
        else:
            hauteur_net_cible = max(hauteur_net_cible, 1.6)

        importance_net = -np.log(hauteur_net_cible - 1.55) + 1 if hauteur_net_cible >= 1.55 else 0.0
        importance_atterrissage = -2 * np.sqrt(distance_atterrissage) + 0.3 * distance_atterrissage + 4

        points_cibles = [(0.0, float(hauteur_net_cible)), (float(distance_atterrissage), 0.0)]
        poids = [float(importance_net), float(importance_atterrissage)]
        if liste_points_2D:
            points_cibles.extend([(x / c, y) for x, y in liste_points_2D])
            poids.extend([1.0] * len(liste_points_2D))

        position_net = -distance_net  # filet à x=0

        # Objectifs GPU (un pour DE en dt grossier, un pour SLSQP en dt fin)
        sim_de = SimulateurTrajPourOptiGPU(points_cibles, poids, position_net, dt=DT_DE, t_max=t_max, device=device)

        bounds = [(7.0, 35.0), (0.0, 85.0)]

        start_time = time.time()

        # Grid search instead of differential evolution
        delta_v = 0.12
        delta_theta = 360/800

        v_array = np.arange(bounds[0][0], bounds[0][1] + delta_v / 2.0, delta_v)
        theta_array = np.arange(bounds[1][0], bounds[1][1] + delta_theta / 2.0, delta_theta)

        V, Theta = np.meshgrid(v_array, theta_array, indexing='ij')
        params = np.stack((V.ravel(), Theta.ravel()), axis=1)

        costs = sim_de(params)

        idx = np.argmin(costs)
        best_params = params[idx]

        elapsed_time = time.time() - start_time
        print(f"Temps d’optimisation (GPU via WebGPU) : {elapsed_time:.2f} s")
        return best_params

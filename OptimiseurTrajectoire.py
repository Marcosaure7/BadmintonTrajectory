import time
import numpy as np
import plotly.graph_objects as go
from SimulateurTrajPourOpti import SimulateurTrajPourOptiGPU

DT_DE = 0.0005


class OptimiseurTrajectoire:

    def __init__(self, device=None):
        """
        Initialise l'optimiseur avec un simulateur GPU réutilisable.
        Les targets seront mis à jour via update_targets dans optimiser_trajectoire.
        """
        # Création une seule fois du simulateur GPU avec paramètres fixes (sans targets initiaux)
        self.sim = SimulateurTrajPourOptiGPU(
            dt=DT_DE, t_max=8.0, device=device
        )
        print("Simulateur GPU chargé et prêt pour réutilisation.")

    @staticmethod
    def calculer_distance_max(angle_horizontal_rad):
        s, c = np.sin(angle_horizontal_rad), np.cos(angle_horizontal_rad)
        terme1 = (3.05 / np.abs(s) - 1.98 / c) if not np.isclose(s, 0) else float("inf")
        terme2 = (6.7 / c) if not np.isclose(c, 0) else float("inf")
        return min(terme1, terme2)

    @staticmethod
    def determiner_hauteur_net_cible(distance_atterrissage):
        return 0.05 * (distance_atterrissage - 0.5) ** 2 + 1.6

    def optimiser_trajectoire(self, distance_atterrissage_x, angle_horizontal_rad,
                              hauteur_net_cible=None, liste_points_2D=None,
                              popsize=128, maxiter=120, t_max=8.0, device=None):
        """
        Renvoie [vitesse_initiale, angle_vertical_deg].
        Réutilise self.sim en mettant à jour les targets.
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

        # Mise à jour des targets dans le simulateur réutilisable (pas de nouvelle création)
        self.sim.update_targets(points_cibles, poids, position_net)

        bounds = [(7.0, 35.0), (0.0, 85.0)]

        start_time = time.perf_counter()

        # Grid search instead of differential evolution
        delta_v = 0.12 / 2
        delta_theta = 360 / 800 / 2

        # Pré-calcul des arrays pour éviter des recréations inutiles (micro-opti)
        v_array = np.linspace(bounds[0][0], bounds[0][1],
                              int((bounds[0][1] - bounds[0][0]) / delta_v) + 1)

        theta_array = np.linspace(bounds[1][0], bounds[1][1],
                                  int((bounds[1][1] - bounds[1][0]) / delta_theta) + 1)

        # Revenir à sparse=False pour compatibilité et simplicité (mémoire négligeable pour ~44k points)
        V, Theta = np.meshgrid(v_array, theta_array, indexing='ij')
        params = np.column_stack((V.ravel(), Theta.ravel()))  # Efficace pour 2 arrays 1D

        costs = self.sim(params)  # Réutilisation du simulateur

        idx = np.argmin(costs)
        best_params = params[idx]

        elapsed_time = time.perf_counter() - start_time
        print(f"Temps d’optimisation (GPU via WebGPU, réutilisation) : {elapsed_time * 1000 : .2f} ms ")

        cost2d = np.reshape(costs, (len(v_array), len(theta_array)))
        cost2d = np.clip(cost2d, None, 5)
        print(str(min(costs)))
        fig = go.Figure(data=[go.Surface(

            x=V,
            y=Theta,
            z=cost2d,
            colorscale="Viridis"
        )])

        fig.update_layout(
            scene=dict(
                xaxis_title="Vitesse (v_array)",
                yaxis_title="Theta (deg)",
                zaxis_title="Cost"
            )
        )

        fig.show()

        return best_params

    def destroy(self):
        """Optionnel : Détruit le simulateur GPU si plus besoin (libère VRAM)."""
        self.sim.destroy()
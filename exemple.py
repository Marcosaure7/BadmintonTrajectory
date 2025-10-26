import numpy as np

# import SimulateurTrajPourOpti  # Non utilisé directement ici (chargé via OptimiseurTrajectoire)
from BadmintonTrajectory import BadmintonTrajectory
from BadmintonTrajectoryVisualizer import BadmintonTrajectoryVisualizer
from OptimiseurTrajectoire import OptimiseurTrajectoire

if __name__ == "__main__":
    att_x = 7
    att_y = 0
    points = None  # Liste de points 2D optionnels, e.g., [(x1, y1), (x2, y2)]
    hauteur_net_cible = None  # On laisse None pour laisser l'optimiseur déterminer la hauteur au filet
    angle_horiz = np.arctan(att_y / (att_x + 1.98))

    # Création de l'instance de l'optimiseur (charge le simulateur GPU une seule fois)
    opti = OptimiseurTrajectoire(device=None)
    print("Optimiseur initialisé avec simulateur GPU.")

    # Optimisation de la trajectoire (réutilise le simulateur)
    vitesse_optimale, angle_optimal = opti.optimiser_trajectoire(
        distance_atterrissage_x=att_x,
        hauteur_net_cible=hauteur_net_cible,
        angle_horizontal_rad=angle_horiz,
        liste_points_2D=points,
    )

    print(f"Vitesse initiale optimale : {vitesse_optimale:.2f} m/s")
    print(f"Angle vertical optimal : {angle_optimal:.2f} degrés")

    # Calcul et visualisation de la trajectoire
    traj_x, traj_y, traj_z = BadmintonTrajectory.calculer_traj(
        vitesse_ini=vitesse_optimale,
        angle_vertical=angle_optimal,
        angle_horiz_rad=angle_horiz
    )
    BadmintonTrajectoryVisualizer.afficher_graphique_interactif(traj_x, traj_y, traj_z)

    # Libération des ressources GPU (bonne pratique)
    opti.destroy()
    print("Ressources GPU libérées.")

# Exemple d'utilisation
import numpy as np

from BadmintonTrajectory import BadmintonTrajectory
import calculer_param

if __name__ == "__main__":
    att_x = 6.7
    att_y = 3
    points = [(5, 5)]
    hauteur_au_filet = None
    angle_horiz = np.arctan(att_y / (att_x + 1.98))
    vitesse_optimale, angle_optimal = calculer_param.optimize_trajectory(distance_atterissage_axe_x=att_x,
                                                                         hauteur_au_filet=hauteur_au_filet,
                                                                         angle_horizontale_rad=angle_horiz,
                                                                         liste_Points_2D=points)

    print(f"  Vitesse initiale optimale : {vitesse_optimale:.2f} m/s")
    print(f"  Angle vertical optimal : {angle_optimal:.2f} degrés")

    traj = BadmintonTrajectory(
        vitesse_ini=vitesse_optimale,
        angle_vertical=angle_optimal,
        angle_horizontal=angle_horiz,
        pos_init=(-1.98, 1.55, 0.0),

    )
    traj.afficher_graphique_interactif()

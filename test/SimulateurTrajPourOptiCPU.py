import numpy as np

COEFFICIENT_FROTTEMENT = 0.25

class SimulateurTrajPourOptiCPU:
    def __init__(self, vitesse_initiale, angle_vertical_deg, position_net=0.0, dt=1e-3):
        self.simuler_trajectoire(vitesse_initiale, angle_vertical_deg, position_net, dt)

    def simuler_trajectoire(self, vitesse_initiale, angle_vertical_deg, position_net, dt):
        angle_rad = np.radians(angle_vertical_deg)

        # vitesses initiales
        v_x = np.cos(angle_rad) * vitesse_initiale
        v_y = np.sin(angle_rad) * vitesse_initiale

        # positions initiales
        pos_x, pos_y = position_net, 1.55

        self.pos_x_list, self.pos_y_list = [pos_x], [pos_y]

        # simulation
        while pos_y >= 0:
            vitesse_normale = np.hypot(v_x, v_y)

            v_x += -COEFFICIENT_FROTTEMENT * v_x * vitesse_normale * dt
            v_y += -(9.81 + COEFFICIENT_FROTTEMENT * v_y * vitesse_normale) * dt

            pos_x += v_x * dt
            pos_y += v_y * dt

            self.pos_x_list.append(pos_x)
            self.pos_y_list.append(pos_y)

        self.pos_x_list = np.array(self.pos_x_list, dtype=np.float64)
        self.pos_y_list = np.array(self.pos_y_list, dtype=np.float64)

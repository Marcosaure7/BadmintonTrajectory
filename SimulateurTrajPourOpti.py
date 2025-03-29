import numpy as np

COEFFICIENT_FROTTEMENT = 0.25

class SimulateurTrajPourOpti:
    def __init__(self, vitesse_initiale, angle_vertical, position_net, dt):
        self.simuler_trajectoire(vitesse_initiale, angle_vertical, position_net, dt)

    def simuler_trajectoire(self, vitesse_initiale, angle_vertical, position_net, dt):
        angle_rad = np.radians(angle_vertical)
        v_x = np.cos(angle_rad) * vitesse_initiale
        v_y = np.sin(angle_rad) * vitesse_initiale
        pos_x, pos_y = position_net, 1.55
        self.pos_x_list, self.pos_y_list = [pos_x], [pos_y]

        while pos_y >= 0:
            vitesse_normale = np.hypot(v_x, v_y)
            v_x += -COEFFICIENT_FROTTEMENT * v_x * vitesse_normale * dt
            v_y += -(9.81 + COEFFICIENT_FROTTEMENT * v_y * vitesse_normale) * dt
            pos_x += v_x * dt
            pos_y += v_y * dt
            self.pos_x_list.append(pos_x)
            self.pos_y_list.append(pos_y)

        self.pos_x_list, self.pos_y_list = np.array(self.pos_x_list), np.array(self.pos_y_list)

    def get_hauteur_net(self):
        idx = np.searchsorted(self.pos_x_list, 0)
        if idx == 0 or idx >= len(self.pos_x_list):
            return 0
        x1, x2 = self.pos_x_list[idx - 1], self.pos_x_list[idx]
        y1, y2 = self.pos_y_list[idx - 1], self.pos_y_list[idx]
        return y1 + (-x1) * (y2 - y1) / (x2 - x1)

    def calculer_distance_ponderee(self, points_cibles, poids):
        distance_ponderee = sum(
            p * np.min(np.sqrt((px - self.pos_x_list) ** 2 + (py - self.pos_y_list) ** 2))
            for (px, py), p in zip(points_cibles, poids) if p > 0)
        return distance_ponderee
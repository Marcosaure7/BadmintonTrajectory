import cupy as cp

COEFFICIENT_FROTTEMENT = 0.25

class SimulateurTrajPourOptiGPU:
    def __init__(self, vitesse_initiale, angle_vertical_deg, position_net=0.0, dt=1e-3):
        self.simuler_trajectoire(vitesse_initiale, angle_vertical_deg, position_net, dt)

    def simuler_trajectoire(self, vitesse_initiale, angle_vertical_deg, position_net, dt):
        angle_rad = cp.radians(cp.float64(angle_vertical_deg))

        # vitesses initiales
        v_x = cp.cos(angle_rad) * cp.float64(vitesse_initiale)
        v_y = cp.sin(angle_rad) * cp.float64(vitesse_initiale)

        # positions initiales
        pos_x, pos_y = cp.float64(position_net), cp.float64(1.55)

        pos_x_list = [pos_x]
        pos_y_list = [pos_y]

        # simulation
        while pos_y >= 0:
            vitesse_normale = cp.hypot(v_x, v_y)

            v_x += -COEFFICIENT_FROTTEMENT * v_x * vitesse_normale * dt
            v_y += -(9.81 + COEFFICIENT_FROTTEMENT * v_y * vitesse_normale) * dt

            pos_x += v_x * dt
            pos_y += v_y * dt

            pos_x_list.append(pos_x)
            pos_y_list.append(pos_y)

        self.pos_x_list = cp.asnumpy(cp.array(pos_x_list, dtype=cp.float64))
        self.pos_y_list = cp.asnumpy(cp.array(pos_y_list, dtype=cp.float64))

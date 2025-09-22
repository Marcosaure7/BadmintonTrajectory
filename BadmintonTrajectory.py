import numpy as np
# Constante de frottement
co_fraine = (0.25)

SERVICE_DIST = (1.98)


class BadmintonTrajectory:
    @staticmethod
    def calculer_traj(vitesse_ini, angle_vertical, angle_horiz_rad, dt = (0.0001), pos_init=(-SERVICE_DIST, (1.55), (0.0))):
        angle_vert_rad = np.radians(angle_vertical)

        v_x = (vitesse_ini) * np.cos(angle_vert_rad) * np.cos(angle_horiz_rad)
        v_y = (vitesse_ini) * np.sin(angle_vert_rad)
        v_z = (vitesse_ini) * np.cos(angle_vert_rad) * np.sin(angle_horiz_rad)

        pos_x, pos_y, pos_z = pos_init
        pos_x_list = [pos_x]
        pos_y_list = [pos_y]
        pos_z_list = [pos_z]

        while pos_y >= 0:
            v_norm = np.sqrt(v_x ** 2 + v_y ** 2 + v_z ** 2)
            v_x += -co_fraine * v_x * v_norm * dt
            v_y += -((9.81) + co_fraine * v_y * v_norm) * dt
            v_z += -co_fraine * v_z * v_norm * dt

            pos_x += v_x * dt
            pos_y += v_y * dt
            pos_z += v_z * dt

            pos_x_list.append(pos_x)
            pos_y_list.append(pos_y)
            pos_z_list.append(pos_z)

        pos_x_list = np.array(pos_x_list)
        pos_y_list = np.array(pos_y_list)
        pos_z_list = np.array(pos_z_list)
        return np.array(pos_x_list), np.array(pos_y_list), np.array(pos_z_list)
import numpy as np
# Constante de frottement
co_fraine = (0.25)

SERVICE_DIST = (1.98)


class BadmintonTrajectory:
    @staticmethod
    def calculer_traj(vitesse_ini, angle_vertical, angle_horiz_rad, dt = (0.01), pos_init=(-SERVICE_DIST, (1.55), (0.0))):
        angle_vert_rad = np.radians(angle_vertical)
        temps_total = 0
        v_x = (vitesse_ini) * np.cos(angle_vert_rad) * np.cos(angle_horiz_rad)
        v_y = (vitesse_ini) * np.sin(angle_vert_rad)
        v_z = (vitesse_ini) * np.cos(angle_vert_rad) * np.sin(angle_horiz_rad)

        pos_x, pos_y, pos_z = pos_init
        pos_x_list = [pos_x]
        pos_y_list = [pos_y]
        pos_z_list = [pos_z]

        g = 9.81
        k = co_fraine

        while pos_y >= 0:
            # RK4 stage 1
            speed1 = np.sqrt(v_x**2 + v_y**2 + v_z**2)
            ax1 = -k * v_x * speed1
            ay1 = -g - k * v_y * speed1
            az1 = -k * v_z * speed1
            k1_x = dt * v_x
            k1_y = dt * v_y
            k1_z = dt * v_z
            k1_vx = dt * ax1
            k1_vy = dt * ay1
            k1_vz = dt * az1

            # RK4 stage 2
            v_x2 = v_x + 0.5 * k1_vx
            v_y2 = v_y + 0.5 * k1_vy
            v_z2 = v_z + 0.5 * k1_vz
            speed2 = np.sqrt(v_x2**2 + v_y2**2 + v_z2**2)
            ax2 = -k * v_x2 * speed2
            ay2 = -g - k * v_y2 * speed2
            az2 = -k * v_z2 * speed2
            k2_x = dt * (v_x + 0.5 * k1_vx)
            k2_y = dt * (v_y + 0.5 * k1_vy)
            k2_z = dt * (v_z + 0.5 * k1_vz)
            k2_vx = dt * ax2
            k2_vy = dt * ay2
            k2_vz = dt * az2

            # RK4 stage 3
            v_x3 = v_x + 0.5 * k2_vx
            v_y3 = v_y + 0.5 * k2_vy
            v_z3 = v_z + 0.5 * k2_vz
            speed3 = np.sqrt(v_x3**2 + v_y3**2 + v_z3**2)
            ax3 = -k * v_x3 * speed3
            ay3 = -g - k * v_y3 * speed3
            az3 = -k * v_z3 * speed3
            k3_x = dt * (v_x + 0.5 * k2_vx)
            k3_y = dt * (v_y + 0.5 * k2_vy)
            k3_z = dt * (v_z + 0.5 * k2_vz)
            k3_vx = dt * ax3
            k3_vy = dt * ay3
            k3_vz = dt * az3

            # RK4 stage 4
            v_x4 = v_x + k3_vx
            v_y4 = v_y + k3_vy
            v_z4 = v_z + k3_vz
            speed4 = np.sqrt(v_x4**2 + v_y4**2 + v_z4**2)
            ax4 = -k * v_x4 * speed4
            ay4 = -g - k * v_y4 * speed4
            az4 = -k * v_z4 * speed4
            k4_x = dt * (v_x + k3_vx)
            k4_y = dt * (v_y + k3_vy)
            k4_z = dt * (v_z + k3_vz)
            k4_vx = dt * ax4
            k4_vy = dt * ay4
            k4_vz = dt * az4

            # Weighted averages for deltas
            delta_x = (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x) / 6.0
            delta_y = (k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y) / 6.0
            delta_z = (k1_z + 2.0 * k2_z + 2.0 * k3_z + k4_z) / 6.0
            delta_vx = (k1_vx + 2.0 * k2_vx + 2.0 * k3_vx + k4_vx) / 6.0
            delta_vy = (k1_vy + 2.0 * k2_vy + 2.0 * k3_vy + k4_vy) / 6.0
            delta_vz = (k1_vz + 2.0 * k2_vz + 2.0 * k3_vz + k4_vz) / 6.0

            # Update positions and velocities
            pos_x += delta_x
            pos_y += delta_y
            pos_z += delta_z
            v_x += delta_vx
            v_y += delta_vy
            v_z += delta_vz

            pos_x_list.append(pos_x)
            pos_y_list.append(pos_y)
            pos_z_list.append(pos_z)
            temps_total += dt

            # Break if y < 0 after update (to avoid continuing after ground hit)
            if pos_y < 0:
                break

        pos_x_list = np.array(pos_x_list)
        pos_y_list = np.array(pos_y_list)
        pos_z_list = np.array(pos_z_list)

        print("Temps_total:", temps_total)
        return np.array(pos_x_list), np.array(pos_y_list), np.array(pos_z_list)
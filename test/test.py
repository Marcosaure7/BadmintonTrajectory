import matplotlib.pyplot as plt
import numpy as np

# ====== Code 1 : 2D ======
COEFFICIENT_FROTTEMENT = np.float32(0.25)
GRAVITE = np.float32(9.81)

class SimulateurTrajPourOpti:
    def __init__(self, vitesse_initiale, angle_vertical, position_net, dt):
        self.simuler_trajectoire(
            np.float32(vitesse_initiale),
            np.float32(angle_vertical),
            np.float32(position_net),
            np.float32(dt)
        )

    def simuler_trajectoire(self, vitesse_initiale, angle_vertical, position_net, dt):
        angle_rad = np.radians(angle_vertical).astype(np.float32)

        v_x = np.cos(angle_rad, dtype=np.float32) * vitesse_initiale
        v_y = np.sin(angle_rad, dtype=np.float32) * vitesse_initiale

        pos_x, pos_y = np.float32(position_net), np.float32(1.55)

        self.pos_x_list = [pos_x]
        self.pos_y_list = [pos_y]

        while pos_y >= 0:
            vitesse_normale = np.hypot(v_x, v_y).astype(np.float32)

            v_x += -COEFFICIENT_FROTTEMENT * v_x * vitesse_normale * dt
            v_y += -(GRAVITE + COEFFICIENT_FROTTEMENT * v_y * vitesse_normale) * dt

            pos_x += v_x * dt
            pos_y += v_y * dt

            self.pos_x_list.append(np.float32(pos_x))
            self.pos_y_list.append(np.float32(pos_y))

        self.pos_x_list = np.array(self.pos_x_list, dtype=np.float32)
        self.pos_y_list = np.array(self.pos_y_list, dtype=np.float32)


# ====== Code 2 : 3D ======
co_fraine = np.float32(0.25)
SERVICE_DIST = np.float32(1.98)

class BadmintonTrajectory:
    @staticmethod
    def calculer_traj(
        vitesse_ini,
        angle_vertical,
        angle_horiz_rad,
        dt=np.float32(0.0001),
        pos_init=(-SERVICE_DIST, np.float32(1.55), np.float32(0.0))
    ):
        vitesse_ini = np.float32(vitesse_ini)
        angle_vertical = np.float32(angle_vertical)
        angle_horiz_rad = np.float32(angle_horiz_rad)
        dt = np.float32(dt)

        angle_vert_rad = np.radians(angle_vertical).astype(np.float32)

        v_x = vitesse_ini * np.cos(angle_vert_rad, dtype=np.float32) * np.cos(angle_horiz_rad, dtype=np.float32)
        v_y = vitesse_ini * np.sin(angle_vert_rad, dtype=np.float32)
        v_z = vitesse_ini * np.cos(angle_vert_rad, dtype=np.float32) * np.sin(angle_horiz_rad, dtype=np.float32)

        pos_x, pos_y, pos_z = map(np.float32, pos_init)

        pos_x_list = [pos_x]
        pos_y_list = [pos_y]
        pos_z_list = [pos_z]

        while pos_y >= 0:
            v_norm = np.sqrt(v_x ** 2 + v_y ** 2 + v_z ** 2, dtype=np.float32)

            v_x += -co_fraine * v_x * v_norm * dt
            v_y += -(GRAVITE + co_fraine * v_y * v_norm) * dt
            v_z += -co_fraine * v_z * v_norm * dt

            pos_x += v_x * dt
            pos_y += v_y * dt
            pos_z += v_z * dt

            pos_x_list.append(np.float32(pos_x))
            pos_y_list.append(np.float32(pos_y))
            pos_z_list.append(np.float32(pos_z))

        return (
            np.array(pos_x_list, dtype=np.float32),
            np.array(pos_y_list, dtype=np.float32),
            np.array(pos_z_list, dtype=np.float32)
        )


# ====== Test comparatif ======
if __name__ == "__main__":
    vitesse = 30   # m/s
    angle = 30     # deg
    dt = 0.0001

    # Code 1 (2D)
    sim = SimulateurTrajPourOpti(vitesse, angle, -SERVICE_DIST, dt)

    # Code 2 (3D avec angle horiz = 0 → équivalent 2D)
    x2, y2, z2 = BadmintonTrajectory.calculer_traj(vitesse, angle, np.float32(0), dt)

    # Tracé comparatif
    plt.figure(figsize=(8,5))
    plt.plot(sim.pos_x_list, sim.pos_y_list, label="Code 1 : SimulateurTrajPourOpti (2D)")
    plt.plot(x2, y2, '--', label="Code 2 : BadmintonTrajectory (3D, horiz=0)")

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Comparaison des deux simulateurs (float32)")
    plt.legend()
    plt.grid(True)
    plt.show()
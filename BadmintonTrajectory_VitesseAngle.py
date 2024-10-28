import numpy as np
import matplotlib.pyplot as plt


class BadmintonTrajectory:
    def __init__(self, vitesse_ini, angle):
        self.pos_x_list = []
        self.pos_y_list = []
        self.affichage_des_coo(vitesse_ini, angle)

    def affichage_des_coo(self, vitesse_ini, angle):
        angle_rad = np.radians(angle)
        vitesse_x = np.cos(angle_rad) * vitesse_ini
        vitesse_y = np.sin(angle_rad) * vitesse_ini
        pos_x = 0
        pos_y = 1
        dt = 0.001
        vitesse_x_prec = vitesse_x

        self.pos_x_list.append(pos_x)
        self.pos_y_list.append(pos_y)

        # Mise à jour des positions et vitesses
        while pos_y >= 0:
            vitesse_x += acceleration_x(vitesse_x, vitesse_y, dt)
            vitesse_y += acceleration_y(vitesse_x_prec, vitesse_y, dt)
            vitesse_x_prec = vitesse_x

            pos_x += vitesse_x * dt
            pos_y += vitesse_y * dt

            self.pos_x_list.append(pos_x)
            self.pos_y_list.append(pos_y)


def acceleration_x(vitesse_x, vitesse_y, increment):
    return -0.22 * vitesse_x * np.sqrt(vitesse_x**2 + vitesse_y**2) * increment

def acceleration_y(vitesse_x, vitesse_y, increment):
    return -(9.81 + 0.22 * vitesse_y * np.sqrt(vitesse_x**2 + vitesse_y**2)) * increment

def plot_trajectory(trajet):
    plt.figure(figsize=(12, 6))
    plt.plot(trajet.pos_x_list, trajet.pos_y_list, color='blue')

    # Définition des étiquettes et du titre
    plt.title('Trajectoire du volant de badminton')
    plt.xlabel('Position X (mètres)')
    plt.ylabel('Position Y (mètres)')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    traj = BadmintonTrajectory(33.33, 35)
    plot_trajectory(traj)
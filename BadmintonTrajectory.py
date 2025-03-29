import numpy as np
import plotly.graph_objects as go

# Constante de frottement
co_fraine = 0.25

# Dimensions officielles (en mètres)
LONGUEUR_TERRAIN = 13.4  # longueur totale (donc 6.7 m de chaque côté du filet)
LARGEUR_TERRAIN = 6.1    # pour doubles (±3.05 m)
LARGEUR_SINGLES = 5.18   # pour singles (±2.59 m)
SERVICE_DIST = 1.98
DECALAGE_SERVICE_SIMPLE = 0.76# distance du filet à la ligne de service court

class BadmintonTrajectory:
    def __init__(self, vitesse_ini, angle_vertical, angle_horizontal, dt = 0.001, pos_init=(-SERVICE_DIST, 1.55, 0.0)):
        self.calculer_traj(vitesse_ini, angle_vertical, angle_horizontal, dt, pos_init)

    def calculer_traj(self, vitesse_ini, angle_vertical, angle_horiz_rad, dt, pos_init):
        angle_vert_rad = np.radians(angle_vertical)

        v_x = vitesse_ini * np.cos(angle_vert_rad) * np.cos(angle_horiz_rad)
        v_y = vitesse_ini * np.sin(angle_vert_rad)
        v_z = vitesse_ini * np.cos(angle_vert_rad) * np.sin(angle_horiz_rad)

        pos_x, pos_y, pos_z = pos_init
        pos_x_list = [pos_x]
        pos_y_list = [pos_y]
        pos_z_list = [pos_z]

        while pos_y >= 0:
            v_norm = np.sqrt(v_x ** 2 + v_y ** 2 + v_z ** 2)
            v_x += -co_fraine * v_x * v_norm * dt
            v_y += -(9.81 + co_fraine * v_y * v_norm) * dt
            v_z += -co_fraine * v_z * v_norm * dt

            pos_x += v_x * dt
            pos_y += v_y * dt
            pos_z += v_z * dt

            pos_x_list.append(pos_x)
            pos_y_list.append(pos_y)
            pos_z_list.append(pos_z)

        self.pos_x_list = np.array(pos_x_list)
        self.pos_y_list = np.array(pos_y_list)
        self.pos_z_list = np.array(pos_z_list)

    def afficher_graphique_interactif(self):
        traces = []

        # Trajectoire du volant
        trace_traj = go.Scatter3d(
            x=self.pos_x_list,
            y=self.pos_z_list,
            z=self.pos_y_list,
            mode='lines',
            line=dict(color='blue', width=4),
            name='Trajectoire'
        )
        traces.append(trace_traj)

        # Contour du terrain doubles (noir)
        x_doubles = [-LONGUEUR_TERRAIN / 2, -LONGUEUR_TERRAIN / 2, LONGUEUR_TERRAIN / 2, LONGUEUR_TERRAIN / 2,
                     -LONGUEUR_TERRAIN / 2]
        y_doubles = [-LARGEUR_TERRAIN / 2, LARGEUR_TERRAIN / 2, LARGEUR_TERRAIN / 2, -LARGEUR_TERRAIN / 2,
                     -LARGEUR_TERRAIN / 2]
        z_doubles = [0] * len(x_doubles)
        trace_doubles = go.Scatter3d(
            x=x_doubles,
            y=y_doubles,
            z=z_doubles,
            mode='lines',
            line=dict(color='black', width=2),
            name='Terrain Doubles'
        )
        traces.append(trace_doubles)

        # Contour du terrain simples (rouge en tirets)
        x_singles = [-LONGUEUR_TERRAIN / 2, -LONGUEUR_TERRAIN / 2, LONGUEUR_TERRAIN / 2, LONGUEUR_TERRAIN / 2,
                     -LONGUEUR_TERRAIN / 2]
        y_singles = [-LARGEUR_SINGLES / 2, LARGEUR_SINGLES / 2, LARGEUR_SINGLES / 2, -LARGEUR_SINGLES / 2,
                     -LARGEUR_SINGLES / 2]
        z_singles = [0] * len(x_singles)
        trace_singles = go.Scatter3d(
            x=x_singles,
            y=y_singles,
            z=z_singles,
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Terrain Simples'
        )
        traces.append(trace_singles)

        # Lignes de service pour chaque camp (service court)
        # Camp joueur (x négatif)
        x_service_joueur = [-SERVICE_DIST, -SERVICE_DIST]
        trace_service_joueur_gauche = go.Scatter3d(
            x=x_service_joueur,
            y=[-LARGEUR_SINGLES / 2, 0],
            z=[0, 0],
            mode='lines',
            line=dict(color='orange', width=2),
            name='Service Court Joueur Gauche'
        )
        trace_service_joueur_droit = go.Scatter3d(
            x=x_service_joueur,
            y=[0, LARGEUR_SINGLES / 2],
            z=[0, 0],
            mode='lines',
            line=dict(color='orange', width=2),
            name='Service Court Joueur Droit'
        )
        # Camp adversaire (x positif)
        x_service_adv = [SERVICE_DIST, SERVICE_DIST]
        trace_service_adv_gauche = go.Scatter3d(
            x=x_service_adv,
            y=[-LARGEUR_SINGLES / 2, 0],
            z=[0, 0],
            mode='lines',
            line=dict(color='orange', width=2),
            name='Service Court Adversaire Gauche'
        )
        trace_service_adv_droit = go.Scatter3d(
            x=x_service_adv,
            y=[0, LARGEUR_SINGLES / 2],
            z=[0, 0],
            mode='lines',
            line=dict(color='orange', width=2),
            name='Service Court Adversaire Droit'
        )
        traces.extend([trace_service_joueur_gauche, trace_service_joueur_droit,
                       trace_service_adv_gauche, trace_service_adv_droit])

        # Lignes centrales dans le service
        trace_centre_joueur = go.Scatter3d(
            x=[-SERVICE_DIST, -LONGUEUR_TERRAIN / 2],
            y=[0, 0],
            z=[0, 0],
            mode='lines',
            line=dict(color='green', width=2),
            name='Centre Service Joueur'
        )
        trace_centre_adv = go.Scatter3d(
            x=[SERVICE_DIST, LONGUEUR_TERRAIN / 2],
            y=[0, 0],
            z=[0, 0],
            mode='lines',
            line=dict(color='green', width=2),
            name='Centre Service Adversaire'
        )
        traces.extend([trace_centre_joueur, trace_centre_adv])

        # Filet : ligne supérieure (en violet)
        trace_filet_sup = go.Scatter3d(
            x=[0, 0],
            y=[-LARGEUR_TERRAIN / 2, LARGEUR_TERRAIN / 2],
            z=[1.55, 1.55],
            mode='lines',
            line=dict(color='purple', width=4),
            name='Filet (haut)'
        )
        traces.append(trace_filet_sup)

        # Dessous du filet : surface verticale semi-transparente
        trace_filet_bas = go.Mesh3d(
            x=[0, 0, 0, 0],
            y=[-LARGEUR_TERRAIN / 2, LARGEUR_TERRAIN / 2, -LARGEUR_TERRAIN / 2, LARGEUR_TERRAIN / 2],
            z=[1.55, 1.55, 0, 0],
            i=[0, 1],
            j=[1, 3],
            k=[2, 2],
            color='purple',
            opacity=0.5,
            name='Dessous du filet'
        )
        traces.append(trace_filet_bas)

        # Trait supplémentaire allant de (-1.98, 0, 1.55) à (-1.98, 0, 0)
        trace_trait = go.Scatter3d(
            x=[-SERVICE_DIST, -SERVICE_DIST],
            y=[0, 0],
            z=[1.55, 0],
            mode='lines',
            line=dict(color='brown', width=4),
            name='Trait supplémentaire'
        )
        traces.append(trace_trait)

        # Lignes du corridor du fond (délimitant la zone doubles des zones non utilisées en simple)
        # Pour le camp adversaire (x = LONGUEUR_TERRAIN/2)
        trace_corridor_adv_gauche = go.Scatter3d(
            x=[LONGUEUR_TERRAIN / 2, LONGUEUR_TERRAIN / 2],
            y=[-LARGEUR_TERRAIN / 2, -LARGEUR_SINGLES / 2],
            z=[0, 0],
            mode='lines',
            line=dict(color='grey', width=2),
            name='Corridor Adversaire Gauche'
        )
        trace_corridor_adv_droit = go.Scatter3d(
            x=[LONGUEUR_TERRAIN / 2, LONGUEUR_TERRAIN / 2],
            y=[LARGEUR_SINGLES / 2, LARGEUR_TERRAIN / 2],
            z=[0, 0],
            mode='lines',
            line=dict(color='grey', width=2),
            name='Corridor Adversaire Droit'
        )
        # Pour le camp joueur (x = -LONGUEUR_TERRAIN/2)
        trace_corridor_joueur_gauche = go.Scatter3d(
            x=[-LONGUEUR_TERRAIN / 2, -LONGUEUR_TERRAIN / 2],
            y=[-LARGEUR_TERRAIN / 2, -LARGEUR_SINGLES / 2],
            z=[0, 0],
            mode='lines',
            line=dict(color='grey', width=2),
            name='Corridor Joueur Gauche'
        )
        trace_corridor_joueur_droit = go.Scatter3d(
            x=[-LONGUEUR_TERRAIN / 2, -LONGUEUR_TERRAIN / 2],
            y=[LARGEUR_SINGLES / 2, LARGEUR_TERRAIN / 2],
            z=[0, 0],
            mode='lines',
            line=dict(color='grey', width=2),
            name='Corridor Joueur Droit'
        )
        traces.extend([trace_corridor_adv_gauche, trace_corridor_adv_droit,
                       trace_corridor_joueur_gauche, trace_corridor_joueur_droit])

        # Lignes de service en simple (à 0,76 m du fond)
        # Pour le camp adversaire
        trace_service_simple_adv = go.Scatter3d(
            x=[LONGUEUR_TERRAIN / 2 - DECALAGE_SERVICE_SIMPLE, LONGUEUR_TERRAIN / 2 - DECALAGE_SERVICE_SIMPLE],
            y=[-LARGEUR_TERRAIN / 2, LARGEUR_TERRAIN / 2],
            z=[0, 0],
            mode='lines',
            line=dict(color='cyan', width=2),
            name='Service Simple Adversaire'
        )
        # Pour le camp joueur
        trace_service_simple_joueur = go.Scatter3d(
            x=[-LONGUEUR_TERRAIN / 2 + DECALAGE_SERVICE_SIMPLE, -LONGUEUR_TERRAIN / 2 + DECALAGE_SERVICE_SIMPLE],
            y=[-LARGEUR_TERRAIN / 2, LARGEUR_TERRAIN / 2],
            z=[0, 0],
            mode='lines',
            line=dict(color='cyan', width=2),
            name='Service Simple Joueur'
        )
        traces.extend([trace_service_simple_adv, trace_service_simple_joueur])

        # Configuration de la mise en page
        layout = go.Layout(
            title='Trajectoire 3D avec filet, corridor et lignes de service simples',
            scene=dict(
                xaxis=dict(title='X (m)', range=[-LONGUEUR_TERRAIN / 2 - 1, LONGUEUR_TERRAIN / 2 + 1]),
                yaxis=dict(title='Y (m)', range=[-LARGEUR_TERRAIN / 2 - 0.5, LARGEUR_TERRAIN / 2 + 0.5]),
                zaxis=dict(title='Hauteur (m)', range=[0, max(self.pos_y_list) * 1.2]),
                aspectmode='manual',
                aspectratio=dict(x=LONGUEUR_TERRAIN / 6.1, y=1, z=0.5)
            ),
            showlegend=True
        )

        fig = go.Figure(data=traces, layout=layout)
        fig.show()

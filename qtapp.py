import sys
import numpy as np
import ast
import plotly.graph_objects as go
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFormLayout
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QThread, pyqtSignal, QObject

# Tes imports
from BadmintonTrajectory import BadmintonTrajectory
from OptimiseurTrajectoire import OptimiseurTrajectoire
from BadmintonTrajectoryVisualizer import BadmintonTrajectoryVisualizer


# -------- Worker thread --------
class OptimisationWorker(QObject):
    finished = pyqtSignal(float, float, object, object, object)  # vitesse, angle, traj_x,y,z
    error = pyqtSignal(str)

    def __init__(self, opti, att_x, att_y, waypoints, hauteur_net_cible, angle_horiz):
        super().__init__()
        self.opti = opti
        self.att_x = att_x
        self.att_y = att_y
        self.waypoints = waypoints
        self.hauteur_net_cible = hauteur_net_cible
        self.angle_horiz = angle_horiz


    def run(self):
        try:
            # --- Optimisation ---
            vitesse_optimale, angle_optimal = self.opti.optimiser_trajectoire(
                distance_atterrissage_x=self.att_x,
                hauteur_net_cible=self.hauteur_net_cible,
                angle_horizontal_rad=self.angle_horiz,
                liste_points_2D=self.waypoints,
            )

            # --- Calcul trajectoire ---
            traj_x, traj_y, traj_z = BadmintonTrajectory.calculer_traj(
                vitesse_ini=vitesse_optimale,
                angle_vertical=angle_optimal,
                angle_horiz_rad=self.angle_horiz,
            )

            self.finished.emit(vitesse_optimale, angle_optimal, traj_x, traj_y, traj_z)

        except Exception as e:
            self.error.emit(str(e))


# -------- MainWindow --------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optimisation Trajectoire Badminton")
        self.resize(1400, 900)

        central = QWidget()
        main_layout = QHBoxLayout(central)
        self.setCentralWidget(central)

        # ---- Inputs ----
        left_panel = QWidget()
        form_layout = QFormLayout(left_panel)

        self.input_x = QLineEdit("7")
        self.input_y = QLineEdit("0")
        self.input_waypoints = QLineEdit("")
        self.input_net = QLineEdit("")

        self.btn_run = QPushButton("Optimiser Trajectoire")
        self.btn_run.clicked.connect(self.run_optimisation)
        self.input_x.textEdited.connect(self.run_optimisation)
        self.input_y.textEdited.connect(self.run_optimisation)

        self.label_result = QLabel("Résultats affichés ici")

        form_layout.addRow("X atterrissage (m):", self.input_x)
        form_layout.addRow("Y atterrissage (m):", self.input_y)
        form_layout.addRow("Waypoints [(x,y),...]:", self.input_waypoints)
        form_layout.addRow("Hauteur filet (m):", self.input_net)
        form_layout.addRow(self.btn_run)
        form_layout.addRow(self.label_result)

        main_layout.addWidget(left_panel, 1)

        # ---- Graph ----
        self.webview = QWebEngineView()
        main_layout.addWidget(self.webview, 3)
        html_init = """
        <html>
        <head><script src="https://cdn.plot.ly/plotly-latest.min.js"></script></head>
        <body><div id="plot" style="width:100%; height:100%;"></div>
        <script>Plotly.newPlot('plot', [], {margin:{t:0}}); window.camera_initialized=false;</script>
        </body>
        </html>
        """
        self.webview.setHtml(html_init)

        # Optimiseur
        self.opti = OptimiseurTrajectoire(device=None)
        self.thread = None

        # Pour déplacer avec WASD
        self.step = 0.1  # incrémentation par touche

    # ---------------- Gestion clavier ----------------
    def keyPressEvent(self, event):
        key = event.key()
        try:
            x = float(self.input_x.text())
            y = float(self.input_y.text())
        except ValueError:
            return

        if key == 87:  # W
            y += self.step
        elif key == 83:  # S
            y -= self.step
        elif key == 68:  # D
            x += self.step
        elif key == 65:  # A
            x -= self.step
        else:
            return

        self.input_x.setText(str(x))
        self.input_y.setText(str(y))

        # Relancer optimisation
        self.run_optimisation()

    # --- Garder ton run_optimisation, on_optimisation_done, on_error ---
    def run_optimisation(self):
        try:
            att_x = float(self.input_x.text())
            att_y = float(self.input_y.text())
        except ValueError:
            self.label_result.setText("Erreur: X et Y doivent être numériques")
            return

        try:
            waypoints = ast.literal_eval(self.input_waypoints.text()) if self.input_waypoints.text() else None
        except Exception:
            self.label_result.setText("Erreur: format des waypoints invalide")
            return

        try:
            hauteur_net_cible = float(self.input_net.text()) if self.input_net.text() else None
        except ValueError:
            self.label_result.setText("Erreur: hauteur filet invalide")
            return

        angle_horiz = np.arctan(att_y / (att_x + 1.98))

        # Lancer worker dans un thread
        self.thread = QThread()
        self.worker = OptimisationWorker(self.opti, att_x, att_y, waypoints, hauteur_net_cible, angle_horiz)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_optimisation_done)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()
        self.label_result.setText("Optimisation en cours...")

    def on_optimisation_done(self, vitesse, angle, traj_x, traj_y, traj_z):
        self.label_result.setText(f"Vitesse: {vitesse:.2f} m/s | Angle vertical: {angle:.2f}°")

        import json
        vis = BadmintonTrajectoryVisualizer()
        vis.afficher_graphique_interactif(traj_x, traj_y, traj_z)

        data_list = []
        for trace in vis.traces:
            trace_dict = trace.to_plotly_json()
            for key in ["x", "y", "z"]:
                if key in trace_dict:
                    trace_dict[key] = list(trace_dict[key])
            data_list.append(trace_dict)

        layout_dict = vis.layout.to_plotly_json()

        js = """
        (function() {
            var plot = document.getElementById('plot');
            var fig = %s;
            if(plot && window.camera_initialized) {
                fig.layout.scene.camera = plot.layout.scene.camera;
            }
            Plotly.react('plot', fig.data, fig.layout);
            window.camera_initialized = true;
        })();
        """ % json.dumps({'data': data_list, 'layout': layout_dict})

        self.webview.page().runJavaScript(js)

    def on_error(self, msg):
        self.label_result.setText(f"Erreur: {msg}")

    def closeEvent(self, event):
        self.opti.destroy()
        event.accept()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

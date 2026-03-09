import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFrame,
    QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QStackedWidget
)

from gui.screens.load_data import LoadDataScreen
from gui.screens.detect_bends_screen import DetectBendsScreen
from gui.screens.features_screen import FeaturesScreen
from gui.screens.cluster_screen import ClusterScreen
from gui.screens.visualize_results_screen import VisualizeResultsScreen


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("River Bends Clustering Tool")
        self.resize(1400, 820)

        self._ds = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        top = QFrame()
        top.setFixedHeight(70)
        top.setStyleSheet("background:#1f2937; color:white;")
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(18, 12, 18, 12)

        title = QLabel("River Bends Clustering Tool")
        title.setStyleSheet("font-size:18px; font-weight:700;")
        top_layout.addWidget(title)
        top_layout.addStretch(1)

        root.addWidget(top)

        mid = QFrame()
        mid_layout = QHBoxLayout(mid)
        mid_layout.setContentsMargins(0, 0, 0, 0)
        mid_layout.setSpacing(0)
        root.addWidget(mid, 1)

        # Sidebar
        sidebar = QFrame()
        sidebar.setFixedWidth(230)
        sidebar.setStyleSheet("background:#f1f5f9; border-right:1px solid #e2e8f0;")
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.setSpacing(8)

        def nav_btn(text: str) -> QPushButton:
            b = QPushButton(text)
            b.setCursor(Qt.PointingHandCursor)
            b.setMinimumHeight(44)
            b.setStyleSheet("""
                QPushButton {
                    text-align: left;
                    padding: 10px 12px;
                    border-radius: 10px;
                    background: transparent;
                    color: #0f172a;
                    font-weight: 600;
                }
                QPushButton:hover {
                    background: #e2e8f0;
                }
                QPushButton:disabled {
                    color: #94a3b8;
                }
                QPushButton[active="true"] {
                    background: #0f172a;
                    color: white;
                    font-weight: 800;
                }
                QPushButton[active="true"]:hover {
                    background: #0b1220;
                }
            """)
            return b

        # Buttons (with step numbers for visibility)
        self.btn_load = nav_btn("1. Load Data")
        self.btn_detect = nav_btn("2. Detect Bends")
        self.btn_features = nav_btn("3. Compute Features")
        self.btn_cluster = nav_btn("4. Run Clustering")
        self.btn_visualize = nav_btn("5. Visualize Results")
        # self.btn_export = nav_btn("6. Export Data")

        self._nav_buttons = [
            self.btn_load,
            self.btn_detect,
            self.btn_features,
            self.btn_cluster,
            self.btn_visualize,
            # self.btn_export,
        ]

        for b in self._nav_buttons:
            side_layout.addWidget(b)

        side_layout.addStretch(1)
        mid_layout.addWidget(sidebar)

        # Stacked screens
        self.stack = QStackedWidget()
        self.stack.setStyleSheet("background:white;")
        mid_layout.addWidget(self.stack, 1)

        # Screens
        self.load_screen = LoadDataScreen()
        self.detect_screen = DetectBendsScreen()
        self.features_screen = FeaturesScreen()
        self.cluster_screen = ClusterScreen()
        self.visualize_results_screen = VisualizeResultsScreen()

        self.stack.addWidget(self.load_screen)
        self.stack.addWidget(self.detect_screen)
        self.stack.addWidget(self.features_screen)
        self.stack.addWidget(self.cluster_screen)
        self.stack.addWidget(self.visualize_results_screen)
        # Initial screen
        self.stack.setCurrentWidget(self.load_screen)
        self.set_active(self.btn_load)

        for b in [self.btn_detect, self.btn_features, self.btn_cluster, self.btn_visualize]: # self.btn_export]:
            b.setEnabled(False)

        self.status_label = QLabel("No dataset loaded")
        self.statusBar().addWidget(self.status_label)

        self.btn_load.clicked.connect(self.go_load)
        self.btn_detect.clicked.connect(self.go_detect)
        self.btn_features.clicked.connect(self.go_features)
        self.btn_cluster.clicked.connect(self.go_cluster)
        self.btn_visualize.clicked.connect(self.go_visualize)

        # dataset loaded signal
        self.load_screen.dataset_loaded.connect(self.on_dataset_loaded)

    def go_load(self):
        self.stack.setCurrentWidget(self.load_screen)
        self.set_active(self.btn_load)

    def go_detect(self):
        if self._ds is None:
            return

        self.detect_screen.set_dataset(self._ds)

        self.stack.setCurrentWidget(self.detect_screen)
        self.set_active(self.btn_detect)

    def go_features(self):
        try:
            if hasattr(self.detect_screen, "get_last_csv_path") and hasattr(self.features_screen, "set_csv_path"):
                csv_path = self.detect_screen.get_last_csv_path()
                if csv_path is not None:
                    self.features_screen.set_csv_path(csv_path)
        except Exception:
            # Don't break navigation if anything goes wrong
            pass

        self.stack.setCurrentWidget(self.features_screen)
        self.set_active(self.btn_features)

    # Dataset loaded
    def on_dataset_loaded(self, ds):
        self._ds = ds

        for b in [self.btn_detect, self.btn_features, self.btn_cluster, self.btn_visualize]: #self.btn_export]:
            b.setEnabled(True)

        self.status_label.setText(
            f"Loaded: {ds.display_name} | CRS: {ds.crs} | Vertices: {ds.vertex_count}"
        )

    def set_active(self, active_button: QPushButton):
        for b in self._nav_buttons:
            b.setProperty("active", b is active_button)
            b.style().unpolish(b)
            b.style().polish(b)
            b.update()

    def go_cluster(self):
        # pull last detection result from detect screen
        x_s, y_s, bends_final = (None, None, None)
        if hasattr(self.detect_screen, "get_cluster_inputs"):
            x_s, y_s, bends_final = self.detect_screen.get_cluster_inputs()

        if x_s is None or y_s is None or bends_final is None:
            # still switch screens but it will show "Run Detect Bends first"
            pass
        else:
            self.cluster_screen.set_inputs(x_s, y_s, bends_final)

        self.stack.setCurrentWidget(self.cluster_screen)
        self.set_active(self.btn_cluster)

    def go_visualize(self):
        self.stack.setCurrentWidget(self.visualize_results_screen)
        self.set_active(self.btn_visualize)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

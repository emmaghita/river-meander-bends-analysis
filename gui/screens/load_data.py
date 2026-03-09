from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QGroupBox, QFormLayout, QMessageBox
)

from gui.widgets.mpl_canvas import MplCanvas
from app.io_loader import DatasetLoader, LoadedDataset


class LoadDataScreen(QWidget):
    dataset_loaded = Signal(object)  # emits LoadedDataset

    def __init__(self):
        super().__init__()
        self.loader = DatasetLoader()

        # Main layout
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        left_widget = QWidget()
        left_widget.setMinimumWidth(340)
        left_widget.setStyleSheet("""
            QWidget {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
            }
        """)

        left = QVBoxLayout(left_widget)
        left.setContentsMargins(14, 14, 14, 14)
        left.setSpacing(12)
        root.addWidget(left_widget, 0)

        self.btn_open = QPushButton("Open .shp or .zip")
        self.btn_open.setMinimumHeight(44)
        self.btn_open.setStyleSheet("""
            QPushButton {
                background: #e2e8f0;
                border: 1px solid #94a3b8;
                border-radius: 10px;
                padding: 10px;
                text-align: left;
                font-weight: 700;
                color: #0f172a;
            }
            QPushButton:hover {
                background: #cbd5e1;
            }
            QPushButton:pressed {
                background: #94a3b8;
            }
        """)
        self.btn_open.clicked.connect(self.on_open_file)
        left.addWidget(self.btn_open)

        meta = QGroupBox("Dataset metadata")
        meta.setStyleSheet("""
            QGroupBox {
                font-weight: 700;
                color: #0f172a;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                margin-top: 10px;
                padding: 10px;
                background: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QLabel {
                color: #0f172a;
            }
        """)
        form = QFormLayout(meta)
        form.setContentsMargins(10, 10, 10, 10)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)

        self.lbl_path = QLabel("-")
        self.lbl_crs = QLabel("-")
        self.lbl_len = QLabel("-")
        self.lbl_vertices = QLabel("-")

        self.lbl_path.setWordWrap(True)

        form.addRow("Path:", self.lbl_path)
        form.addRow("CRS:", self.lbl_crs)
        form.addRow("Length:", self.lbl_len)
        form.addRow("Vertices:", self.lbl_vertices)
        left.addWidget(meta)

        # Export plot button
        self.btn_export_plot = QPushButton("Export preview as PDF")
        self.btn_export_plot.setMinimumHeight(40)
        self.btn_export_plot.setEnabled(False)  # enabled only after plot exists

        self.btn_export_plot.setStyleSheet("""
            QPushButton {
                background: #e2e8f0;
                border: 1px solid #94a3b8;
                border-radius: 10px;
                padding: 10px;
                text-align: left;
                font-weight: 700;
                color: #0f172a;
            }
            QPushButton:hover {
                background: #cbd5e1;
            }
            QPushButton:pressed {
                background: #94a3b8;
            }
            QPushButton:disabled {
                background: #f1f5f9;
                color: #94a3b8;
                border: 1px solid #e2e8f0;
            }
        """)
        self.btn_export_plot.clicked.connect(self.on_export_plot)
        left.addWidget(self.btn_export_plot)



        hint = QLabel("Tip: ZIP should contain .shp + .shx + .dbf (+ .prj).")
        hint.setStyleSheet("color:#475569;")
        hint.setWordWrap(True)
        left.addWidget(hint)

        left.addStretch(1)

        self.canvas = MplCanvas(width=7, height=5, dpi=110)
        root.addWidget(self.canvas, 1)

        # Optional: make the plot area look cleaner initially
        self.canvas.ax.set_title("Preview: river centerline")
        self.canvas.ax.grid(True, alpha=0.25)
        self.canvas.draw()

    def on_open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select shapefile (.shp) or zipped shapefile (.zip)",
            "",
            "Shapefile (*.shp);;Zipped Shapefile (*.zip)"
        )
        if not path:
            return

        try:
            ds: LoadedDataset = self.loader.load_centerline(path)
            self._update_meta(ds)
            self._plot(ds.x, ds.y)

            self.btn_export_plot.setEnabled(True)
            self.dataset_loaded.emit(ds)
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))

    def _update_meta(self, ds: LoadedDataset):
        self.lbl_path.setText(ds.source_path)
        self.lbl_crs.setText(ds.crs)
        self.lbl_len.setText(f"{ds.length:.1f}")
        self.lbl_vertices.setText(str(ds.vertex_count))

    def _plot(self, x, y):
        self.canvas.ax.clear()
        self.canvas.ax.plot(x, y, linewidth=1.2)
        self.canvas.ax.set_title("River centerline")
        self.canvas.ax.set_aspect("equal", adjustable="datalim")
        self.canvas.ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def on_export_plot(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export preview as PDF",
            "centerline_preview.pdf",
            "PDF files (*.pdf)"
        )
        if not path:
            return

        try:
            self.canvas.figure.savefig(
                path,
                format="pdf",
                bbox_inches="tight"
            )

            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Exported")

            msg.setText("Centerline plot exported successfully.")
            msg.setInformativeText(path)

            msg.setStyleSheet("""
                QMessageBox {
                    background-color: white;
                }
                QLabel {
                    color: #0f172a;
                    font-size: 12px;
                }
                QPushButton {
                    min-width: 90px;
                    color: #0f172a;
                    background-color: #e2e8f0;
                    border: 1px solid #94a3b8;
                    border-radius: 6px;
                    padding: 6px 12px;
                    font-weight: 600;
                }
                QPushButton:hover {
                    background-color: #cbd5e1;
                }
            """)

            ok_btn = msg.addButton("OK", QMessageBox.AcceptRole)
            ok_btn.setDefault(True)

            msg.exec()

        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))

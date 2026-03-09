from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QFormLayout,
    QLabel, QComboBox, QCheckBox, QDoubleSpinBox, QPushButton,
    QFileDialog, QMessageBox, QSizePolicy
)

from gui.widgets.mpl_canvas import MplCanvas

FEATURE_COLUMNS = {
    "Amplitude (A_bend_m)": "A_bend_m",
    "Normalized amplitude (A_bend_star)": "A_bend_star",

    "Sinuosity (S)": "S",

    "Asymmetry ratio (AR)": "AR",
    "Upstream amplitude (A_up_m)": "A_up_m",
    "Downstream amplitude (A_down_m)": "A_down_m",

    "Openness (deg)": "openness",

    "Arc length (L_m)": "L_m",
    "Chord length (C_m)": "C_m",
    "Baseline length (baseline_m)": "baseline_m",

    "Number of arcs (n_arcs)": "n_arcs",
}


class FeaturesScreen(QWidget):
    def __init__(self):
        super().__init__()

        self._csv_path: Optional[Path] = None
        self._df = None  # pandas DataFrame

        root = QHBoxLayout(self)
        root.setContentsMargins(10, 12, 10, 12)
        root.setSpacing(12)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(12)
        left.setMinimumWidth(320)
        left.setMaximumWidth(380)

        self.btn_open = QPushButton("Open CSV…")
        self.btn_open.clicked.connect(self._choose_csv)

        self.box = QGroupBox("Feature histograms")
        form = QFormLayout(self.box)

        self.lbl_source = QLabel("-")
        self.cmb_feature = QComboBox()
        self.cmb_feature.addItems(list(FEATURE_COLUMNS.keys()))
        self.cmb_feature.currentIndexChanged.connect(self._refresh_plot)

        self.lbl_n = QLabel("-")
        self.lbl_median = QLabel("-")
        self.lbl_minmax = QLabel("-")

        self.chk_threshold = QCheckBox("Show threshold line")
        self.chk_threshold.stateChanged.connect(self._refresh_plot)

        self.spn_threshold = QDoubleSpinBox()
        self.spn_threshold.setDecimals(6)
        self.spn_threshold.setRange(-1e18, 1e18)
        self.spn_threshold.setValue(0.0)
        self.spn_threshold.valueChanged.connect(self._refresh_plot)

        form.addRow("CSV source:", self.lbl_source)
        form.addRow("Feature:", self.cmb_feature)
        form.addRow(self.chk_threshold)
        form.addRow("Threshold:", self.spn_threshold)
        form.addRow("N values:", self.lbl_n)
        form.addRow("Median:", self.lbl_median)
        form.addRow("Min / Max:", self.lbl_minmax)

        left_layout.addWidget(self.btn_open)
        left_layout.addWidget(self.box)
        left_layout.addStretch(1)

        root.addWidget(left, 1)

        # ---------- Right plot ----------
        self.canvas = MplCanvas(width=12, height=6, dpi=110)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.canvas, 10)

        self._init_plot_style()
        self._show_empty("Open a bends CSV to view feature histograms.")

        # Optional: simple style (black text, clean UI)
        self.setStyleSheet("""
            QWidget { color: #0f172a; }
            QLabel  { color: #0f172a; }
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
            QPushButton {
                background: #e2e8f0;
                border: 1px solid #94a3b8;
                border-radius: 10px;
                padding: 10px;
                text-align: left;
                font-weight: 700;
                color: #0f172a;
                min-height: 36px;
            }
            QPushButton:hover { background: #cbd5e1; }
        """)

    def _choose_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open bends CSV",
            "outputs",
            "CSV files (*.csv)"
        )
        if not path:
            return

        self.set_csv_path(Path(path))

    def set_csv_path(self, csv_path: Path):
        self._csv_path = csv_path
        self.lbl_source.setText(csv_path.as_posix())
        self._load_csv()

    def _load_csv(self):
        if not self._csv_path or not self._csv_path.exists():
            self._show_empty("CSV not found.")
            return

        try:
            import pandas as pd
            self._df = pd.read_csv(self._csv_path)
        except Exception as e:
            QMessageBox.critical(self, "Failed to load CSV", str(e))
            self._df = None
            return

        self._refresh_plot()

    def _refresh_plot(self):
        if self._df is None:
            self._show_empty("Open a bends CSV to view histograms.")
            return

        feat_label = self.cmb_feature.currentText()
        col = FEATURE_COLUMNS[feat_label]

        if col not in self._df.columns:
            self._show_empty(f"Missing column '{col}' in CSV.")
            return

        vals = self._df[col].to_numpy()
        vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            self._show_empty(f"No finite values in '{col}'.")
            return

        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        med = float(np.median(vals))

        self.lbl_n.setText(str(vals.size))
        self.lbl_median.setText(f"{med:.6g}")
        self.lbl_minmax.setText(f"{vmin:.6g} / {vmax:.6g}")

        ax = self.canvas.ax
        ax.clear()
        self._init_plot_style()

        # histogram
        ax.hist(vals, bins="auto", edgecolor="black", alpha=0.75)

        # median line
        ax.axvline(med, linestyle="--", linewidth=2, label=f"Median = {med:.6g}")

        # threshold line
        if self.chk_threshold.isChecked():
            t = float(self.spn_threshold.value())
            ax.axvline(t, linestyle=":", linewidth=2, label=f"Threshold = {t:.6g}")

        ax.set_title(f"Histogram: {feat_label}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.legend()

        self.canvas.fig.tight_layout(pad=1.2)
        self.canvas.draw()

    def _show_empty(self, msg: str):
        self.lbl_n.setText("-")
        self.lbl_median.setText("-")
        self.lbl_minmax.setText("-")
        ax = self.canvas.ax
        ax.clear()
        self._init_plot_style()
        ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
        self.canvas.draw()

    def _init_plot_style(self):
        ax = self.canvas.ax
        self.canvas.fig.set_facecolor("white")
        ax.set_facecolor("white")
        ax.grid(True, alpha=0.3)

import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame,
    QPushButton, QLabel, QFileDialog, QComboBox, QMessageBox,
    QTextEdit, QSplitter
)

from gui.widgets.mpl_canvas import MplCanvas


FEATURES = [
    ("Amplitude (A_bend_m)", "A_bend_m"),
    ("Sinuosity (S)", "S"),
    ("Asymmetry ratio (AR)", "AR"),
    ("Openness (deg)", "openness"),
]

# Placeholder descriptions (you said we'll refine later)
CLUSTER_DESCRIPTIONS = {
    0: "Baseline bends with moderate geometry (often the most common cluster).",
    1: "Tighter / more compact bends compared to baseline.",
    2: "More asymmetric bends (upstream vs downstream imbalance).",
    3: "More open / wide bends (larger openness angles).",
    4: "Higher sinuosity / stronger curvature tendencies.",
    5: "Mixed / transitional bends (properties overlap others).",
}


class VisualizeResultsScreen(QWidget):

    def __init__(self):
        super().__init__()

        self.df = None
        self.csv_path = None

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        header = QFrame()
        header.setObjectName("vr_header")
        header_layout = QGridLayout(header)
        header_layout.setContentsMargins(14, 12, 14, 12)
        header_layout.setHorizontalSpacing(10)
        header_layout.setVerticalSpacing(8)

        title = QLabel("Visualize results (from clustered CSV)")
        title.setObjectName("vr_title")
        header_layout.addWidget(title, 0, 0, 1, 3)

        self.lbl_file = QLabel("No file loaded")
        self.lbl_file.setTextInteractionFlags(Qt.TextSelectableByMouse)
        header_layout.addWidget(QLabel("CSV:"), 1, 0)
        header_layout.addWidget(self.lbl_file, 1, 1, 1, 2)

        # Buttons + Cluster selector row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.btn_load = QPushButton("Load clustered CSV…")
        self.btn_load.clicked.connect(self.on_load_csv)
        btn_row.addWidget(self.btn_load)

        self.btn_refresh = QPushButton("Refresh plots")
        self.btn_refresh.clicked.connect(self.refresh_plots)
        self.btn_refresh.setEnabled(False)
        btn_row.addWidget(self.btn_refresh)

        self.btn_export = QPushButton("Export histograms (PDF)…")
        self.btn_export.clicked.connect(self.export_histograms_pdf)
        self.btn_export.setEnabled(False)
        btn_row.addWidget(self.btn_export)

        btn_row.addStretch(1)

        btn_row.addWidget(QLabel("Cluster:"))
        self.cluster_combo = QComboBox()
        self.cluster_combo.currentIndexChanged.connect(self.on_cluster_changed)
        self.cluster_combo.setEnabled(False)
        self.cluster_combo.setMinimumWidth(160)
        btn_row.addWidget(self.cluster_combo)

        header_layout.addLayout(btn_row, 2, 0, 1, 3)

        root.addWidget(header)

        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)

        plots_panel = QFrame()
        plots_panel.setObjectName("vr_plots")
        plots_layout = QGridLayout(plots_panel)
        plots_layout.setContentsMargins(12, 12, 12, 12)
        plots_layout.setHorizontalSpacing(10)
        plots_layout.setVerticalSpacing(10)

        self.canv_amp = MplCanvas(width=6, height=3.5, dpi=100)
        self.canv_sin = MplCanvas(width=6, height=3.5, dpi=100)
        self.canv_ar = MplCanvas(width=6, height=3.5, dpi=100)
        self.canv_open = MplCanvas(width=6, height=3.5, dpi=100)

        plots_layout.addWidget(self.canv_amp, 0, 0)
        plots_layout.addWidget(self.canv_sin, 0, 1)
        plots_layout.addWidget(self.canv_ar,  1, 0)
        plots_layout.addWidget(self.canv_open,1, 1)

        plots_layout.setRowStretch(0, 1)
        plots_layout.setRowStretch(1, 1)
        plots_layout.setColumnStretch(0, 1)
        plots_layout.setColumnStretch(1, 1)

        split.addWidget(plots_panel)

        summary = QFrame()
        summary.setObjectName("vr_summary")
        summary.setMinimumWidth(340)
        summary.setMaximumWidth(440)
        summary_layout = QVBoxLayout(summary)
        summary_layout.setContentsMargins(12, 12, 12, 12)
        summary_layout.setSpacing(10)

        lbl_sum = QLabel("Cluster summary")
        lbl_sum.setObjectName("vr_subtitle")
        summary_layout.addWidget(lbl_sum)

        self.desc_label = QLabel("Load a CSV to view cluster description.")
        self.desc_label.setWordWrap(True)
        summary_layout.addWidget(self.desc_label)

        self.stats_box = QTextEdit()
        self.stats_box.setReadOnly(True)
        self.stats_box.setText("Stats will appear here.")
        summary_layout.addWidget(self.stats_box, 1)

        split.addWidget(summary)

        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)

        root.addWidget(split, 1)

        self.setStyleSheet("""
        
        QLabel {
            color: #0f172a;
        }

        QPushButton {
            color: #0f172a;
            font-weight: 600;
        }

        QComboBox {
            color: #0f172a;
        }

        QFrame#vr_header {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
        }

        QLabel#vr_title {
            font-size: 16px;
            font-weight: 800;
        }

        
        QFrame#vr_plots,
        QFrame#vr_summary {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
        }

        QLabel#vr_subtitle {
            font-size: 14px;
            font-weight: 800;
        }

        QPushButton {
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 10px;
            padding: 6px 14px;
        }

        QPushButton:hover {
            background: #f1f5f9;
            border-color: #94a3b8;
        }

        QPushButton:pressed {
            background: #e2e8f0;
        }

        QPushButton:disabled {
            background: #f8fafc;
            border-color: #e2e8f0;
            color: #64748b;
        }

        QComboBox {
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 10px;
            padding: 4px 10px;
            font-weight: 600;
        }

        QComboBox:hover {
            border-color: #94a3b8;
        }

        QComboBox:disabled {
            background: #f8fafc;
            border-color: #e2e8f0;
            color: #64748b;
        }

        QComboBox QAbstractItemView {
            background: #ffffff;
            color: #0f172a;
            selection-background-color: #e2e8f0;
            selection-color: #0f172a;
            border: 1px solid #cbd5e1;
        }

        QTextEdit {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 10px;
            font-family: Consolas, monospace;
            color: #0f172a;
        }

        QPushButton:disabled,
        QComboBox:disabled {
            color: #64748b;
        }
        """)

    def on_load_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select clustered bends CSV",
            "",
            "CSV files (*.csv);;All files (*.*)"
        )
        if not path:
            return

        self.load_csv_path(path)

    def load_csv_path(self, path: str):
        try:
            df = pd.read_csv(path)
        except Exception as e:
            QMessageBox.critical(self, "Load error", f"Failed to read CSV:\n{e}")
            return

        # Validate required columns
        missing = []
        if "cluster" not in df.columns:
            missing.append("cluster")
        for _, col in FEATURES:
            if col not in df.columns:
                missing.append(col)

        if missing:
            QMessageBox.critical(
                self,
                "Invalid CSV",
                "CSV is missing required columns:\n" + ", ".join(missing)
            )
            return

        # Clean types
        df = df.copy()
        df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce")
        df = df.dropna(subset=["cluster"])
        df["cluster"] = df["cluster"].astype(int)

        for _, col in FEATURES:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        self.df = df
        self.csv_path = path
        self.lbl_file.setText(path)

        self.populate_clusters()

        self.btn_refresh.setEnabled(True)
        self.btn_export.setEnabled(True)

        if self.cluster_combo.count() > 0:
            self.cluster_combo.setCurrentIndex(0)
            self.update_description()
            self.update_stats()
            self.refresh_plots()

    def populate_clusters(self):
        self.cluster_combo.blockSignals(True)
        self.cluster_combo.clear()

        clusters = sorted(self.df["cluster"].unique().tolist())
        for k in clusters:
            self.cluster_combo.addItem(f"Cluster {k}", k)

        self.cluster_combo.setEnabled(True)
        self.cluster_combo.blockSignals(False)

    def on_cluster_changed(self):
        if self.df is None or self.cluster_combo.count() == 0:
            return
        self.update_description()
        self.update_stats()
        self.refresh_plots()

    def current_cluster(self) -> int:
        return int(self.cluster_combo.currentData())

    def update_description(self):
        k = self.current_cluster()
        desc = CLUSTER_DESCRIPTIONS.get(
            k,
            "No hardcoded description for this cluster yet. You can add one in CLUSTER_DESCRIPTIONS."
        )
        n = int((self.df["cluster"] == k).sum())
        self.desc_label.setText(f"{desc}\n\nBends in cluster: {n}")

    def update_stats(self):
        k = self.current_cluster()
        df_k = self.df[self.df["cluster"] == k]

        lines = [f"n = {len(df_k)}"]
        for _, col in FEATURES:
            s = pd.to_numeric(df_k[col], errors="coerce").replace([float("inf"), float("-inf")], pd.NA).dropna()
            if len(s) == 0:
                lines.append(f"{col}: no valid values")
            else:
                lines.append(
                    f"{col}: mean={s.mean():.3f}, std={s.std():.3f}, "
                    f"min={s.min():.3f}, max={s.max():.3f}"
                )

        self.stats_box.setText("\n".join(lines))

    def refresh_plots(self):
        if self.df is None or self.cluster_combo.count() == 0:
            return

        k = self.current_cluster()
        df_k = self.df[self.df["cluster"] == k]

        self._plot_hist(self.canv_amp, df_k["A_bend_m"], "Amplitude (A_bend_m)", "A_bend_m (m)")
        self._plot_hist(self.canv_sin, df_k["S"], "Sinuosity (S)", "S")
        self._plot_hist(self.canv_ar, df_k["AR"], "Asymmetry ratio (AR)", "AR")
        self._plot_hist(self.canv_open, df_k["openness"], "Openness (deg)", "openness (deg)")

    def _plot_hist(self, canvas: MplCanvas, series, title: str, xlabel: str):
        # Your MplCanvas provides: canvas.ax and canvas.fig
        ax = canvas.ax
        ax.clear()

        s = pd.to_numeric(series, errors="coerce").replace([float("inf"), float("-inf")], pd.NA).dropna()

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")

        if len(s) == 0:
            ax.text(0.5, 0.5, "No valid values", ha="center", va="center", transform=ax.transAxes)
            canvas.fig.tight_layout()
            canvas.draw()
            return

        ax.hist(s.values, bins=20)
        ax.grid(True, alpha=0.25)

        # Force readable matplotlib text
        ax.tick_params(colors="black")
        ax.title.set_color("black")
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")

        canvas.fig.tight_layout()
        canvas.draw()

    def export_histograms_pdf(self):
        if self.df is None or self.cluster_combo.count() == 0:
            return

        k = self.current_cluster()
        df_k = self.df[self.df["cluster"] == k]

        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export histograms to PDF",
            f"cluster_{k}_histograms.pdf",
            "PDF (*.pdf)"
        )
        if not out_path:
            return

        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            axes = axes.ravel()

            cols = [
                ("A_bend_m", "Amplitude (m)"),
                ("S", "Sinuosity"),
                ("AR", "Asymmetry ratio"),
                ("openness", "Openness (deg)"),
            ]

            for ax, (col, xlabel) in zip(axes, cols):
                s = pd.to_numeric(df_k[col], errors="coerce").replace([float("inf"), float("-inf")], pd.NA).dropna()
                ax.hist(s.values, bins=20)
                ax.set_title(col)
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Count")
                ax.grid(True, alpha=0.25)

            fig.suptitle(f"Cluster {k} metric distributions", y=0.98)
            fig.tight_layout()
            fig.savefig(out_path)
            plt.close(fig)

            QMessageBox.information(self, "Export complete", f"Saved:\n{out_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export error", f"Failed to export PDF:\n{e}")

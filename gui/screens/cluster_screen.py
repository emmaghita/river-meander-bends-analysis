from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QFormLayout, QLabel,
    QPushButton, QFileDialog, QMessageBox, QSizePolicy
)
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D

from gui.widgets.mpl_canvas import MplCanvas
from cluster_bends import run_clustering, draw_separator_tick


class ClusterScreen(QWidget):
    def __init__(self):
        super().__init__()

        self._x_s: Optional[np.ndarray] = None
        self._y_s: Optional[np.ndarray] = None
        self._bends_final: Optional[List[Dict[str, Any]]] = None
        self._last_k: Optional[int] = None

        root = QHBoxLayout(self)
        root.setContentsMargins(8, 12, 8, 12)
        root.setSpacing(12)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(12)

        left_widget.setMinimumWidth(300)
        left_widget.setMaximumWidth(360)
        root.addWidget(left_widget, 1)

        self.box_summary = QGroupBox("Clustering summary")
        form = QFormLayout(self.box_summary)

        self.lbl_status = QLabel("-")
        self.lbl_k = QLabel("-")
        self.lbl_n_bends = QLabel("-")

        form.addRow("Status:", self.lbl_status)
        form.addRow("Chosen k:", self.lbl_k)
        form.addRow("Bends:", self.lbl_n_bends)

        left_layout.addWidget(self.box_summary)

        controls = QHBoxLayout()
        self.btn_run = QPushButton("Run clustering")
        self.btn_export = QPushButton("Export plot")
        self.btn_export.setEnabled(False)

        self.btn_run.clicked.connect(self.on_run)
        self.btn_export.clicked.connect(self.on_export)

        controls.addWidget(self.btn_run)
        controls.addWidget(self.btn_export)
        left_layout.addLayout(controls)

        self.lbl_hint = QLabel("Tip: Run clustering after Detect Bends. Export saves PDF/PNG/SVG.")
        self.lbl_hint.setWordWrap(True)
        left_layout.addWidget(self.lbl_hint)

        left_layout.addStretch(1)

        self.canvas = MplCanvas(width=12, height=6, dpi=110)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.canvas, 10)
        root.setStretch(0, 1)
        root.setStretch(1, 10)

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
            QPushButton:pressed { background: #94a3b8; }
            QPushButton:disabled {
                background: #f1f5f9;
                color: #94a3b8;
                border: 1px solid #e2e8f0;
            }
        """)

        self._init_plot_style()
        self._set_enabled(False)

    def set_inputs(self, x_s, y_s, bends_final):
        self._x_s = np.asarray(x_s, dtype=float)
        self._y_s = np.asarray(y_s, dtype=float)
        self._bends_final = list(bends_final) if bends_final is not None else None
        self._set_enabled(self._bends_final is not None and len(self._bends_final) > 0)

        if self._bends_final is not None:
            self.lbl_status.setText("Ready")
            self.lbl_n_bends.setText(str(len(self._bends_final)))
        else:
            self.lbl_status.setText("No bends available")
            self.lbl_n_bends.setText("-")

    def _init_plot_style(self):
        ax = self.canvas.ax
        ax.set_title("Clustered bends")
        ax.set_aspect("auto")
        ax.grid(True, color="#cbd5e1", alpha=0.6)
        for sp in ax.spines.values():
            sp.set_color("#cbd5e1")
        ax.tick_params(colors="#475569")

        ax.ticklabel_format(style="plain", useOffset=False, axis="both")
        fmt = ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)

        ax.set_facecolor("white")
        self.canvas.figure.set_facecolor("white")

    def on_run(self):
        if self._x_s is None or self._y_s is None or not self._bends_final:
            QMessageBox.warning(self, "Missing data", "Run Detect Bends first.")
            return

        try:
            self.lbl_status.setText("Running...")
            self.btn_export.setEnabled(False)

            # Compute clusters (no plotting inside, no show)
            res = run_clustering(
                bends_csv="outputs/bends_table.csv",
                auto_k=True,
                save_csv=False
            )

            self._last_k = res.best_k
            self.lbl_k.setText(str(res.best_k))
            self.lbl_status.setText("Done")

            # Draw into existing canvas
            self._plot_cluster_map(self._x_s, self._y_s, self._bends_final, res.df)

            self.btn_export.setEnabled(True)

        except Exception as e:
            self.lbl_status.setText("Failed")
            QMessageBox.critical(self, "Clustering failed", str(e))

    def _plot_cluster_map(self, x_s, y_s, bends_final, clusters_df):
        ax = self.canvas.ax
        ax.clear()
        self._init_plot_style()

        # Build (i0,i1)->cluster lookup from df
        cluster_map = {
            (int(r.i0), int(r.i1)): int(r.cluster)
            for r in clusters_df.itertuples(index=False)
        }

        bends = []
        for b in bends_final:
            b2 = dict(b)
            b2["cluster"] = cluster_map.get((int(b2["i0"]), int(b2["i1"])), -1)
            bends.append(b2)

        # clusters set (ignore -1)
        clusters = sorted({int(b["cluster"]) for b in bends if int(b["cluster"]) >= 0})
        cmap = self.canvas.figure.get_cmap() if hasattr(self.canvas.figure, "get_cmap") else None
        cmap = __import__("matplotlib").pyplot.get_cmap("tab10", max(len(clusters), 1))

        # base river
        ax.plot(x_s, y_s, linewidth=1.0, alpha=0.15)

        # plot segments
        for b in bends:
            i0, i1 = int(b["i0"]), int(b["i1"])
            cl = int(b["cluster"])
            if i1 <= i0:
                continue

            if cl < 0:
                ax.plot(x_s[i0:i1 + 1], y_s[i0:i1 + 1], color="gray", linewidth=2.0, alpha=0.35)
            else:
                ci = clusters.index(cl)
                ax.plot(x_s[i0:i1 + 1], y_s[i0:i1 + 1], color=cmap(ci), linewidth=2.8, alpha=0.9)

        # separator ticks (optional but matches your style)
        for b in bends:
            draw_separator_tick(ax, x_s, y_s, int(b["i1"]))

        # nice bounds
        xmin, xmax = float(x_s.min()), float(x_s.max())
        ymin, ymax = float(y_s.min()), float(y_s.max())
        dx = xmax - xmin
        dy = ymax - ymin
        pad_x = 0.08 * dx if dx > 0 else 1.0
        pad_y = 0.08 * dy if dy > 0 else 1.0
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)

        # legend
        handles = [
            Line2D([0], [0], color=cmap(i), lw=3, label=f"Cluster {cl}")
            for i, cl in enumerate(clusters)
        ]
        handles.append(Line2D([0], [0], color="gray", lw=3, alpha=0.35, label="Unclustered"))
        ax.legend(handles=handles, framealpha=0.9, loc="upper right")

        k_txt = f"(k={self._last_k})" if self._last_k is not None else ""
        ax.set_title(f"Final bends colored by KMeans cluster {k_txt}".strip())

        self.canvas.draw()

    def on_export(self):
        if self._last_k is None:
            QMessageBox.warning(self, "Nothing to export", "Run clustering first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export clustered map",
            f"clustered_bends_k{self._last_k}.pdf",
            "PDF (*.pdf);;PNG (*.png);;SVG (*.svg)"
        )
        if not path:
            return

        try:
            self.canvas.figure.savefig(path, dpi=300, bbox_inches="tight")
            QMessageBox.information(self, "Exported", f"Saved:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))

    def _set_enabled(self, enabled: bool):
        self.btn_run.setEnabled(enabled)
        self.btn_export.setEnabled(False)

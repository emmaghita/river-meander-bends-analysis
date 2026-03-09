from __future__ import annotations

from typing import Any, Dict, Optional, List
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout, QLabel,
    QPushButton, QFileDialog, QMessageBox, QComboBox, QSizePolicy
)
from matplotlib.ticker import ScalarFormatter

from gui.widgets.mpl_canvas import MplCanvas
from app.pipeline import (
    run_bend_pipeline_from_dataset,
    PipelineParams,
    export_bend_table_csv,
)


class DetectBendsScreen(QWidget):
    def __init__(self):
        super().__init__()

        self._ds = None
        self._result: Optional[Dict[str, Any]] = None
        self._bend_table: List[Dict[str, Any]] = []

        root = QHBoxLayout(self)
        # slightly smaller margins to free width for the plot
        root.setContentsMargins(8, 12, 8, 12)
        root.setSpacing(12)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(12)

        # Make the left panel narrower so plot is wider
        left_widget.setMinimumWidth(300)
        left_widget.setMaximumWidth(360)

        root.addWidget(left_widget, 1)

        self.box_summary = QGroupBox("Detection summary")
        form = QFormLayout(self.box_summary)

        self.lbl_spacing = QLabel("-")
        self.lbl_vertices = QLabel("-")
        self.lbl_inflections = QLabel("-")
        self.lbl_segments = QLabel("-")
        self.lbl_arcs = QLabel("-")
        self.lbl_bends = QLabel("-")
        self.lbl_simple_compound = QLabel("-")

        self.lbl_Astar = QLabel("-")
        self.lbl_S = QLabel("-")
        self.lbl_AR = QLabel("-")

        form.addRow("Spacing (m):", self.lbl_spacing)
        form.addRow("Resampled vertices:", self.lbl_vertices)
        form.addRow("Inflections (raw→filtered):", self.lbl_inflections)
        form.addRow("Segments after merge:", self.lbl_segments)
        form.addRow("Arc / Straight:", self.lbl_arcs)
        form.addRow("Bends mapped → final:", self.lbl_bends)
        form.addRow("Simple / Compound:", self.lbl_simple_compound)
        form.addRow("A_bend* (min/med/max):", self.lbl_Astar)
        form.addRow("S (min/med/max):", self.lbl_S)
        form.addRow("AR (min/med/max):", self.lbl_AR)

        left_layout.addWidget(self.box_summary)

        controls = QHBoxLayout()
        self.btn_rerun = QPushButton("Re-run")
        self.btn_export_plot = QPushButton("Export plot PDF")

        self.btn_rerun.clicked.connect(self.on_rerun)
        self.btn_export_plot.clicked.connect(self.on_export_plot_pdf)

        self.cmb_color = QComboBox()
        self.cmb_color.addItems(["Color: sign", "Color: compound"])
        self.cmb_color.currentIndexChanged.connect(self.on_color_mode_changed)

        controls.addWidget(self.btn_rerun)
        controls.addWidget(self.btn_export_plot)
        controls.addStretch(1)
        controls.addWidget(self.cmb_color)
        left_layout.addLayout(controls)

        self.lbl_autosave = QLabel("CSV (auto): -")
        self.lbl_autosave.setStyleSheet("color: #0f172a; padding: 4px 2px;")
        left_layout.addWidget(self.lbl_autosave)

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

            QComboBox {
                background: white;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                padding: 6px 10px;
                color: #0f172a;
                min-height: 32px;
            }
        """)

        self._init_plot_style()
        self._set_enabled(False)

    def set_dataset(self, ds):
        self._ds = ds
        self.run_pipeline(auto=True)

    def get_last_result(self):
        return self._result

    def get_cluster_inputs(self):

        if not self._result:
            return None, None, None
        x_s = self._result["series"]["x_s"]
        y_s = self._result["series"]["y_s"]
        bends_final = self._result["bends"]["bends_final"]
        return x_s, y_s, bends_final

    def run_pipeline(self, auto: bool = False):
        if self._ds is None:
            return

        try:
            params = PipelineParams(wc=40.0)

            self._result = run_bend_pipeline_from_dataset(self._ds, params=params)
            self._bend_table = self._result["bend_table"]

            self._update_summary(self._result["diagnostics"])
            self._plot_result(self._result)

            out_csv = self._autosave_csv()
            self.lbl_autosave.setText(f"CSV (auto): {out_csv.as_posix()}")

            self._set_enabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Detect bends failed", str(e))

    def _autosave_csv(self) -> Path:
        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)

        ds_name = getattr(self._ds, "name", None) or getattr(self._ds, "stem", None) or "bends"
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(ds_name))
        out_csv = out_dir / f"{safe}_table.csv"

        export_bend_table_csv(self._bend_table, str(out_csv))
        return out_csv

    def _update_summary(self, diag: Dict[str, Any]):
        c = diag["counts"]
        s = diag["stats"]

        self.lbl_spacing.setText(f'{c["spacing_m"]:.1f}')
        self.lbl_vertices.setText(str(c["resampled_vertices"]))
        self.lbl_inflections.setText(f'{c["raw_inflections"]} → {c["filtered_inflections"]}')
        self.lbl_segments.setText(str(c["segments_after_merge"]))
        self.lbl_arcs.setText(f'{c["n_arc"]} / {c["n_straight"]}  (frac={c["arc_fraction"]:.3f})')
        self.lbl_bends.setText(f'{c["bends_mapped"]} → {c["bends_final"]}')
        self.lbl_simple_compound.setText(f'{c["n_simple"]} / {c["n_compound"]}')

        self.lbl_Astar.setText(self._fmt_min_med_max(s["A_bend_star"]))
        self.lbl_S.setText(self._fmt_min_med_max(s["S"]))
        self.lbl_AR.setText(self._fmt_min_med_max(s["AR"]))

    @staticmethod
    def _fmt_min_med_max(d: Dict[str, float]) -> str:
        return f'{d["min"]:.3g} / {d["median"]:.3g} / {d["max"]:.3g}'

    def _init_plot_style(self):
        ax = self.canvas.ax
        ax.set_title("Detected bends (Limaye-style)")
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

    def _plot_result(self, result: Dict[str, Any]):
        ax = self.canvas.ax
        ax.clear()
        self._init_plot_style()

        x_s = result["series"]["x_s"]
        y_s = result["series"]["y_s"]
        bends = result["bends"]["bends_final"]

        ax.plot(x_s, y_s, linewidth=1.1, color="#334155", alpha=0.35)

        mode = self.cmb_color.currentIndex()
        for b in bends:
            i0, i1 = int(b["i0"]), int(b["i1"])
            if mode == 0:
                col = "#2563eb" if b["sign"] > 0 else "#f97316"
            else:
                col = "#7c3aed" if b.get("is_compound", False) else "#0ea5e9"

            ax.plot(x_s[i0:i1 + 1], y_s[i0:i1 + 1], linewidth=2.4, color=col, alpha=0.95)

        xmin, xmax = float(x_s.min()), float(x_s.max())
        ymin, ymax = float(y_s.min()), float(y_s.max())
        dx = xmax - xmin
        dy = ymax - ymin
        pad_x = 0.08 * dx if dx > 0 else 1.0
        pad_y = 0.08 * dy if dy > 0 else 1.0
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)

        self.canvas.figure.tight_layout(pad=1.2)
        self.canvas.draw()

    def on_rerun(self):
        self.run_pipeline(auto=False)

    def on_color_mode_changed(self):
        if self._result:
            self._plot_result(self._result)

    def on_export_plot_pdf(self):
        if not self._result:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export plot as PDF", "detected_bends.pdf", "PDF files (*.pdf)"
        )
        if not path:
            return
        self.canvas.figure.savefig(path, format="pdf", bbox_inches="tight")

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Exported")
        msg.setText("Plot exported successfully.")
        msg.setInformativeText(path)
        msg.exec()

    def _set_enabled(self, enabled: bool):
        self.btn_rerun.setEnabled(enabled)
        self.btn_export_plot.setEnabled(enabled)
        self.cmb_color.setEnabled(enabled)

"""PySide6 desktop interface for the shared Extract Stills engine."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import threading
from typing import Any


_QT_DLL_HANDLES: list[Any] = []


def _prepare_qt_runtime() -> None:
    """Keep a packaged app independent of other applications' Qt runtimes."""
    if not getattr(sys, "frozen", False):
        return
    for key in ("QT_PLUGIN_PATH", "QML2_IMPORT_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH"):
        os.environ.pop(key, None)
    bundle = getattr(sys, "_MEIPASS", "")
    directories = [bundle, os.path.join(bundle, "PySide6"), os.path.join(bundle, "shiboken6")]
    directories = [directory for directory in directories if os.path.isdir(directory)]
    os.environ["PATH"] = os.pathsep.join(directories + [os.environ.get("PATH", "")])
    if hasattr(os, "add_dll_directory"):
        for directory in directories:
            _QT_DLL_HANDLES.append(os.add_dll_directory(directory))


_prepare_qt_runtime()

from PySide6.QtCore import Qt, QSettings, QThread, Signal, QUrl, QSize, QPoint
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QPainter, QPixmap, QPen, QPalette, QPolygon
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFrame,
    QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QMainWindow,
    QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSizePolicy,
    QSpinBox, QToolButton, QVBoxLayout, QWidget,
)

from .core import (
    CancelledError, analyze_video, discover_inputs, export_analysis, extract_legacy,
)
from .types import Options


THEMES = {
    "light": {
        "top": "#dce7f1", "bottom": "#f0f3f7", "card": "#ffffff",
        "soft": "#f3f6fa", "text": "#202b39", "muted": "#68778a",
        "border": "#dfe6ee", "accent": "#24354a", "accent_text": "#ffffff",
        "green": "#16794e", "green_bg": "#e6f5ed", "amber": "#90610a",
        "amber_bg": "#fff3d7", "red": "#b73f47", "red_bg": "#fcecef",
        "preview": "#e7edf4",
    },
    "dark": {
        "top": "#101a2c", "bottom": "#18253b", "card": "#1c2c45",
        "soft": "#233650", "text": "#edf3fa", "muted": "#a4b2c5",
        "border": "#344761", "accent": "#ecf3fb", "accent_text": "#17263b",
        "green": "#80e3b4", "green_bg": "#193f37", "amber": "#ffd48a",
        "amber_bg": "#483d2b", "red": "#ffa0a8", "red_bg": "#482d3d",
        "preview": "#142037",
    },
}


def build_stylesheet(theme: dict[str, str]) -> str:
    return f"""
        QWidget {{
            color: {theme['text']};
            font-family: 'Segoe UI', '.AppleSystemUIFont', 'Helvetica Neue', Arial;
            font-size: 13px;
        }}
        QLabel, QFrame, QScrollArea, QScrollArea > QWidget > QWidget {{
            background: transparent;
        }}
        QWidget#Root {{
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 {theme['top']}, stop:1 {theme['bottom']});
        }}
        QFrame#Card {{
            background: {theme['card']}; border: 1px solid {theme['border']};
            border-radius: 17px;
        }}
        QFrame#Candidate {{
            background: {theme['soft']}; border: 1px solid {theme['border']};
            border-radius: 12px;
        }}
        QFrame#Candidate[selected="true"] {{ border: 2px solid {theme['green']}; }}
        QFrame#DropZone {{
            background: {theme['card']}; border: 2px dashed {theme['border']};
            border-radius: 17px;
        }}
        QLabel#Title {{ font-size: 28px; font-weight: 700; }}
        QLabel#Subtitle, QLabel#Muted {{ color: {theme['muted']}; }}
        QLabel#Section {{ color: {theme['muted']}; font-size: 11px; font-weight: 600; }}
        QLabel#CardTitle {{ font-size: 16px; font-weight: 600; }}
        QLabel#Warning {{ color: {theme['amber']}; }}
        QLabel#Error {{ color: {theme['red']}; }}
        QLabel#Success {{ color: {theme['green']}; }}
        QLabel#High {{
            color: {theme['green']}; background: {theme['green_bg']};
            border-radius: 8px; padding: 3px 8px; font-size: 11px; font-weight: 600;
        }}
        QLabel#Mid {{
            color: {theme['amber']}; background: {theme['amber_bg']};
            border-radius: 8px; padding: 3px 8px; font-size: 11px; font-weight: 600;
        }}
        QLabel#Low {{
            color: {theme['red']}; background: {theme['red_bg']};
            border-radius: 8px; padding: 3px 8px; font-size: 11px; font-weight: 600;
        }}
        QToolButton, QPushButton {{
            background: {theme['soft']}; border: 1px solid {theme['border']};
            border-radius: 10px; padding: 8px 13px; font-weight: 600;
        }}
        QToolButton:hover, QPushButton:hover {{ border-color: {theme['muted']}; }}
        QToolButton#Primary, QPushButton#Primary {{
            background: {theme['accent']}; color: {theme['accent_text']};
            border: 1px solid {theme['accent']};
        }}
        QToolButton:disabled, QPushButton:disabled {{
            color: {theme['muted']}; background: {theme['soft']};
            border-color: {theme['border']};
        }}
        QToolButton#Thumbnail {{
            background: {theme['preview']}; padding: 0; border: none; border-radius: 7px;
        }}
        QLineEdit, QComboBox, QSpinBox, QListWidget, QPlainTextEdit {{
            background: {theme['soft']}; border: 1px solid {theme['border']};
            border-radius: 8px; padding: 6px;
        }}
        QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QCheckBox:disabled {{
            color: {theme['muted']};
        }}
        QComboBox::drop-down {{ border: none; width: 23px; }}
        QComboBox::down-arrow {{ image: none; }}
        QComboBox QAbstractItemView {{
            background: {theme['card']}; color: {theme['text']};
            selection-background-color: {theme['accent']};
            selection-color: {theme['accent_text']};
        }}
        QListWidget::item {{ padding: 3px; }}
        QCheckBox {{ spacing: 7px; }}
        QCheckBox::indicator {{
            width: 15px; height: 15px; border-radius: 4px;
            border: 1px solid {theme['muted']}; background: {theme['card']};
        }}
        QCheckBox::indicator:checked {{ background: {theme['green']}; border-color: {theme['green']}; }}
        QCheckBox::indicator:disabled {{ border-color: {theme['border']}; }}
        QProgressBar {{
            background: {theme['soft']}; border: none; border-radius: 4px;
            min-height: 7px; max-height: 7px;
        }}
        QProgressBar::chunk {{ background: {theme['green']}; border-radius: 4px; }}
        QScrollArea {{ border: none; }}
        QScrollBar:vertical {{
            background: transparent; width: 9px; margin: 0;
        }}
        QScrollBar::handle:vertical {{
            background: {theme['border']}; border-radius: 4px; min-height: 30px;
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
        QDialog {{ background: {theme['card']}; }}
    """


def app_icon() -> QIcon:
    image = QPixmap(256, 256)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor("#247955"))
    painter.drawRoundedRect(12, 12, 232, 232, 54, 54)
    painter.setPen(QPen(QColor("#ffffff"), 12))
    painter.setBrush(Qt.NoBrush)
    painter.drawRoundedRect(57, 69, 142, 118, 13, 13)
    painter.drawLine(65, 171, 110, 124)
    painter.drawLine(110, 124, 140, 151)
    painter.drawLine(140, 151, 168, 118)
    painter.drawLine(168, 118, 191, 145)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor("#ffffff"))
    painter.drawEllipse(156, 91, 18, 18)
    painter.end()
    return QIcon(image)


def time_label(seconds: float) -> str:
    milliseconds = max(0, round(float(seconds) * 1000))
    seconds, millis = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{millis:03}"


def _button(text: str, primary: bool = False) -> QToolButton:
    button = QToolButton()
    button.setText(text)
    button.setToolButtonStyle(Qt.ToolButtonTextOnly)
    button.setObjectName("Primary" if primary else "Ghost")
    button.setAttribute(Qt.WA_StyledBackground, True)
    button.setAttribute(Qt.WA_MacShowFocusRect, False)
    button.setCursor(Qt.PointingHandCursor)
    return button


def _label(text: str, style: str | None = None) -> QLabel:
    label = QLabel(text)
    if style:
        label.setObjectName(style)
    return label


def _preview_path(analysis: dict, candidate: dict) -> Path:
    path = Path(candidate.get("preview") or "")
    return path if path.is_absolute() else Path(analysis["analysis_dir"]) / path


class ChoiceComboBox(QComboBox):
    """Keep the disclosure arrow visible with native and Fusion styles."""

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        group = QPalette.Active if self.isEnabled() else QPalette.Disabled
        painter.setBrush(self.palette().color(group, QPalette.Text))
        painter.setPen(Qt.NoPen)
        x, y = self.width() - 14, self.height() // 2
        painter.drawPolygon(QPolygon([
            QPoint(x - 4, y - 2), QPoint(x + 4, y - 2), QPoint(x, y + 3),
        ]))
        painter.end()


class PreviewDialog(QDialog):
    """Show the analysis preview and the evidence behind a candidate's score."""

    def __init__(self, analysis: dict, candidate: dict, parent: QWidget):
        super().__init__(parent)
        self.setWindowTitle(f"Frame {candidate['frame_index']} · {Path(analysis['source']).name}")
        self.resize(940, 700)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(12)
        title = _label(
            f"{time_label(candidate['timestamp'])}  ·  "
            f"Frame {candidate['frame_index']}  ·  "
            f"{candidate.get('tier', 'Mid')} · Score {candidate.get('score', 0):.2f}",
            "CardTitle",
        )
        title.setWordWrap(True)
        layout.addWidget(title)
        self.image = QLabel()
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setMinimumSize(200, 160)
        self.image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.pixmap = QPixmap(str(_preview_path(analysis, candidate)))
        layout.addWidget(self.image, 1)
        reasons = list(candidate.get("reasons", []))
        if candidate.get("duplicate_of") is not None:
            reasons.append(f"Similar to frame {int(candidate['duplicate_of'])}")
        details = _label(" · ".join(str(reason) for reason in reasons) or "No additional scoring notes.", "Muted")
        details.setWordWrap(True)
        layout.addWidget(details)
        metrics = candidate.get("metrics", {})
        if metrics:
            values = []
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    values.append(f"{name.replace('_', ' ')}: {value:.3g}")
                elif isinstance(value, (str, bool)):
                    values.append(f"{name.replace('_', ' ')}: {value}")
            if values:
                metrics_label = _label(" · ".join(values), "Muted")
                metrics_label.setWordWrap(True)
                layout.addWidget(metrics_label)
        footer = QHBoxLayout()
        footer.addWidget(_label("Zero-based source frame IDs · preview only · export uses the original video", "Muted"), 1)
        close = _button("Close")
        close.clicked.connect(self.accept)
        footer.addWidget(close)
        layout.addLayout(footer)
        self._scale_image()

    def _scale_image(self) -> None:
        if self.pixmap.isNull():
            self.image.setText("Preview unavailable")
        else:
            self.image.setPixmap(self.pixmap.scaled(
                self.image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation,
            ))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._scale_image()


class CandidateCard(QFrame):
    selection_changed = Signal(int, bool)

    def __init__(self, analysis: dict, candidate: dict, selected: bool, parent: QWidget | None = None):
        super().__init__(parent)
        self.analysis = analysis
        self.candidate = candidate
        self.setObjectName("Candidate")
        self.setProperty("selected", selected)
        self.setMinimumWidth(205)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(7)
        self.thumbnail = _button("")
        self.thumbnail.setObjectName("Thumbnail")
        self.thumbnail.setMinimumHeight(108)
        self.thumbnail.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        preview = QPixmap(str(_preview_path(analysis, candidate)))
        if preview.isNull():
            self.thumbnail.setText("Preview unavailable")
        else:
            self.thumbnail.setIcon(QIcon(preview))
            self.thumbnail.setToolButtonStyle(Qt.ToolButtonIconOnly)
            self.thumbnail.setIconSize(QSize(224, 108))
        self.thumbnail.setToolTip("Open a larger preview")
        self.thumbnail.clicked.connect(self.open_preview)
        layout.addWidget(self.thumbnail)
        row = QHBoxLayout()
        row.addWidget(_label(time_label(candidate["timestamp"]), "CardTitle"))
        row.addStretch(1)
        tier = candidate.get("tier", "Mid")
        row.addWidget(_label(tier, tier if tier in ("Low", "Mid", "High") else "Mid"))
        layout.addLayout(row)
        end_card = candidate["id"] == analysis.get("end_card_id")
        subtitle = f"Frame {candidate['frame_index']}  ·  Score {candidate.get('score', 0):.2f}"
        if end_card:
            subtitle += "  ·  End card"
        source_label = _label(subtitle, "Muted")
        source_label.setToolTip("Zero-based original source frame ID, matching the CLI, reports and exported filenames")
        layout.addWidget(source_label)
        reasons = [str(reason) for reason in candidate.get("reasons", [])]
        if candidate.get("duplicate_of") is not None:
            reasons.append(f"Similar to frame {int(candidate['duplicate_of'])}")
        if not candidate.get("eligible", True):
            reasons.append("Outside automatic selection")
        note = _label(" · ".join(reasons[:2])[:135] or "Candidate for this scene", "Muted")
        note.setWordWrap(True)
        note.setToolTip("\n".join(reasons))
        layout.addWidget(note)
        layout.addStretch(1)
        self.checkbox = QCheckBox("Select this frame")
        self.checkbox.setObjectName(f"candidate_{candidate['id']}")
        self.checkbox.setChecked(selected)
        self.checkbox.toggled.connect(self._selected)
        layout.addWidget(self.checkbox)

    def _selected(self, selected: bool) -> None:
        self.setProperty("selected", selected)
        self.style().unpolish(self)
        self.style().polish(self)
        self.selection_changed.emit(self.candidate["id"], selected)

    def open_preview(self) -> None:
        PreviewDialog(self.analysis, self.candidate, self).exec()


class SceneCandidateGroup(QWidget):
    """Page large scenes so long videos do not allocate every thumbnail widget."""

    PAGE_SIZE = 12

    def __init__(self, owner: "VideoReviewCard", scene: dict, candidates: list[dict]):
        super().__init__(owner)
        self.owner = owner
        self.scene = scene
        self.candidates = sorted(candidates, key=lambda item: (item["timestamp"], item["frame_index"]))
        self.page = 0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        heading = QHBoxLayout()
        heading.addWidget(_label(
            f"Scene {int(scene['id']) + 1:02}  ·  "
            f"{time_label(scene.get('start', 0))} – {time_label(scene.get('end', 0))}",
            "CardTitle",
        ))
        heading.addStretch(1)
        self.page_label = _label("", "Muted")
        heading.addWidget(self.page_label)
        self.previous = _button("Previous")
        self.next = _button("Next")
        self.previous.clicked.connect(lambda: self._change_page(-1))
        self.next.clicked.connect(lambda: self._change_page(1))
        heading.addWidget(self.previous)
        heading.addWidget(self.next)
        self.previous.setVisible(len(self.candidates) > self.PAGE_SIZE)
        self.next.setVisible(len(self.candidates) > self.PAGE_SIZE)
        layout.addLayout(heading)
        self.grid = QGridLayout()
        self.grid.setSpacing(10)
        for column in range(3):
            self.grid.setColumnStretch(column, 1)
        layout.addLayout(self.grid)
        self._render_page()

    def _change_page(self, delta: int) -> None:
        last = max(0, (len(self.candidates) - 1) // self.PAGE_SIZE)
        self.page = max(0, min(last, self.page + delta))
        self._render_page()

    def _render_page(self) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)
                widget.deleteLater()
        start = self.page * self.PAGE_SIZE
        stop = min(len(self.candidates), start + self.PAGE_SIZE)
        for offset, candidate in enumerate(self.candidates[start:stop]):
            card = CandidateCard(self.owner.analysis, candidate, candidate["id"] in self.owner.selection, self)
            card.selection_changed.connect(self.owner.set_selected)
            self.grid.addWidget(card, offset // 3, offset % 3)
        self.page_label.setText(f"{start + 1}–{stop} of {len(self.candidates)} candidates")
        self.previous.setEnabled(self.page > 0)
        self.next.setEnabled(stop < len(self.candidates))
        self.owner.apply_busy_to_candidates()


class VideoReviewCard(QFrame):
    export_requested = Signal(object)

    def __init__(self, analysis: dict, review: bool = True, parent: QWidget | None = None):
        super().__init__(parent)
        self.analysis = analysis
        self.selection = set(analysis.get("selected_ids", []))
        self.busy = False
        self._details_built = False
        self.setObjectName("Card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)
        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = _label(Path(analysis["source"]).name, "CardTitle")
        title.setWordWrap(True)
        title.setToolTip(analysis["source"])
        title_box.addWidget(title)
        self.summary = _label("", "Muted")
        title_box.addWidget(self.summary)
        header.addLayout(title_box, 1)
        self.review_button = _button("Hide candidates" if review else "Review candidates")
        self.review_button.clicked.connect(self.toggle_details)
        header.addWidget(self.review_button, alignment=Qt.AlignTop)
        self.export_button = _button("Export selected", primary=True)
        self.export_button.clicked.connect(lambda: self.export_requested.emit(self))
        self.export_button.setVisible(review)
        header.addWidget(self.export_button, alignment=Qt.AlignTop)
        layout.addLayout(header)
        self.result_label = _label(
            "Ready for review · choose frames, then export" if review else "Analyzed · exporting selected stills…",
            "Muted",
        )
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)
        warnings = [str(value) for value in analysis.get("warnings", [])]
        self.warning_label = _label("\n".join(warnings), "Warning")
        self.warning_label.setWordWrap(True)
        self.warning_label.setVisible(bool(warnings))
        layout.addWidget(self.warning_label)
        controls = QHBoxLayout()
        self.restore_button = _button("Restore recommendation")
        self.restore_button.clicked.connect(self.restore_selection)
        self.clear_button = _button("Clear selection")
        self.clear_button.clicked.connect(self.clear_selection)
        controls.addWidget(self.restore_button)
        controls.addWidget(self.clear_button)
        controls.addStretch(1)
        controls.addWidget(_label("Click a thumbnail to enlarge", "Muted"))
        self.controls = QWidget()
        self.controls.setLayout(controls)
        self.controls.setVisible(review)
        layout.addWidget(self.controls)
        self.details = QWidget()
        self.details_layout = QVBoxLayout(self.details)
        self.details_layout.setContentsMargins(0, 0, 0, 0)
        self.details_layout.setSpacing(20)
        self.details.setVisible(review)
        layout.addWidget(self.details)
        self.open_button = _button("Open output folder")
        self.open_button.setVisible(False)
        layout.addWidget(self.open_button, alignment=Qt.AlignLeft)
        self.output_dir: str | None = None
        self.open_button.clicked.connect(self.open_output)
        if review:
            self._build_details()
        self._update_summary()

    def _build_details(self) -> None:
        if self._details_built:
            return
        self._details_built = True
        grouped: dict[int, list[dict]] = {}
        for candidate in self.analysis.get("candidates", []):
            grouped.setdefault(candidate["scene_id"], []).append(candidate)
        scenes = {scene["id"]: scene for scene in self.analysis.get("scenes", [])}
        for scene_id in sorted(grouped):
            candidates = grouped[scene_id]
            scene = scenes.get(scene_id, {
                "id": scene_id, "start": candidates[0]["timestamp"], "end": candidates[-1]["timestamp"],
            })
            self.details_layout.addWidget(SceneCandidateGroup(self, scene, candidates))
        if not grouped:
            self.details_layout.addWidget(_label("No candidates were available for this video.", "Warning"))

    def toggle_details(self) -> None:
        show = self.details.isHidden()
        if show:
            self._build_details()
        self.details.setVisible(show)
        self.controls.setVisible(show)
        self.export_button.setVisible(show)
        self.review_button.setText("Hide candidates" if show else "Review candidates")
        self._update_summary()

    def _update_summary(self) -> None:
        self.summary.setText(
            f"{len(self.selection)} selected  ·  target {self.analysis.get('target', 0)}  ·  "
            f"{len(self.analysis.get('scenes', []))} scenes"
        )
        self.export_button.setEnabled(bool(self.selection) and not self.busy)

    def set_selected(self, candidate_id: int, selected: bool) -> None:
        if selected:
            self.selection.add(candidate_id)
        else:
            self.selection.discard(candidate_id)
        self._update_summary()

    def _refresh_selection(self) -> None:
        for card in self.findChildren(CandidateCard):
            selected = card.candidate["id"] in self.selection
            card.checkbox.setChecked(selected)
        self._update_summary()

    def restore_selection(self) -> None:
        self.selection = set(self.analysis.get("selected_ids", []))
        self._refresh_selection()

    def clear_selection(self) -> None:
        self.selection.clear()
        self._refresh_selection()

    def selected_ids(self) -> list[int]:
        return sorted(self.selection)

    def apply_busy_to_candidates(self) -> None:
        for card in self.findChildren(CandidateCard):
            card.checkbox.setEnabled(not self.busy)

    def set_busy(self, busy: bool) -> None:
        self.busy = busy
        self.restore_button.setEnabled(not busy)
        self.clear_button.setEnabled(not busy)
        self.apply_busy_to_candidates()
        self._update_summary()

    def set_export_result(self, result: dict) -> None:
        self.output_dir = str(result.get("output_dir", ""))
        count = result.get("selected_count", len(result.get("files", [])))
        self.result_label.setObjectName("Success")
        self.result_label.setText(f"Exported {count} stills\n{self.output_dir}")
        self.result_label.style().unpolish(self.result_label)
        self.result_label.style().polish(self.result_label)
        self.open_button.setVisible(bool(self.output_dir))
        self.export_button.setText("Export selected again")
        warnings = list(dict.fromkeys(
            [str(value) for value in self.analysis.get("warnings", [])]
            + [str(value) for value in result.get("warnings", [])]
        ))
        self.warning_label.setText("\n".join(warnings))
        self.warning_label.setVisible(bool(warnings))

    def set_error(self, message: str) -> None:
        self.result_label.setObjectName("Error")
        self.result_label.setText(f"Export failed: {message}")
        self.result_label.style().unpolish(self.result_label)
        self.result_label.style().polish(self.result_label)

    def open_output(self) -> None:
        if self.output_dir:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.output_dir))


class ExtractionWorker(QThread):
    """Run a batch using the same entry points as the command-line interface."""

    item_ready = Signal(object)
    failed = Signal(str, str)
    progress = Signal(str, int, int)
    batch_progress = Signal(int, int)
    completed = Signal(object)

    def __init__(self, jobs: list[dict], parent: QWidget | None = None):
        super().__init__(parent)
        self.jobs = jobs
        self.cancel_event = threading.Event()

    def cancel(self) -> None:
        self.cancel_event.set()
        self.requestInterruption()

    def run(self) -> None:
        succeeded = 0
        failures = 0
        cancelled = False
        try:
            for index, job in enumerate(self.jobs):
                if self.cancel_event.is_set():
                    cancelled = True
                    break
                source = str(job.get("source") or job["analysis"]["source"])
                self.batch_progress.emit(index, len(self.jobs))

                def report(stage: str, done: int, total: int) -> None:
                    self.progress.emit(f"{Path(source).name} · {stage}", int(done), int(total))

                try:
                    if job["action"] == "legacy":
                        result = extract_legacy(source, progress=report, cancel=self.cancel_event.is_set)
                        self.item_ready.emit({"kind": "legacy", "source": source, "result": result})
                    elif job["action"] == "export":
                        result = export_analysis(
                            job["analysis"], job["options"], frame_ids=job["frame_ids"],
                            progress=report, cancel=self.cancel_event.is_set,
                        )
                        self.item_ready.emit({"kind": "export", "source": source, "result": result})
                    else:
                        analysis = analyze_video(
                            source, job["options"], progress=report, cancel=self.cancel_event.is_set,
                        )
                        self.item_ready.emit({
                            "kind": "analysis", "source": source, "analysis": analysis,
                            "review": job.get("review", False),
                        })
                        if not job.get("review", False):
                            if self.cancel_event.is_set():
                                raise CancelledError("Cancelled before export")
                            result = export_analysis(
                                analysis, job["options"], progress=report, cancel=self.cancel_event.is_set,
                            )
                            self.item_ready.emit({"kind": "export", "source": source, "result": result})
                    succeeded += 1
                except CancelledError:
                    cancelled = True
                    break
                except Exception as error:
                    failures += 1
                    self.failed.emit(source, str(error) or type(error).__name__)
                self.batch_progress.emit(index + 1, len(self.jobs))
        finally:
            self.completed.emit({
                "succeeded": succeeded, "failed": failures,
                "cancelled": cancelled or self.cancel_event.is_set(), "total": len(self.jobs),
            })


class MainWindow(QMainWindow):
    def __init__(self, settings: QSettings | None = None):
        super().__init__()
        app = QApplication.instance()
        if app is not None and sys.platform != "darwin":
            app.setStyle("Fusion")
        self.setWindowTitle("Extract Stills · V2.0")
        self.setWindowIcon(app_icon())
        self.resize(1060, 880)
        self.setMinimumSize(900, 650)
        self.setAcceptDrops(True)
        self.settings = settings or QSettings("Filmkraft", "Extract Stills")
        saved = self.settings.value("theme", "light")
        self.theme_name = saved if saved in THEMES else "light"
        self.input_paths: list[Path] = []
        self.worker: ExtractionWorker | None = None
        self.cards: dict[str, VideoReviewCard] = {}
        self._closing = False
        self._busy = False
        self._build_ui()
        self._apply_theme()
        self._update_options()
        self._show_empty()

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("Root")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(25, 22, 25, 20)
        root.setSpacing(15)
        header = QHBoxLayout()
        title = QVBoxLayout()
        title.setSpacing(3)
        title.addWidget(_label("Extract Stills", "Title"))
        title.addWidget(_label("Find clear, distinct moments in your videos", "Subtitle"))
        header.addLayout(title, 1)
        header.addWidget(_label("V2.0", "Muted"), alignment=Qt.AlignTop)
        self.theme_button = _button("Dark")
        self.theme_button.clicked.connect(self._toggle_theme)
        header.addWidget(self.theme_button, alignment=Qt.AlignTop)
        root.addLayout(header)

        source_card = QFrame()
        source_card.setObjectName("Card")
        source = QVBoxLayout(source_card)
        source.setContentsMargins(19, 16, 19, 16)
        source.setSpacing(10)
        source_header = QHBoxLayout()
        source_header.addWidget(_label("SOURCE VIDEOS", "Section"), 1)
        self.add_files_button = _button("Add videos")
        self.add_files_button.clicked.connect(self._choose_files)
        self.add_folder_button = _button("Add folder")
        self.add_folder_button.clicked.connect(self._choose_folder)
        self.remove_button = _button("Remove")
        self.remove_button.clicked.connect(self._remove_selected)
        self.clear_button = _button("Clear")
        self.clear_button.clicked.connect(self._clear_inputs)
        for button in (self.add_files_button, self.add_folder_button, self.remove_button, self.clear_button):
            source_header.addWidget(button)
        source.addLayout(source_header)
        self.source_list = QListWidget()
        self.source_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.source_list.setMaximumHeight(91)
        self.source_list.setMinimumHeight(63)
        self.source_list.setToolTip("Drop videos or folders anywhere in this window")
        source.addWidget(self.source_list)
        source_footer = QHBoxLayout()
        self.input_label = _label("Drop videos or folders here, or add them above", "Muted")
        source_footer.addWidget(self.input_label, 1)
        self.recursive_checkbox = QCheckBox("Include subfolders")
        self.recursive_checkbox.setToolTip("Find videos inside all subfolders of selected folders")
        self.recursive_checkbox.toggled.connect(self._update_input_summary)
        source_footer.addWidget(self.recursive_checkbox)
        source.addLayout(source_footer)
        root.addWidget(source_card)

        self.options_card = QFrame()
        self.options_card.setObjectName("Card")
        options_layout = QVBoxLayout(self.options_card)
        options_layout.setContentsMargins(19, 16, 19, 16)
        options_layout.setSpacing(11)
        options_layout.addWidget(_label("EXTRACTION", "Section"))
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(5)
        self.mode_combo = ChoiceComboBox()
        self.mode_combo.addItem("Smart extraction", "smart")
        self.mode_combo.addItem("Legacy extraction", "legacy")
        self.mode_combo.currentIndexChanged.connect(self._update_options)
        self.count_auto = QCheckBox("Automatic")
        self.count_auto.setChecked(True)
        self.count_auto.setToolTip("Choose a duration-based target and cover worthwhile, distinct scenes")
        self.count_auto.toggled.connect(self._update_options)
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 9999)
        self.count_spin.setValue(10)
        self.count_spin.setToolTip("Limit smart extraction to this many stills")
        count_widget = QWidget()
        count_layout = QHBoxLayout(count_widget)
        count_layout.setContentsMargins(0, 0, 0, 0)
        count_layout.setSpacing(8)
        count_layout.addWidget(self.count_auto)
        count_layout.addWidget(self.count_spin)
        self.format_combo = ChoiceComboBox()
        for label, value in (("PNG", "png"), ("TIFF", "tiff"), ("JPEG", "jpeg")):
            self.format_combo.addItem(label, value)
        self.format_combo.currentIndexChanged.connect(self._format_changed)
        self.depth_combo = ChoiceComboBox()
        self.depth_combo.addItem("8-bit", 8)
        self.depth_combo.addItem("16-bit", 16)
        self.depth_combo.currentIndexChanged.connect(self._depth_changed)
        self.color_combo = ChoiceComboBox()
        self.color_combo.addItem("sRGB", "srgb")
        self.color_combo.addItem("Source color", "source")
        self.color_combo.setToolTip("Source color is available with 16-bit PNG or TIFF")
        self.color_combo.currentIndexChanged.connect(self._color_changed)
        for column, (label, widget) in enumerate((
            ("Method", self.mode_combo), ("Stills", count_widget),
            ("Format", self.format_combo), ("Depth", self.depth_combo), ("Color", self.color_combo),
        )):
            grid.addWidget(_label(label, "Muted"), 0, column)
            grid.addWidget(widget, 1, column)
        grid.setColumnStretch(0, 2)
        grid.setColumnStretch(1, 2)
        options_layout.addLayout(grid)
        output_row = QHBoxLayout()
        output_row.addWidget(_label("Output", "Muted"))
        self.output_edit = QLineEdit()
        self.output_edit.setReadOnly(True)
        self.output_edit.setPlaceholderText("Adjacent stills folder · a separate run folder for each video")
        output_row.addWidget(self.output_edit, 1)
        self.output_button = _button("Choose folder")
        self.output_button.clicked.connect(self._choose_output)
        output_row.addWidget(self.output_button)
        self.reset_output_button = _button("Reset")
        self.reset_output_button.clicked.connect(self.output_edit.clear)
        output_row.addWidget(self.reset_output_button)
        options_layout.addLayout(output_row)
        flags = QHBoxLayout()
        self.review_checkbox = QCheckBox("Review before export")
        self.review_checkbox.setToolTip("Inspect scored candidates and adjust the selection before writing stills")
        self.report_checkbox = QCheckBox("Detailed score reports")
        self.report_checkbox.setToolTip("Save HTML, JSON and CSV reports alongside the extracted stills")
        flags.addWidget(self.review_checkbox)
        flags.addWidget(self.report_checkbox)
        flags.addStretch(1)
        options_layout.addLayout(flags)
        self.mode_note = _label("", "Muted")
        self.mode_note.setWordWrap(True)
        options_layout.addWidget(self.mode_note)
        root.addWidget(self.options_card)

        run_row = QHBoxLayout()
        progress_box = QVBoxLayout()
        progress_box.setSpacing(5)
        self.status_label = _label("Ready when you are", "Muted")
        self.status_label.setWordWrap(True)
        progress_box.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_box.addWidget(self.progress_bar)
        run_row.addLayout(progress_box, 1)
        self.batch_label = _label("", "Muted")
        run_row.addWidget(self.batch_label)
        self.cancel_button = _button("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)
        run_row.addWidget(self.cancel_button)
        self.run_button = _button("Extract stills", primary=True)
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self._run)
        run_row.addWidget(self.run_button)
        root.addLayout(run_row)

        self.results_scroll = QScrollArea()
        self.results_scroll.setWidgetResizable(True)
        self.results_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        container = QWidget()
        self.results_layout = QVBoxLayout(container)
        self.results_layout.setContentsMargins(0, 0, 6, 0)
        self.results_layout.setSpacing(14)
        self.results_scroll.setWidget(container)
        root.addWidget(self.results_scroll, 1)
        log_row = QHBoxLayout()
        self.log_toggle = _button("Show progress log")
        self.log_toggle.clicked.connect(self._toggle_log)
        log_row.addWidget(self.log_toggle)
        log_row.addStretch(1)
        log_row.addWidget(_label("Offline processing · original videos stay untouched", "Muted"))
        root.addLayout(log_row)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(1000)
        self.log_box.setFixedHeight(120)
        self.log_box.setVisible(False)
        root.addWidget(self.log_box)
        self._source_controls = [
            self.add_files_button, self.add_folder_button, self.remove_button,
            self.clear_button, self.source_list, self.recursive_checkbox,
        ]
        self._smart_controls = [
            self.count_auto, self.format_combo, self.depth_combo, self.color_combo,
            self.output_edit, self.output_button, self.reset_output_button,
            self.review_checkbox, self.report_checkbox,
        ]
        self.review_checkbox.toggled.connect(self._update_run_label)

    def _apply_theme(self) -> None:
        self.setStyleSheet(build_stylesheet(THEMES[self.theme_name]))
        self.theme_button.setText("Light" if self.theme_name == "dark" else "Dark")

    def _toggle_theme(self) -> None:
        self.theme_name = "dark" if self.theme_name == "light" else "light"
        self.settings.setValue("theme", self.theme_name)
        self._apply_theme()

    def _update_run_label(self) -> None:
        review = self.mode_combo.currentData() == "smart" and self.review_checkbox.isChecked()
        self.run_button.setText("Analyze videos" if review else "Extract stills")

    def _update_options(self, *_args) -> None:
        smart = self.mode_combo.currentData() == "smart"
        for widget in self._smart_controls:
            widget.setEnabled(smart and not self._busy)
        self.count_spin.setEnabled(smart and not self._busy and not self.count_auto.isChecked())
        self.depth_combo.setEnabled(smart and not self._busy and self.format_combo.currentData() != "jpeg")
        self.mode_note.setText(
            "Automatic selection balances clarity, scene coverage and variety. The ending frame is exported last."
            if smart else
            "Legacy: every 24th frame plus the literal last frame, original PNG names, saved in the adjacent stills folder."
        )
        self._update_run_label()

    def _format_changed(self, *_args) -> None:
        if self.format_combo.currentData() == "jpeg":
            self.depth_combo.setCurrentIndex(0)
            self.color_combo.setCurrentIndex(0)
        self._update_options()

    def _depth_changed(self, *_args) -> None:
        if self.depth_combo.currentData() == 8 and self.color_combo.currentData() == "source":
            self.color_combo.setCurrentIndex(0)

    def _color_changed(self, *_args) -> None:
        if self.color_combo.currentData() == "source":
            if self.format_combo.currentData() == "jpeg":
                self.format_combo.setCurrentIndex(0)
            self.depth_combo.setCurrentIndex(1)
        self._update_options()

    def current_options(self, mode: str | None = None) -> Options:
        requested_mode = mode or self.mode_combo.currentData()
        if requested_mode == "legacy":
            return Options(mode="legacy", recursive=self.recursive_checkbox.isChecked())
        return Options(
            mode=requested_mode,
            count=None if self.count_auto.isChecked() else self.count_spin.value(),
            format=self.format_combo.currentData(), bit_depth=self.depth_combo.currentData(),
            color=self.color_combo.currentData(), output=self.output_edit.text().strip() or None,
            report=self.report_checkbox.isChecked(), recursive=self.recursive_checkbox.isChecked(),
        )

    def add_paths(self, paths) -> None:
        if self._busy:
            return
        seen = {os.path.normcase(str(path)) for path in self.input_paths}
        for value in paths:
            path = Path(value).expanduser().resolve()
            if path.exists() and os.path.normcase(str(path)) not in seen:
                self.input_paths.append(path)
                self.source_list.addItem(str(path))
                seen.add(os.path.normcase(str(path)))
        self._update_input_summary()

    def _choose_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Choose videos", "",
            "Video files (*.mp4 *.mov *.mkv *.avi *.mxf *.m4v *.webm *.mpg *.mpeg *.mts *.m2ts);;All files (*.*)",
        )
        self.add_paths(paths)

    def _choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose folder of videos")
        if path:
            self.add_paths([path])

    def _choose_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose output folder", self.output_edit.text())
        if path:
            self.output_edit.setText(str(Path(path).resolve()))

    def _remove_selected(self) -> None:
        for item in self.source_list.selectedItems():
            row = self.source_list.row(item)
            self.source_list.takeItem(row)
            self.input_paths.pop(row)
        self._update_input_summary()

    def _clear_inputs(self) -> None:
        self.input_paths.clear()
        self.source_list.clear()
        self._update_input_summary()

    def _update_input_summary(self, *_args) -> None:
        try:
            files = discover_inputs(self.input_paths, recursive=self.recursive_checkbox.isChecked())
            count = len(files)
            self.input_label.setText(
                f"{count} video{'s' if count != 1 else ''} queued"
                if self.input_paths else "Drop videos or folders here, or add them above"
            )
        except Exception as error:
            count = 0
            self.input_label.setText(str(error))
        self.run_button.setEnabled(count > 0 and not self._busy)
        self.remove_button.setEnabled(bool(self.input_paths) and not self._busy)
        self.clear_button.setEnabled(bool(self.input_paths) and not self._busy)

    def dragEnterEvent(self, event) -> None:
        if not self._busy and event.mimeData().hasUrls():
            if any(url.isLocalFile() for url in event.mimeData().urls()):
                event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        if not self._busy:
            self.add_paths([url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()])
            event.acceptProposedAction()

    def _clear_results(self) -> None:
        self.cards.clear()
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)
                widget.deleteLater()

    def _show_empty(self) -> None:
        self._clear_results()
        empty = QFrame()
        empty.setObjectName("DropZone")
        layout = QVBoxLayout(empty)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.addStretch(1)
        title = _label("A stronger selection of stills", "CardTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        description = _label(
            "Add videos to find clear frames from each scene.\n"
            "Turn on Review before export to inspect and choose your moments.",
            "Muted",
        )
        description.setAlignment(Qt.AlignCenter)
        description.setWordWrap(True)
        layout.addWidget(description)
        layout.addStretch(1)
        self.results_layout.addWidget(empty, 1)

    def _run(self) -> None:
        if self._busy:
            return
        try:
            sources = discover_inputs(self.input_paths, recursive=self.recursive_checkbox.isChecked())
        except Exception as error:
            self.status_label.setText(f"Could not read inputs: {error}")
            return
        if not sources:
            self.status_label.setText("No supported videos were found in the selected inputs")
            return
        options = self.current_options()
        review = options.mode == "smart" and self.review_checkbox.isChecked()
        jobs = [{
            "source": path, "options": options, "review": review,
            "action": "legacy" if options.mode == "legacy" else "smart",
        } for path in sources]
        self._clear_results()
        self.log_box.clear()
        self._start_worker(jobs)

    def _start_worker(self, jobs: list[dict]) -> None:
        self._set_busy(True)
        self.status_label.setText("Starting…")
        self.progress_bar.setRange(0, 0)
        self.worker = ExtractionWorker(jobs, self)
        self.worker.item_ready.connect(self._on_item)
        self.worker.failed.connect(self._on_failure)
        self.worker.progress.connect(self._on_progress)
        self.worker.batch_progress.connect(self._on_batch_progress)
        self.worker.completed.connect(self._on_completed)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        for control in self._source_controls:
            control.setEnabled(not busy)
        self.mode_combo.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.cancel_button.setText("Cancel")
        for card in self.cards.values():
            card.set_busy(busy)
        self._update_options()
        self._update_input_summary()

    def _cancel(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.cancel_button.setEnabled(False)
            self.cancel_button.setText("Cancelling…")
            self.status_label.setText("Cancelling · finishing the current operation safely…")
            self.log_box.appendPlainText("Cancellation requested")

    def _on_progress(self, stage: str, done: int, total: int) -> None:
        if self.worker is not None and self.worker.cancel_event.is_set():
            return
        self.status_label.setText(f"{stage}  ({done}/{total})" if total else stage)
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(min(done, total))
        else:
            self.progress_bar.setRange(0, 0)
        self.log_box.appendPlainText(self.status_label.text())

    def _on_batch_progress(self, done: int, total: int) -> None:
        self.batch_label.setText(f"{done}/{total} finished")

    def _on_item(self, item: dict) -> None:
        source = item["source"]
        if item["kind"] == "analysis":
            card = VideoReviewCard(item["analysis"], review=item["review"])
            card.export_requested.connect(self._export_card)
            card.set_busy(self._busy)
            self.cards[source] = card
            self.results_layout.addWidget(card)
            self.log_box.appendPlainText(f"Analysis ready: {source}")
        elif item["kind"] == "export":
            if source in self.cards:
                self.cards[source].set_export_result(item["result"])
            self.log_box.appendPlainText(f"Export complete: {source}")
        else:
            result = item["result"]
            card = QFrame()
            card.setObjectName("Card")
            layout = QVBoxLayout(card)
            layout.setContentsMargins(20, 18, 20, 18)
            layout.addWidget(_label(Path(source).name, "CardTitle"))
            files = result.get("files", [])
            output = str(result.get("output_dir", Path(source).parent / "stills"))
            label = _label(f"Legacy extraction · {len(files)} stills\n{output}", "Success")
            label.setWordWrap(True)
            layout.addWidget(label)
            open_button = _button("Open output folder")
            open_button.clicked.connect(lambda _checked=False, path=output: QDesktopServices.openUrl(QUrl.fromLocalFile(path)))
            layout.addWidget(open_button, alignment=Qt.AlignLeft)
            self.results_layout.addWidget(card)
            self.log_box.appendPlainText(f"Legacy export complete: {source}")

    def _on_failure(self, source: str, message: str) -> None:
        self.log_box.appendPlainText(f"Failed: {source}\n{message}")
        if source in self.cards:
            self.cards[source].set_error(message)
            return
        card = QFrame()
        card.setObjectName("Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.addWidget(_label(Path(source).name, "CardTitle"))
        detail = _label(message, "Error")
        detail.setWordWrap(True)
        layout.addWidget(detail)
        self.results_layout.addWidget(card)

    def _on_completed(self, result: dict) -> None:
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0 if result["cancelled"] else 100)
        if result["cancelled"]:
            self.status_label.setText(
                f"Cancelled · {result['succeeded']} completed · {result['failed']} failed"
            )
            for card in self.cards.values():
                if not card.output_dir:
                    card.result_label.setText("Analysis available · review the candidates and export when ready")
                    if card.details.isHidden():
                        card.toggle_details()
        else:
            review = any(job.get("review", False) for job in (self.worker.jobs if self.worker else []))
            self.status_label.setText(
                f"{'Ready for review' if review else 'Done'} · {result['succeeded']} completed · "
                f"{result['failed']} failed"
            )
        self.log_box.appendPlainText(self.status_label.text())

    def _on_worker_finished(self) -> None:
        worker = self.worker
        self.worker = None
        if worker is not None:
            worker.deleteLater()
        self._set_busy(False)
        if self._closing:
            self.close()

    def _export_card(self, card: VideoReviewCard) -> None:
        if self._busy or not card.selection:
            return
        self._start_worker([{
            "action": "export", "source": card.analysis["source"], "analysis": card.analysis,
            "frame_ids": card.selected_ids(), "options": self.current_options(mode="smart"),
        }])

    def _toggle_log(self) -> None:
        show = not self.log_box.isVisible()
        self.log_box.setVisible(show)
        self.log_toggle.setText("Hide progress log" if show else "Show progress log")

    def closeEvent(self, event) -> None:
        if self.worker is not None and self.worker.isRunning():
            self._closing = True
            self._cancel()
            self.status_label.setText("Closing after cancellation finishes safely…")
            event.ignore()
        else:
            event.accept()


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--legacy-worker":
        from .cli import main as cli_main
        return cli_main(sys.argv[1:])
    app = QApplication.instance() or QApplication([sys.argv[0]])
    app.setApplicationName("Extract Stills")
    app.setOrganizationName("Filmkraft")
    app.setWindowIcon(app_icon())
    window = MainWindow()
    window.show()
    if "--smoke-test" in sys.argv or os.environ.get("EXTRACT_STILLS_GUI_SMOKE_TEST") == "1":
        app.processEvents()
        window.close()
        app.processEvents()
        return 0
    return app.exec()

"""Offscreen GUI integration smoke with real Qt workers and controlled engine results.

Run: .venv/Scripts/python.exe tests/gui_smoke.py
The smoke saves light/dark screenshots under validation/ for visual inspection.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPainter, QPixmap
from PySide6.QtWidgets import QApplication

from stills_tool import gui


APP = QApplication.instance() or QApplication([])


def register_fonts() -> None:
    directory = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    for filename in ("segoeui.ttf", "segoeuib.ttf", "seguisb.ttf", "seguisym.ttf"):
        path = directory / filename
        if path.exists():
            QFontDatabase.addApplicationFont(str(path))


register_fonts()


def pump_until(predicate, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        APP.processEvents()
        if time.monotonic() >= deadline:
            raise AssertionError("GUI operation did not finish within the smoke timeout")
        time.sleep(0.005)
    APP.processEvents()


def make_analysis(directory: Path, name: str = "Campaign_30_HD_Social_H264.mp4") -> dict:
    source = directory / name
    source.touch()
    previews = directory / "analysis"
    previews.mkdir(exist_ok=True)
    candidates = []
    rows = [
        (0, 0.0, 0, 57.2, "Mid", ["Opening transition", "Text still entering"]),
        (48, 2.0, 0, 86.4, "High", ["Clear subject", "Settled title"]),
        (144, 6.0, 1, 64.8, "Mid", ["Small expression change", "Moderate sharpness"]),
        (168, 7.0, 1, 91.7, "High", ["Stable ending", "Strong text contrast"]),
    ]
    for index, timestamp, scene, score, tier, reasons in rows:
        filename = f"frame_{index}.png"
        pixmap = QPixmap(640, 360)
        pixmap.fill(QColor("#246e71" if scene == 0 else "#273753"))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#e9c6a1"))
        painter.drawEllipse(267, 78 if index != 144 else 84, 106, 110)
        painter.setBrush(QColor("#efe7d6"))
        painter.drawRoundedRect(242, 188, 157, 192, 56, 56)
        painter.setBrush(QColor("#172d39"))
        painter.drawEllipse(294, 117, 9, 6)
        painter.drawEllipse(338, 117, 9, 6)
        painter.setPen(QColor("#7e4e44"))
        painter.drawArc(305, 139, 32, 20, 190 * 16, 160 * 16)
        painter.setPen(QColor("#ffffff"))
        painter.setFont(QFont("Segoe UI", 28, QFont.Bold))
        painter.drawText(25, 48, "A moment worth keeping" if scene == 0 else "MAKE IT MATTER")
        painter.setFont(QFont("Segoe UI", 15))
        painter.drawText(27, 328, "Campaign film · smoke fixture")
        painter.end()
        assert pixmap.save(str(previews / filename))
        candidates.append({
            "id": index, "frame_index": index, "timestamp": timestamp, "scene_id": scene,
            "score": score, "tier": tier, "reasons": reasons,
            "metrics": {"sharpness": score / 100, "text_stability": 0.95},
            "preview": filename, "eligible": index != 0, "duplicate_of": None,
        })
    return {
        "source": str(source), "analysis_dir": str(previews),
        "video": {"duration": 8.0, "width": 1920, "height": 1080},
        "scenes": [{"id": 0, "start": 0.0, "end": 5.0}, {"id": 1, "start": 5.0, "end": 8.0}],
        "candidates": candidates, "selected_ids": [48, 168], "end_card_id": 168,
        "target": 3, "base_target": 3,
        "warnings": ["Selected 2 distinct frames; the target was 3."],
    }


def export_result(analysis: dict, frame_ids: list[int] | None = None) -> dict:
    selected = analysis["selected_ids"] if frame_ids is None else frame_ids
    return {
        "source": analysis["source"], "output_dir": str(Path(analysis["analysis_dir"]).parent / "stills"),
        "files": [{"frame_id": value, "path": f"still_{number:03}.png"} for number, value in enumerate(selected, 1)],
        "selected_count": len(selected), "target": analysis["target"], "warnings": analysis["warnings"],
    }


class GuiSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="extract-stills-gui-")
        self.directory = Path(self.temp.name)
        self.analysis = make_analysis(self.directory)
        self.settings = QSettings(str(self.directory / "settings.ini"), QSettings.IniFormat)
        self.window = gui.MainWindow(settings=self.settings)
        self.window.show()
        APP.processEvents()

    def tearDown(self):
        if self.window.worker is not None:
            self.window.worker.cancel()
            pump_until(lambda: self.window.worker is None)
        self.window.close()
        APP.processEvents()
        self.temp.cleanup()

    def test_smart_defaults_and_legacy_preserves_original_contract(self):
        defaults = self.window.current_options()
        defaults.validate()
        self.assertEqual((defaults.mode, defaults.format, defaults.bit_depth, defaults.color), ("smart", "png", 8, "srgb"))
        self.assertIsNone(defaults.count)
        self.assertFalse(self.window.review_checkbox.isChecked())
        self.window.count_auto.setChecked(False)
        self.window.count_spin.setValue(17)
        self.window.format_combo.setCurrentIndex(1)
        self.window.color_combo.setCurrentIndex(1)
        self.window.output_edit.setText(str(self.directory / "custom"))
        self.window.report_checkbox.setChecked(True)
        self.window.mode_combo.setCurrentIndex(1)
        legacy = self.window.current_options()
        legacy.validate()
        self.assertEqual((legacy.format, legacy.bit_depth, legacy.color), ("png", 8, "srgb"))
        self.assertIsNone(legacy.count)
        self.assertIsNone(legacy.output)
        self.assertFalse(legacy.report)
        self.assertFalse(self.window.output_button.isEnabled())
        self.assertFalse(self.window.review_checkbox.isEnabled())
        self.assertFalse(self.window.count_spin.isEnabled())
        self.assertTrue(self.window.recursive_checkbox.isEnabled())
        self.window.mode_combo.setCurrentIndex(0)
        smart = self.window.current_options()
        smart.validate()
        self.assertEqual((smart.count, smart.format, smart.bit_depth, smart.color), (17, "tiff", 16, "source"))
        self.assertTrue(self.window.output_button.isEnabled())

    def test_format_depth_and_color_are_always_compatible(self):
        self.window.color_combo.setCurrentIndex(1)
        self.assertEqual(self.window.depth_combo.currentData(), 16)
        self.window.format_combo.setCurrentIndex(2)
        self.assertEqual(self.window.depth_combo.currentData(), 8)
        self.assertEqual(self.window.color_combo.currentData(), "srgb")
        self.assertFalse(self.window.depth_combo.isEnabled())
        self.window.current_options().validate()
        self.window.color_combo.setCurrentIndex(1)
        self.assertEqual(self.window.format_combo.currentData(), "png")
        self.assertEqual(self.window.depth_combo.currentData(), 16)
        self.window.depth_combo.setCurrentIndex(0)
        self.assertEqual(self.window.color_combo.currentData(), "srgb")
        self.window.current_options().validate()

    def test_theme_persists_and_both_palettes_apply(self):
        self.assertEqual(self.window.theme_name, "light")
        self.window._toggle_theme()
        self.assertEqual(self.settings.value("theme"), "dark")
        self.assertIn(gui.THEMES["dark"]["card"], self.window.styleSheet())
        other = gui.MainWindow(settings=self.settings)
        self.assertEqual(other.theme_name, "dark")
        other.close()
        self.window._toggle_theme()
        self.assertIn(gui.THEMES["light"]["card"], self.window.styleSheet())

    def test_word_cues_are_plain_text_compact_and_optional(self):
        ordinary = gui.CandidateCard(self.analysis, self.analysis["candidates"][0], False)
        self.assertFalse(hasattr(ordinary, "text_snippet"))
        ordinary.close()
        reading = copy.deepcopy(self.analysis["candidates"][1])
        reading.update(
            text_contents=["Visible <b>wording</b> & a long subtitle repeated " * 8],
            text_ready=False, text_reason="Detected word crop may omit letters",
        )
        card = gui.CandidateCard(self.analysis, reading, True)
        self.window._clear_results()
        self.window.results_layout.addWidget(card)
        APP.processEvents()
        self.assertEqual(card.text_readiness.text(), "Words incomplete")
        self.assertEqual(card.text_snippet.textFormat(), Qt.PlainText)
        self.assertLessEqual(card.text_snippet.text().count("\n"), 1)
        self.assertIn("&lt;b&gt;", card.text_snippet.toolTip())
        self.assertIn("omit letters", card.text_readiness.toolTip())
        self.assertTrue(self.window.count_auto.isChecked())
        for theme in ("light", "dark"):
            self.window.theme_name = theme
            self.window._apply_theme()
            APP.processEvents()
            self.assertLessEqual(card.text_snippet.height(), card.text_snippet.fontMetrics().lineSpacing() * 2 + 2)

    def test_inputs_deduplicate_and_review_label_updates(self):
        self.window.add_paths([self.analysis["source"], self.analysis["source"]])
        self.assertEqual(self.window.source_list.count(), 1)
        self.assertTrue(self.window.run_button.isEnabled())
        self.window.review_checkbox.setChecked(True)
        self.assertEqual(self.window.run_button.text(), "Analyze videos")
        self.window._clear_inputs()
        self.assertFalse(self.window.run_button.isEnabled())

    def test_scene_pages_retain_selections_and_allow_manual_candidates(self):
        analysis = copy.deepcopy(self.analysis)
        base = analysis["candidates"][0]
        analysis["candidates"] = [
            dict(base, id=index * 24, frame_index=index * 24, timestamp=float(index), scene_id=0)
            for index in range(25)
        ]
        analysis["scenes"] = [{"id": 0, "start": 0.0, "end": 25.0}]
        analysis["selected_ids"] = [0, 576]
        self.window._clear_results()
        card = gui.VideoReviewCard(analysis)
        self.window.results_layout.addWidget(card)
        APP.processEvents()
        group = card.findChild(gui.SceneCandidateGroup)
        self.assertEqual(group.grid.count(), 12)
        first = group.grid.itemAt(0).widget()
        self.assertFalse(first.thumbnail.icon().isNull())
        self.assertEqual(first.thumbnail.toolButtonStyle(), Qt.ToolButtonIconOnly)
        self.assertTrue(first.checkbox.isEnabled(), "Manual review may select candidates rejected by the recommendation")
        group._change_page(1)
        second_page_first = group.grid.itemAt(0).widget()
        self.assertEqual(second_page_first.candidate["id"], 288)
        second_page_first.checkbox.setChecked(True)
        group._change_page(1)
        self.assertEqual(group.grid.count(), 1)
        self.assertTrue(group.grid.itemAt(0).widget().checkbox.isChecked())
        group._change_page(-1)
        self.assertTrue(group.grid.itemAt(0).widget().checkbox.isChecked())
        self.assertEqual(card.selected_ids(), [0, 288, 576])
        card.clear_selection()
        self.assertEqual(card.selected_ids(), [])
        self.assertFalse(card.export_button.isEnabled())
        card.restore_selection()
        self.assertEqual(card.selected_ids(), [0, 576])
        preview = gui.PreviewDialog(analysis, analysis["candidates"][0], self.window)
        preview.show()
        APP.processEvents()
        self.assertFalse(preview.pixmap.isNull())
        self.assertGreater(preview.image.pixmap().width(), 0)
        preview.close()

    def test_background_batch_continues_after_failure_and_auto_exports(self):
        good = self.analysis["source"]
        bad = self.directory / "broken.mp4"
        bad.touch()
        analyzed = []
        exported = []

        def analyze(source, options, progress, cancel):
            analyzed.append(str(source))
            if Path(source).name == "broken.mp4":
                raise ValueError("Decoder rejected the input")
            progress("Scoring", 4, 4)
            return copy.deepcopy(self.analysis)

        def export(analysis, options, frame_ids=None, progress=None, cancel=None):
            exported.append(frame_ids)
            progress("Exporting", 2, 2)
            return export_result(analysis, frame_ids)

        self.window.add_paths([bad, good])
        with patch.object(gui, "analyze_video", side_effect=analyze), patch.object(gui, "export_analysis", side_effect=export):
            self.window._run()
            self.assertFalse(self.window.mode_combo.isEnabled())
            pump_until(lambda: self.window.worker is None)
        self.assertEqual(len(analyzed), 2)
        self.assertEqual(exported, [None])
        self.assertIn("1 completed · 1 failed", self.window.status_label.text())
        self.assertIn("Exported 2 stills", self.window.cards[good].result_label.text())
        self.assertTrue(self.window.cards[good].open_button.isVisible())
        self.assertTrue(self.window.mode_combo.isEnabled())

    def test_review_defers_export_and_manual_selection_reaches_core(self):
        self.window.add_paths([self.analysis["source"]])
        self.window.review_checkbox.setChecked(True)
        with patch.object(gui, "analyze_video", return_value=copy.deepcopy(self.analysis)), patch.object(gui, "export_analysis") as export:
            self.window._run()
            pump_until(lambda: self.window.worker is None)
            export.assert_not_called()
        card = self.window.cards[self.analysis["source"]]
        self.assertFalse(card.details.isHidden())
        card.clear_selection()
        card.set_selected(0, True)
        with patch.object(gui, "export_analysis", return_value=export_result(self.analysis, [0])) as export:
            self.window._export_card(card)
            pump_until(lambda: self.window.worker is None)
            self.assertEqual(export.call_args.kwargs["frame_ids"], [0])
        self.assertIn("Exported 1 stills", card.result_label.text())

    def test_legacy_job_calls_original_extractor_without_smart_options(self):
        self.window.add_paths([self.analysis["source"]])
        self.window.output_edit.setText(str(self.directory / "ignored-smart-output"))
        self.window.report_checkbox.setChecked(True)
        self.window.mode_combo.setCurrentIndex(1)
        with patch.object(gui, "extract_legacy", return_value=export_result(self.analysis)) as legacy:
            self.window._run()
            pump_until(lambda: self.window.worker is None)
        self.assertEqual(legacy.call_args.args, (self.analysis["source"],))
        self.assertEqual(set(legacy.call_args.kwargs), {"progress", "cancel"})
        self.assertIn("1 completed · 0 failed", self.window.status_label.text())

    def test_frozen_desktop_legacy_worker_dispatches_without_opening_window(self):
        arguments = ["Extract Stills.exe", "--legacy-worker", self.analysis["source"]]
        with patch.object(sys, "argv", arguments), patch("stills_tool.cli.main", return_value=0) as cli:
            self.assertEqual(gui.main(), 0)
            cli.assert_called_once_with(arguments[1:])

    def test_close_cancels_worker_and_waits_before_destroying_window(self):
        started = threading.Event()

        def analyze(source, options, progress, cancel):
            started.set()
            while not cancel():
                time.sleep(0.005)
            raise gui.CancelledError("Stopped by user")

        self.window.add_paths([self.analysis["source"]])
        with patch.object(gui, "analyze_video", side_effect=analyze):
            self.window._run()
            pump_until(started.is_set)
            self.window.close()
            self.assertTrue(self.window._closing)
            self.assertTrue(self.window.worker.cancel_event.is_set())
            pump_until(lambda: self.window.worker is None)
        self.assertFalse(self.window.isVisible())
        self.assertIn("Cancelled", self.window.status_label.text())


def save_visuals() -> None:
    output = ROOT / "validation"
    output.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="extract-stills-gui-visual-") as temporary:
        directory = Path(temporary)
        analysis = make_analysis(directory)
        for item in analysis["candidates"]:
            item["text_contents"] = ["A moment worth keeping" if item["scene_id"] == 0 else "Make it matter"]
            item["text_ready"] = item["id"] != 0
            item["text_reason"] = "Text is still entering" if item["id"] == 0 else "Repeated readable observed wording"
        settings = QSettings(str(directory / "settings.ini"), QSettings.IniFormat)
        window = gui.MainWindow(settings=settings)
        window.add_paths([analysis["source"]])
        window.review_checkbox.setChecked(True)
        window._clear_results()
        window._on_item({"kind": "analysis", "source": analysis["source"], "analysis": analysis, "review": True})
        window.status_label.setText("Ready for review · 1 completed · 0 failed")
        window.batch_label.setText("1/1 finished")
        window.progress_bar.setValue(100)
        window.resize(1120, 1670)
        window.show()
        for theme in ("light", "dark"):
            window.theme_name = theme
            window._apply_theme()
            APP.processEvents()
            assert window.grab().save(str(output / f"gui_smoke_{theme}.png"))
        window.close()
        APP.processEvents()
    print(f"Visual QA screenshots: {output / 'gui_smoke_light.png'} and {output / 'gui_smoke_dark.png'}")


def main() -> int:
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GuiSmokeTests))
    if result.wasSuccessful():
        save_visuals()
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Kiểm tra báo cáo được xuất từ output thật và không che giấu lỗi notebook."""

import base64
import json
import tempfile
import unittest
from pathlib import Path

from scripts.ipynb_to_markdown import convert_notebook


class NotebookExportTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.root = Path(self.temp_directory.name)
        self.notebook_path = self.root / "input.ipynb"
        self.report_path = self.root / "report" / "KET_QUA_MO_HINH.md"

    def write_notebook(self, cells, encoding="utf-8"):
        self.notebook_path.write_text(
            json.dumps({"cells": cells}, ensure_ascii=False), encoding=encoding
        )

    def code_cell(self, outputs, source="print('kết quả')", count=1):
        return {
            "cell_type": "code",
            "source": source,
            "execution_count": count,
            "outputs": outputs,
        }

    def test_preserves_math_code_and_actual_stream_without_ansi(self):
        self.write_notebook(
            [
                {"cell_type": "markdown", "source": ["# Toán học\n", "$X=U\\Sigma V^T$"]},
                self.code_cell(
                    [{"output_type": "stream", "text": ["\x1b[32mMacro-F1: ", "0.91\x1b[0m\n"]}],
                    source=["score = 0.91\n", "print(score)"],
                ),
            ],
            encoding="utf-8-sig",
        )
        result = convert_notebook(self.notebook_path, self.report_path)
        report = result.read_text(encoding="utf-8")
        self.assertEqual(result, self.report_path)
        self.assertIn("$X=U\\Sigma V^T$", report)
        self.assertIn("```python\nscore = 0.91\nprint(score)\n```", report)
        self.assertIn("```text\nMacro-F1: 0.91\n```", report)
        self.assertNotIn("\x1b", report)

    def test_prefers_markdown_table_over_png_and_plain_text(self):
        table = "| Model | F1 |\n| --- | --- |\n| SVD | 0.90 |"
        self.write_notebook(
            [self.code_cell([{
                "output_type": "display_data",
                "data": {
                    "text/markdown": [table],
                    "image/png": "this value must not be decoded",
                    "text/plain": "duplicate table representation",
                },
            }])]
        )
        convert_notebook(self.notebook_path, self.report_path)
        report = self.report_path.read_text(encoding="utf-8")
        self.assertIn(table, report)
        self.assertNotIn("duplicate table representation", report)
        self.assertFalse((self.report_path.parent / "outputs/figures").exists())

    def test_writes_png_once_with_relative_link_and_plain_text_output(self):
        image_bytes = b"\x89PNG\r\n\x1a\nexample-output"
        self.write_notebook(
            [self.code_cell([
                {
                    "output_type": "display_data",
                    "data": {
                        "image/png": base64.b64encode(image_bytes).decode("ascii"),
                        "text/plain": "Figure duplicate",
                    },
                },
                {"output_type": "execute_result", "data": {"text/plain": ["(100, ", "10)"]}},
            ])]
        )
        convert_notebook(self.notebook_path, self.report_path)
        image_path = self.report_path.parent / "outputs/figures/notebook/cell_1_output_1.png"
        self.assertEqual(image_path.read_bytes(), image_bytes)
        report = self.report_path.read_text(encoding="utf-8")
        self.assertIn("(<outputs/figures/notebook/cell_1_output_1.png>)", report)
        self.assertNotIn("Figure duplicate", report)
        self.assertIn("```text\n(100, 10)\n```", report)

    def test_rejects_unexecuted_nonempty_code_without_overwriting_report(self):
        self.report_path.parent.mkdir(parents=True)
        self.report_path.write_text("Báo cáo trước", encoding="utf-8")
        self.write_notebook([self.code_cell([], count=None)])
        with self.assertRaisesRegex(ValueError, "chưa được chạy"):
            convert_notebook(self.notebook_path, self.report_path)
        self.assertEqual(self.report_path.read_text(encoding="utf-8"), "Báo cáo trước")

    def test_reuses_existing_identical_figure_without_duplicate_image(self):
        image_bytes = b"\x89PNG\r\n\x1a\nexisting-figure"
        figure = self.report_path.parent / "outputs/figures/test_confusion.png"
        figure.parent.mkdir(parents=True)
        figure.write_bytes(image_bytes)
        self.write_notebook([self.code_cell([{
            "output_type": "display_data",
            "data": {"image/png": base64.b64encode(image_bytes).decode("ascii")},
        }])])
        convert_notebook(self.notebook_path, self.report_path)
        report = self.report_path.read_text(encoding="utf-8")
        self.assertIn("(<outputs/figures/test_confusion.png>)", report)
        self.assertFalse((figure.parent / "notebook").exists())

    def test_rejects_error_output_but_accepts_empty_unexecuted_cell(self):
        self.write_notebook([self.code_cell(
            [{"output_type": "error", "ename": "ValueError", "evalue": "invalid"}]
        )])
        with self.assertRaisesRegex(ValueError, "output lỗi"):
            convert_notebook(self.notebook_path, self.report_path)
        self.assertFalse(self.report_path.exists())

        self.write_notebook([self.code_cell([], source=" \n", count=None)])
        convert_notebook(self.notebook_path, self.report_path)
        self.assertEqual(self.report_path.read_text(encoding="utf-8"), "\n")


if __name__ == "__main__":
    unittest.main()

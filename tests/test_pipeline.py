import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _common import parse_voc, stable_fraction, write_csv  # noqa: E402
from convert_voc_to_yolo import convert  # noqa: E402


class PipelineTests(unittest.TestCase):
    def test_voc_conversion_uses_actual_dimensions(self):
        self.assertEqual(convert((100, 50, 500, 250), 1000, 500), (0.3, 0.3, 0.4, 0.4))

    def test_invalid_voc_box_fails(self):
        with self.assertRaises(ValueError):
            convert((20, 20, 10, 30), 100, 100)

    def test_parse_voc(self):
        content = """<annotation><size><width>800</width><height>700</height></size>
        <object><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>3</xmax><ymax>4</ymax></bndbox></object></annotation>"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.xml"; path.write_text(content)
            self.assertEqual(parse_voc(path), (800, 700, [(1.0, 2.0, 3.0, 4.0)]))

    def test_stable_fraction_is_deterministic(self):
        self.assertEqual(stable_fraction("scene", 42), stable_fraction("scene", 42))
        self.assertNotEqual(stable_fraction("scene", 42), stable_fraction("scene", 43))

    def test_write_csv_header_for_empty_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.csv"; write_csv(path, [], ("image_id",))
            with path.open() as handle:
                self.assertEqual(list(csv.reader(handle)), [["image_id"]])


if __name__ == "__main__":
    unittest.main()

"""Read-only report integration on synthetic files; no GPU or unlock data."""
import json
from pathlib import Path
import tempfile
import unittest

from test_factorial_projection import pair

ROOT=Path(__file__).resolve().parents[1]


class FactorialProjectionCliTests(unittest.TestCase):
    def test_cli_accepts_stock_only_for_an_explicit_pending_report(self):
        from experiments import parser
        args=parser().parse_args(["factorial","--stock","results/stock.json"])
        self.assertIsNone(args.unlocked)

    def test_pending_report_preserves_raw_input_and_does_not_invent_unlocked_cells(self):
        from factorial_results import report
        stock,_=pair()
        with tempfile.TemporaryDirectory(dir=ROOT/".scratch") as directory:
            root=Path(directory)
            source=root/"synthetic-stock.json"
            source.write_text(json.dumps(stock))
            original=source.read_bytes()
            output=root/"pending.md"
            rows=report(source,None,output)
            self.assertEqual(source.read_bytes(),original)
            self.assertIsNone(rows[0]["unlocked_vendor_us"])
            self.assertIsNone(rows[0]["combined_gain_pct"])
            self.assertIn("pending",output.read_text().lower())
            self.assertIn(stock["cases"][0]["weight_tensor"],output.read_text())

    def test_reporting_cannot_overwrite_raw_evidence_or_existing_output(self):
        from factorial_results import report
        stock,unlocked=pair()
        with tempfile.TemporaryDirectory(dir=ROOT/".scratch") as directory:
            root=Path(directory)
            before,after=root/"stock.json",root/"unlocked.json"
            before.write_text(json.dumps(stock))
            after.write_text(json.dumps(unlocked))
            originals={path:path.read_bytes() for path in (before,after)}
            for target in (before,after):
                with self.assertRaises(ValueError):
                    report(before,after,target)
                self.assertEqual(target.read_bytes(),originals[target])
            output=root/"existing.md"
            output.write_text("preserved")
            with self.assertRaises(ValueError):
                report(before,after,output)
            self.assertEqual(output.read_text(),"preserved")

    def test_synthetic_comparison_is_explicitly_labelled_as_no_hardware_result(self):
        from factorial_results import report
        stock,unlocked=pair()
        with tempfile.TemporaryDirectory(dir=ROOT/".scratch") as directory:
            root=Path(directory)
            before,after=root/"stock.json",root/"unlocked.json"
            before.write_text(json.dumps(stock))
            after.write_text(json.dumps(unlocked))
            output=root/"synthetic.md"
            report(before,after,output)
            self.assertIn("synthetic fixture",output.read_text().lower())


if __name__=="__main__":
    unittest.main()

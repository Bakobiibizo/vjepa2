import hashlib, importlib.util, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[2]
def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py"); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
class DownloadTests(unittest.TestCase):
    def setUp(self): self.module = load_script("download_checkpoint")
    def test_rejects_http(self):
        with self.assertRaisesRegex(ValueError, "HTTPS"): self.module.download("http://example.test/a", Path("a"))
    def test_reuses_verified_cache(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"m.pt"; p.write_bytes(b"cached"); digest=hashlib.sha256(b"cached").hexdigest()
            with patch.object(self.module, "urlopen") as mocked: self.assertEqual(self.module.download("https://example.test/m", p, digest), p.resolve()); mocked.assert_not_called()
    def test_rejects_bad_cache(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"m.pt"; p.write_bytes(b"wrong")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"): self.module.download("https://example.test/m", p, "0"*64)
class EnvironmentTests(unittest.TestCase):
    def test_report_contract(self):
        report=load_script("check_environment").inspect_environment(); self.assertIn("ready", report); self.assertIn("accelerator", report); self.assertEqual(set(report["modules"]), {"torch","torchvision","yaml","numpy"})
if __name__ == "__main__": unittest.main()

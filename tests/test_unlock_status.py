import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gpu_unlock_status

ROOT=Path(__file__).resolve().parents[1]


class UnlockInventoryTests(unittest.TestCase):
    def test_actual_device_and_power_snapshot_without_sudo(self):
        with tempfile.TemporaryDirectory(dir=ROOT/".scratch") as directory:
            devices=Path(directory)
            card=devices/"0000:03:00.0"
            (card/"power").mkdir(parents=True)
            for name,value in (("vendor","0x10de"),("class","0x030000"),("device","0xffff"),
                               ("power/runtime_status","suspended"),("power/control","auto")):
                (card/name).write_text(value)
            calls=[]
            def command(args):
                calls.append(args)
                if args[0]=="nvidia-smi":
                    (card/"power/runtime_status").write_text("active")
                    return "test GPU"
                return None
            with patch.object(gpu_unlock_status,"command",side_effect=command):
                result=gpu_unlock_status.inspect(sysfs_root=devices)
            self.assertEqual(result["pci_address"],"0000:03:00.0")
            self.assertFalse(result["device_allowlisted"])
            self.assertEqual(result["runtime_status"],"suspended")
            self.assertIsNone(result["driver_bound"])
            self.assertFalse(any(args[0]=="sudo" for args in calls))

    def test_invalid_pci_path_rejected(self):
        with self.assertRaises(ValueError):
            gpu_unlock_status.inspect(pci_address="../../other")

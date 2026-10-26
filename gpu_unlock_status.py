"""Read-only compatibility inventory. This never loads or installs a module."""

import hashlib
import json
import platform
import re
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def command(args):
    try:
        result=subprocess.run(args,capture_output=True,text=True,timeout=20)
        return result.stdout.strip() if result.returncode==0 else None
    except (OSError,subprocess.TimeoutExpired):
        return None


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def inspect(pci_address=None,sysfs_root=Path("/sys/bus/pci/devices")):
    repo=ROOT/"third_party/gpu-unlock"
    module=repo/"kernel/gpu_unlock/gpu_unlock.ko"
    if pci_address is None:
        cards=[p for p in sorted(sysfs_root.iterdir()) if read(p/"vendor")=="0x10de"
               and int(read(p/"class") or "0",16)>>16==3]
        if not cards: raise ValueError("No NVIDIA display/3D device found in sysfs")
        pci_address=cards[0].name
    if not re.fullmatch(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]",pci_address):
        raise ValueError("Expected a full PCI address such as 0000:01:00.0")
    device=sysfs_root/pci_address
    device_id=read(device/"device")
    vendor_id=read(device/"vendor")
    runtime={"runtime_status":read(device/"power/runtime_status"),
             "runtime_control":read(device/"power/control"),"d3cold_allowed":read(device/"d3cold_allowed")}
    driver=(device/"driver").resolve().name if (device/"driver").exists() else None
    supported=re.search(r"supported_ids\[\]\s*=\s*\{(.*?)\};",
                        (repo/"kernel/gpu_unlock/main.c").read_text(),re.S).group(1)
    supported_ids={int(value,16) for value in re.findall(r"0x[0-9a-fA-F]{4}\b",supported)}
    secure_boot=None
    try:
        for path in Path("/sys/firmware/efi/efivars").glob("SecureBoot-*"):
            data=path.read_bytes()
            if len(data)>4: secure_boot=bool(data[4])
    except OSError:
        pass
    params=read("/proc/driver/nvidia/params") or ""
    power={key:next((line.split(":",1)[1].strip() for line in params.splitlines()
                     if line.startswith(key+":")),None)
           for key in ("DynamicPowerManagement","EnableGpuFirmware","UseKernelSuspendNotifiers")}
    manifest=json.loads((repo/"firmware/artifacts-610.43.03.json").read_text())
    gsp=next(asset for entry in manifest["source_artifacts"] for asset in entry.get("firmware",[])
             if asset["path"].endswith("/gsp_ga10x.bin"))
    expected={**manifest["derived_assets"],"gsp_ga10x.bin":gsp["sha256"],
              "v67-ga102-610.43.03.bin":manifest["generated_payload"]["sha256"]}
    assets=ROOT/".cache/gpu-unlock/assets"
    verified={name:(assets/name).is_file() and hashlib.sha256((assets/name).read_bytes()).hexdigest()==digest
              for name,digest in expected.items()}
    journal=command(["journalctl","-k","-b","--grep","gpu_unlock","--no-pager","-o","cat"])
    return {"os":platform.freedesktop_os_release().get("PRETTY_NAME"),"kernel":platform.release(),
            "gpu":command(["nvidia-smi","-i",pci_address,"--query-gpu=name,uuid,pci.bus_id,pci.device_id,driver_version",
                           "--format=csv,noheader"]),
            "pci_address":pci_address,"device_id":device_id,
            "device_allowlisted":vendor_id=="0x10de" and device_id is not None and int(device_id,16) in supported_ids,
            "board_hardware_validation":"not established by allowlisting or compilation",
            "secure_boot_enabled":secure_boot,"driver_bound":driver,**runtime,
            "power_snapshot_scope":"sysfs read before nvidia-smi; does not establish idle behavior",
            "kernel_lockdown":read("/sys/kernel/security/lockdown"),
            "module_signature_enforce":read("/sys/module/module/parameters/sig_enforce"),
            "nvidia_runtime_parameters":power,
            "repo_commit":command(["git","-C",str(repo),"rev-parse","HEAD"]),
            "module_built":module.is_file(),"module_vermagic":command(["modinfo","-F","vermagic",str(module)]),
            "module_loaded":Path("/sys/module/gpu_unlock").exists(),
            "unlock_boot_option_present": "gpu_unlock.targets=" in (read("/proc/cmdline") or ""),
            "unlock_success_logged":bool(journal and "gpu_unlock: unlocked" in journal),
            "post_flr_selectors_logged":bool(journal and re.search(r"post-FLR[^\n]*SS0=88888888[^\n]*SS1=00000008",journal)),
            "journal_readable":journal is not None,
            "firmware_sha256_checks":verified,"this_command_installs_or_loads_modules":False,
            "scope":"Read-only inventory. Does not establish hardware unlock state or hardware safety."}

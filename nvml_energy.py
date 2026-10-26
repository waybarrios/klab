"""Small read-only NVML energy sensor, no Python package installation."""
import ctypes as ct
import time


class SensorUnavailable(RuntimeError):
    pass


class NvmlEnergy:
    def __init__(self, uuid):
        self.lib = ct.CDLL("libnvidia-ml.so.1")
        self.closed = True
        self.lib.nvmlErrorString.argtypes = [ct.c_int]
        self.lib.nvmlErrorString.restype = ct.c_char_p
        self.lib.nvmlDeviceGetHandleByUUID.argtypes = [ct.c_char_p, ct.POINTER(ct.c_void_p)]
        self.lib.nvmlDeviceGetTotalEnergyConsumption.argtypes = [ct.c_void_p, ct.POINTER(ct.c_ulonglong)]
        self.lib.nvmlDeviceGetPowerUsage.argtypes = [ct.c_void_p, ct.POINTER(ct.c_uint)]
        self.check(self.lib.nvmlInit_v2())
        self.closed = False
        self.handle = ct.c_void_p()
        self.uuid = uuid
        try:
            self.check(self.lib.nvmlDeviceGetHandleByUUID(uuid.encode(), ct.byref(self.handle)))
            self.snapshot()
        except BaseException:
            self.close()
            raise

    def check(self, status):
        if status:
            raise SensorUnavailable(self.lib.nvmlErrorString(status).decode())

    def snapshot(self):
        if self.closed:
            raise RuntimeError("NVML sensor is closed")
        started = time.monotonic()
        energy, power = ct.c_ulonglong(), ct.c_uint()
        self.check(self.lib.nvmlDeviceGetTotalEnergyConsumption(self.handle, ct.byref(energy)))
        self.check(self.lib.nvmlDeviceGetPowerUsage(self.handle, ct.byref(power)))
        return {"uuid": self.uuid, "energy_mj": int(energy.value), "power_mw": int(power.value),
                "monotonic_seconds": time.monotonic(), "query_started_seconds": started}

    def close(self):
        if not self.closed:
            self.lib.nvmlShutdown()
            self.closed = True

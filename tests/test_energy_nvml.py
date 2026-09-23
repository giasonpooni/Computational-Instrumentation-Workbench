"""Fake-ABI boundary checks and explicitly enabled real sensor capability checks."""
import ctypes as ct
from hashlib import sha256
import os
from pathlib import Path

import pytest

from ciw import energy_nvml as nvml

UUID = "GPU-12345678-1234-1234-1234-123456789abc"


class Function:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *arguments):
        return self.callback(*arguments)


class FakeABI:
    """Only an ABI test double: values are never hardware measurement evidence."""
    def __init__(self):
        self.init_code = self.select_code = self.shutdown_code = 0
        self.shutdown_count = 0
        self.calls, self.energies = [], [(0,2**63+17)]
        self.context = {"power_mw":(0,17001),"temperature_c":(0,51),"graphics_clock_mhz":(0,300)}
        self.nvmlInit_v2 = Function(lambda:self.init_code)
        self.nvmlShutdown = Function(self.shutdown)
        self.nvmlDeviceGetHandleByIndex_v2 = Function(self.select_index)
        self.nvmlDeviceGetHandleByUUID = Function(self.select_uuid)
        self.nvmlDeviceGetUUID = Function(lambda _h,b,n:self.text(b,n,UUID))
        self.nvmlDeviceGetName = Function(lambda _h,b,n:self.text(b,n,"ABI-test-only-device"))
        self.nvmlSystemGetDriverVersion = Function(lambda b,n:self.text(b,n,"test-driver"))
        self.nvmlSystemGetNVMLVersion = Function(lambda b,n:self.text(b,n,"test-nvml"))
        self.nvmlDeviceGetTotalEnergyConsumption = Function(self.energy)
        self.nvmlDeviceGetPowerUsage = Function(lambda h,p:self.optional("power_mw",p))
        self.nvmlDeviceGetTemperature = Function(lambda h,s,p:self.optional("temperature_c",p))
        self.nvmlDeviceGetClockInfo = Function(lambda h,s,p:self.optional("graphics_clock_mhz",p))

    def shutdown(self):
        self.shutdown_count += 1
        return self.shutdown_code

    def select_index(self,index,pointer):
        self.calls.append(("index",index))
        ct.cast(pointer,ct.POINTER(ct.c_void_p))[0] = 123
        return self.select_code

    def select_uuid(self,uuid,pointer):
        self.calls.append(("uuid",uuid))
        ct.cast(pointer,ct.POINTER(ct.c_void_p))[0] = 123
        return self.select_code

    @staticmethod
    def text(buffer,length,value):
        assert length > len(value)
        buffer.value = value.encode()
        return 0

    def energy(self,handle,pointer):
        self.calls.append("energy")
        code,value = self.energies[0]
        if len(self.energies)>1:
            self.energies.pop(0)
        ct.cast(pointer,ct.POINTER(ct.c_ulonglong))[0] = value
        return code

    def optional(self,key,pointer):
        self.calls.append(key)
        code,value = self.context[key]
        ct.cast(pointer,ct.POINTER(ct.c_uint))[0] = value
        return code


@pytest.fixture
def abi(monkeypatch,tmp_path):
    fake = FakeABI()
    path = tmp_path/"nvml-test-only-library"
    path.write_bytes(b"explicit fake ABI library identity; not executable")
    monkeypatch.setattr(nvml,"_load_nvml",lambda:(fake,path))
    return fake,path


def test_descriptor_binds_actual_selected_identity_and_is_immutable(abi):
    fake,path = abi
    with nvml.NVMLEnergyCounter(device_index=2) as counter:
        expected = {"backend":"nvml","device_uuid":UUID,"name":"ABI-test-only-device",
            "driver_version":"test-driver","nvml_version":"test-nvml","library_sha256":sha256(path.read_bytes()).hexdigest(),
            "counter_unit":"mJ","counter_scope":"gpu_device","power_unit":"mW",
            "timestamp_semantics":"host_call_brackets","update_interval_s":None,"resolution_j":None,
            "accuracy_j":None,"background_inclusive":True}
        assert counter.identity() == expected
        changed = counter.identity()
        changed["accuracy_j"] = 0
        assert counter.identity() == expected
        assert fake.calls == [("index",2)]
    assert fake.shutdown_count == 1
    counter.close()
    assert fake.shutdown_count == 1 and counter.identity() == expected
    with pytest.raises(RuntimeError,match="closed"):
        counter.read()


def test_uuid_selection_uses_uuid_entrypoint_and_checks_returned_device(abi):
    fake,_ = abi
    with nvml.NVMLEnergyCounter(uuid=UUID) as counter:
        assert counter.identity()["device_uuid"] == UUID
    assert fake.calls == [("uuid",UUID.encode())]
    with pytest.raises(nvml.NVMLUnavailable,match="device_identity"):
        nvml.NVMLEnergyCounter(uuid=UUID.replace("12345678","87654321"))
    assert fake.shutdown_count == 2


@pytest.mark.parametrize("arguments", [{"device_index":True},{"device_index":-1},{"device_index":2**32},
    {"device_index":1.0},{"uuid":"../../other.dll"},{"uuid":"MIG-123"},{"uuid":False}])
def test_bad_selection_never_loads_a_library(monkeypatch,arguments):
    monkeypatch.setattr(nvml,"_load_nvml",lambda:pytest.fail("Invalid caller input loaded a library"))
    with pytest.raises(ValueError):
        nvml.NVMLEnergyCounter(**arguments)


def test_energy_brackets_exclude_optional_context_and_preserve_uint64(abi,monkeypatch):
    fake,_ = abi
    with nvml.NVMLEnergyCounter() as counter:
        fake.calls.clear()
        monos,utcs = iter((100,140)),iter((1000,1048))
        def monotonic():
            fake.calls.append("monotonic")
            return next(monos)
        def utc():
            fake.calls.append("utc")
            return next(utcs)
        monkeypatch.setattr(nvml,"perf_counter_ns",monotonic)
        monkeypatch.setattr(nvml,"time_ns",utc)
        result = counter.read()
    assert result == {"read_start_ns":100,"read_end_ns":140,"utc_start_ns":1000,"utc_end_ns":1048,
        "status":"ok","energy_mj":2**63+17,"error_code":None,"power_mw":17001,
        "temperature_c":51,"graphics_clock_mhz":300,"context_errors":{}}
    assert fake.calls == ["utc","monotonic","energy","monotonic","utc","power_mw","temperature_c","graphics_clock_mhz"]


def test_zero_repeated_and_decreasing_counters_remain_raw(abi):
    fake,_ = abi
    fake.energies = [(0,0),(0,0),(0,100),(0,90)]
    with nvml.NVMLEnergyCounter() as counter:
        records = [counter.read() for _ in range(4)]
    assert [r["energy_mj"] for r in records] == [0,0,100,90]
    assert all(r["status"] == "ok" and r["error_code"] is None for r in records)


def test_failed_energy_and_optional_calls_never_substitute_zero(abi):
    fake,_ = abi
    fake.energies = [(15,987654321)]  # A failing ABI may leave garbage in output storage.
    fake.context.update(power_mw=(3,999),temperature_c=(4,123))
    with nvml.NVMLEnergyCounter() as counter:
        record = counter.read()
    assert record["status"] == "error" and record["energy_mj"] is None and record["error_code"] == 15
    assert record["power_mw"] is None and record["temperature_c"] is None
    assert record["graphics_clock_mhz"] == 300
    assert record["context_errors"] == {"power_mw":3,"temperature_c":4}


def test_missing_optional_export_preserves_successful_energy(abi):
    fake,_ = abi
    del fake.nvmlDeviceGetTemperature
    with nvml.NVMLEnergyCounter() as counter:
        record = counter.read()
    assert record["status"] == "ok"
    assert record["temperature_c"] is None and record["context_errors"] == {"temperature_c":13}


@pytest.mark.parametrize("failure,expected_shutdown", [("init",0),("select",1),("identity",1),("energy_export",1)])
def test_unavailable_initialization_and_selection_clean_up_exactly_once(abi,failure,expected_shutdown):
    fake,_ = abi
    if failure == "init": fake.init_code = 9
    elif failure == "select": fake.select_code = 6
    elif failure == "identity": fake.nvmlSystemGetDriverVersion = Function(lambda *_:3)
    else: del fake.nvmlDeviceGetTotalEnergyConsumption
    with pytest.raises(nvml.NVMLUnavailable) as caught:
        nvml.NVMLEnergyCounter()
    assert caught.value.error_code in (3,6,9,13)
    assert fake.shutdown_count == expected_shutdown


def test_shutdown_failure_is_reported_without_hiding_active_failure(abi):
    fake,_ = abi
    fake.shutdown_code = 1
    with pytest.raises(nvml.NVMLUnavailable,match="nvmlShutdown"):
        with nvml.NVMLEnergyCounter(): pass
    with pytest.raises(ValueError,match="primary"):
        with nvml.NVMLEnergyCounter():
            raise ValueError("primary")
    assert fake.shutdown_count == 2


def test_linux_discovery_uses_only_system_nvml_name(monkeypatch,tmp_path):
    library,path = object(),tmp_path/"system-nvml.so"
    path.write_bytes(b"fake loader target")
    seen = []
    monkeypatch.setattr(nvml,"_PLATFORM","linux")
    monkeypatch.setattr(nvml,"find_library",lambda name:seen.append(("find",name)) or "libnvidia-ml.so.1")
    monkeypatch.setattr(nvml.ct,"CDLL",lambda name:seen.append(("load",name)) or library)
    monkeypatch.setattr(nvml,"_linux_library_path",lambda actual:path if actual is library else pytest.fail("Wrong library"))
    assert nvml._load_nvml() == (library,path)
    assert seen == [("find","nvidia-ml"),("load","libnvidia-ml.so.1")]


def test_windows_discovery_uses_only_resolved_system_library(monkeypatch,tmp_path):
    path = tmp_path/"System32"/"nvml.dll"
    path.parent.mkdir()
    path.write_bytes(b"fake loader target")
    library,seen = object(),[]
    monkeypatch.setattr(nvml,"_PLATFORM","win32")
    monkeypatch.setattr(nvml,"_windows_library_path",lambda:path)
    monkeypatch.setattr(nvml.ct,"CDLL",lambda name:seen.append(name) or library)
    assert nvml._load_nvml() == (library,path.resolve())
    assert seen == [str(path.resolve())]


def test_missing_linux_library_is_explicitly_unavailable(monkeypatch):
    monkeypatch.setattr(nvml,"_PLATFORM","linux")
    monkeypatch.setattr(nvml,"find_library",lambda _:None)
    with pytest.raises(nvml.NVMLUnavailable) as caught:
        nvml._load_nvml()
    assert caught.value.error_code == 12


@pytest.mark.skipif(os.environ.get("CIW_ENERGY_GPU") != "1",reason="Actual GPU energy probe requires CIW_ENERGY_GPU=1")
def test_actual_nvml_hardware_reading_without_gpu_workload():
    with nvml.NVMLEnergyCounter() as counter:
        identity = counter.identity()
        readings = [counter.read(),counter.read()]
    assert identity["counter_scope"] == "gpu_device" and identity["background_inclusive"] is True
    assert identity["resolution_j"] is None and identity["accuracy_j"] is None
    assert len(identity["library_sha256"]) == 64
    for reading in readings:
        assert reading["status"] == "ok" and type(reading["energy_mj"]) is int and reading["energy_mj"] >= 0
        assert reading["read_start_ns"] <= reading["read_end_ns"]

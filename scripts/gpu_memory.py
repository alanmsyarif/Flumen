"""Windows process peak working set and total-device VRAM diagnostic."""
import ctypes
from ctypes import wintypes
import subprocess


def memory_snapshot():
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[
            (name,ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize',
            'QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    counters=Counters(); counters.cb=ctypes.sizeof(counters)
    process=ctypes.windll.kernel32.GetCurrentProcess()
    ctypes.windll.psapi.GetProcessMemoryInfo(wintypes.HANDLE(process),ctypes.byref(counters),counters.cb)
    result={'peak_process_mib':counters.PeakWorkingSetSize/(1024**2)}
    try:
        result['whole_device_used_mib']=int(subprocess.check_output(
            ['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],
            text=True,creationflags=subprocess.CREATE_NO_WINDOW).splitlines()[0])
    except (OSError,ValueError,subprocess.SubprocessError):
        result['whole_device_used_mib']=None
    return result

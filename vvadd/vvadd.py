import allo
from allo.ir.types import float32

M, N, K = 128, 128, 128

def vvadd(A: float32[M], B: float32[M]) -> float32[M]:
    C: float32[M] = 0.0
    for i in allo.grid(M):
        C[i] = A[i] + B[i]
    return C

s = allo.customize(vvadd)
print(s.module)

import subprocess
import os

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Run Bash to source the script and print environment variables
bash_command = "bash -c 'source /opt/xilinx/Vitis/2023.2/settings64.sh && env'"
env_vars = subprocess.run(bash_command, shell=True, capture_output=True, text=True, check=True)


executable = s.build(target="llvm")
import numpy as np

np_A = np.random.rand(M).astype(np.float32)
np_B = np.random.rand(M).astype(np.float32)
np_C = executable(np_A, np_B)

golden_C = np.add(np_A, np_B)
np.testing.assert_allclose(np_C, golden_C, rtol=1e-3, atol=1e-3)
print("\033[92mResults are correct! ✅\033[0m")

# Verify
import allo.backend.hls as hls
print(hls.is_available("pynq"))

s.unroll("i", factor=128)

mod = s.build(target="pynq", mode="csyn", project="vvadd.prj", configs={"device":"ultra96v2"})
mod()


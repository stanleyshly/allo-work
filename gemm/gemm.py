import allo
from allo.ir.types import float32

M, N, K = 128, 128, 128

def gemm(A: float32[M, K], B: float32[K, N]) -> float32[M, N]:
    C: float32[M, N] = 0.0
    for i, j in allo.grid(M, N):
        for k in allo.reduction(K):
            C[i, j] += A[i, k] * B[k, j]
    return C

s = allo.customize(gemm)
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

np_A = np.random.rand(M, K).astype(np.float32)
np_B = np.random.rand(K, N).astype(np.float32)
np_C = executable(np_A, np_B)

golden_C = np.matmul(np_A, np_B)
np.testing.assert_allclose(np_C, golden_C, rtol=1e-3, atol=1e-3)
print("\033[92mResults are correct! ✅\033[0m")


# Verify
import allo.backend.hls as hls
print(hls.is_available("pynq"))

s.reorder("k", "j")
s.buffer_at(s.C, axis="i")
s.pipeline("j")
s.unroll("j", factor=128)

mod = s.build(target="pynq", mode="csyn", project="gemm.prj", configs={"device":"ultra96v2"})
mod()


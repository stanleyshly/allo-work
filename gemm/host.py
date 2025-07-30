import argparse, pynq, os, time
import numpy as np

from pynq.buffer import PynqBuffer

def main():
    parser = argparse.ArgumentParser(description="Host Program for GEMM Accelerator")
    parser.add_argument("bitstream", type=str, help="Path to the bitstream file")
    parser.add_argument("size", type=int, help="Size of the vectors to add")
    args = parser.parse_args()
    bitstream_path:str = args.bitstream
    size:int = args.size
    if not os.path.exists(bitstream_path):
        raise FileNotFoundError(f"Cannot find the bitstream file at {bitstream_path}")
    hwh_path = os.path.splitext(bitstream_path)[0] + ".hwh"
    if not os.path.exists(hwh_path):
        raise FileNotFoundError(f"Cannot find the hwh file at {hwh_path}")
    if size <= 0:
        raise ValueError("Size must be greater than 0.")

    print(f"Programming hardware with bitstream {bitstream_path}")
    overlay = pynq.Overlay(bitstream_path)
    
    vvadd_0_hw = overlay.vvadd_0

    vvadd_0_hw.register_map.CTRL.AP_START = 0

    print(f"Running vector-vector add of size {size}")

    # Create the input matrices
    m0:PynqBuffer = pynq.allocate((size, size), dtype=np.float32)
    m1:PynqBuffer = pynq.allocate((size, size), dtype=np.float32)
    m0[:] = np.random.rand((size, size),).astype(np.float32)
    m1[:] = np.random.rand((size, size),).astype(np.float32)

    # Compute the reference output and time it
    start_time = time.perf_counter()
    out_reference = np.matmul(m0, m1)
    end_time = time.perf_counter()
    sw_time_ms = (end_time - start_time) * 1e3
    print(f"Software matmul finished in {sw_time_ms} ms")

    # allocate space for the output
    out:PynqBuffer = pynq.allocate(size, dtype=np.float32)

    start_time = time.perf_counter()

    # Run the hardware accelerator and time it
    # sync input buffers
    m0.sync_to_device()
    m1.sync_to_device()

    # config both vvadd instances
    vvadd_0_hw.register_map.v0_1 = m0.physical_address
    vvadd_0_hw.register_map.v1_1 = m1.physical_address
    vvadd_0_hw.register_map.v2_1 = out.physical_address

    # start both instances
    vvadd_0_hw.register_map.CTRL.AP_START = 1


    out.sync_from_device()
    end_time = time.perf_counter()
    hw_time_ms = (end_time - start_time) * 1e3

    print("Speedup " + str(hw_time_ms/sw_time_ms) +"x")

    if np.allclose(out, out_reference):
        print("Hardware outputs match")
    else:
        print("Mismatch")

if __name__ == "__main__":
    main()
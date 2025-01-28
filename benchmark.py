# measure_bw.py
import torch
import time

def measure_gpu_bandwidth(size_mb=1024, repeats=10):
    """
    Measure approximate GPU memory bandwidth by transferring large tensors
    between CPU and GPU repeatedly.

    :param size_mb: Size of the test tensor in MB
    :param repeats: Number of transfers to average over
    :return: Approximate bandwidth in GB/s
    """
    # Ensure we are using GPU
    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot measure GPU bandwidth.")
        return 0.0

    device = torch.device("cuda:0")
    size_bytes = size_mb * 1024 * 1024  # MB to bytes

    # Create a random tensor on CPU
    # Make it pinned memory (for faster transfers) if desired
    cpu_tensor = torch.empty(size_mb * 256 * 1024, dtype=torch.float32, pin_memory=True)

    # Warm up GPU
    warmup_gpu_tensor = cpu_tensor.to(device)
    del warmup_gpu_tensor
    torch.cuda.synchronize()

    # Record total time for repeated transfers CPU -> GPU -> CPU
    start = time.perf_counter()
    for _ in range(repeats):
        gpu_tensor = cpu_tensor.to(device, non_blocking=True)
        # Optionally do something on GPU to simulate usage
        gpu_tensor.add_(1.0)
        # Transfer back to CPU
        back_cpu_tensor = gpu_tensor.to("cpu", non_blocking=True)

    torch.cuda.synchronize()
    end = time.perf_counter()

    total_time = end - start

    # Each repeat transfers 2x size_mb (CPU->GPU + GPU->CPU).
    # The total data transfer is size_mb * 2 * repeats in MB.
    total_transfer_gb = (size_mb * 2.0 * repeats) / 1024.0  # MB to GB
    bw_gb_s = total_transfer_gb / total_time

    return bw_gb_s


if __name__ == "__main__":
    size_mb = 2048  # Test with a 2GB transfer
    repeats = 5
    bw = measure_gpu_bandwidth(size_mb, repeats)
    if bw > 0:
        print(f"Measured GPU Memory Bandwidth: ~{bw:.2f} GB/s")
    else:
        print("No GPU found or test did not complete.")

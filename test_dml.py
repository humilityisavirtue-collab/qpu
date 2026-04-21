import torch_directml as dml
import torch

dev = dml.device()
print('DirectML device:', dev)
print('Device name:', dml.device_name(dml.default_device()))

# GF(4) XOR on iGPU -- GF(4) ADD = bitwise XOR
a = torch.tensor([[1,2,3,0],[1,0,3,2]], dtype=torch.int32, device=dev)
b = torch.tensor([[1,1,1,1],[0,1,2,3]], dtype=torch.int32, device=dev)
result = (a ^ b).cpu().tolist()
print('XOR test:', result)
print('Expected: [[0,3,2,1],[1,1,1,1]]')
ok = result == [[0,3,2,1],[1,1,1,1]]
print('PASS' if ok else 'FAIL')

# Bigger batch: 10K x 32 GF(4) elements
import time
batch = torch.randint(0, 4, (10000, 32), dtype=torch.int32, device=dev)
peg   = torch.randint(0, 4, (10000, 32), dtype=torch.int32, device=dev)

# warmup
for _ in range(3):
    _ = (batch ^ peg)

t0 = time.perf_counter()
for _ in range(100):
    out = (batch ^ peg)
elapsed = time.perf_counter() - t0

tok_s = 10000 * 32 * 100 / elapsed
print(f'\n10K x 32 XOR x 100 iters: {tok_s:,.0f} tok/s on iGPU')
print(f'({elapsed*1000:.1f} ms total)')

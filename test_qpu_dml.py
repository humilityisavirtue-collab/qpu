import sys
sys.path.insert(0, 'src')

from qpu.compiler import detect_device, QASMtoGPU
from qpu.core import QAssembler

dev, label = detect_device()
print(f'Backend: {label}')
print(f'Device:  {dev}')

# Full QPU compile + run on iGPU
asm = QASMtoGPU()
print(f'Compiler device: {asm.device} ({asm.backend})')

assembler = QAssembler()

# Plinko inference program (4 layers, no branches -- GPU-compilable)
import numpy as np
PLINKO = """
LOAD    r1, [0x100]
ADD     r0, r1
FMIX    r0, 7, 0
LOAD    r1, [0x120]
ADD     r0, r1
FMIX    r0, 7, 1
LOAD    r1, [0x140]
ADD     r0, r1
FMIX    r0, 7, 2
LOAD    r1, [0x160]
ADD     r0, r1
FMIX    r0, 7, 3
HALT
"""

program = assembler.assemble(PLINKO)
np.random.seed(42)
mem_init = {0x100 + i*0x20: np.random.randint(0,4,32).tolist() for i in range(4)}
gpu_fn = asm.compile(program, mem_init)

import torch, time

for batch_size in [1, 1_000, 10_000, 100_000]:
    inputs = torch.randint(0, 4, (batch_size, 32), dtype=torch.int32, device=asm.device)
    for _ in range(3): gpu_fn(inputs)  # warmup

    t0 = time.perf_counter()
    for _ in range(50): gpu_fn(inputs)
    elapsed = (time.perf_counter() - t0) / 50

    tok_s = batch_size * 32 / elapsed
    print(f'  batch={batch_size:>7,}: {tok_s:>14,.0f} tok/s')

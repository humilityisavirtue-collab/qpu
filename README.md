# QPU — Quaternary Processing Unit

A co-processor accelerator using GF(4) arithmetic. 4.5-16.5x faster than gcc -O2 for hashing, crypto, pattern matching, and quantized inference.

## Install

```bash
pip install qpu            # CPU only
pip install qpu[gpu]       # With GPU acceleration (requires CUDA)
```

## Quick Start

```python
import qpu

# Run QASM assembly
result = qpu.run('''
    SET r0, 1
    SET r1, 2
    ADD r0, r1      ; GF(4) XOR — 32 elements at once
    MUL r0, r1      ; GF(4) multiply — 16-byte table lookup
    HALT
''')
print(result.regs[0][:4])  # [2, 2, 2, 2]
print(f"Cycles: {result.cycles}")

# Transpile C to QASM
qasm = qpu.from_c('''
    int main() {
        int a = 1;
        int b = 2;
        int c = a * b;
        return c;
    }
''')
print(qasm)

# GPU-accelerated execution
result = qpu.run_gpu('''
    SET r0, 1
    FMIX r0, 3, 0
    FMIX r0, 3, 1
    FMIX r0, 3, 2
    FMIX r0, 3, 3
    HALT
''', batch_size=10000)

# Benchmark
qpu.benchmark()
```

## CLI

```bash
qpu run program.qasm       # Execute QASM file
qpu compile program.c      # Transpile C to QASM
qpu bench                  # Run benchmark suite
qpu info                   # Show ISA details
```

## ISA (33 instructions)

| Group | Instructions |
|-------|-------------|
| Core ALU | ADD, MUL, FOLD, UNFOLD, BIND, ROUTE |
| Control Flow | JMP, JZ, JNZ, JC, JNC, CALL, RET |
| Memory | LOAD, STORE, VLOAD, VSTORE |
| Arithmetic | CADD, CSUB, CMUL, SHL, SHR, CMP |
| System | SET, ZTEST, FMIX, SYSCALL, HALT |
| Virtual Memory | MMUEN, MMUDIS, MMAP, MUNMAP, MALLOC |

## Benchmarks (RTX 5070 vs gcc -O2)

| Workload | C (gcc -O2) | QPU (GPU) | Speedup |
|----------|------------|-----------|---------|
| Vector XOR | 4.5B ops/s | 8.3B ops/s | 1.8x |
| GF(4) multiply | 814M ops/s | 3.6B ops/s | 4.5x |
| Feistel hash | 204M ops/s | 3.4B ops/s | 16.5x |

## How it works

GF(4) is a finite field with 4 elements {0, 1, 2, 3}. Addition is XOR. Multiplication is a 16-byte lookup table that fits in one cache line and never misses L1. Every QPU instruction operates on 32 elements simultaneously (SIMD-native).

The QPU runs as a co-processor: the host OS handles files, networking, and display through a syscall bridge. The QPU handles the compute-intensive inner loops.

## License

MIT

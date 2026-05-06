# QPU-Lite

The demo surface of the **Quaternary Processing Unit** project. A small SIMD instruction set over GF(4) — the Galois field with four elements {0, 1, 2, 3} — packaged for `pip install`. Addition is XOR. Multiplication is a 16-byte cache-resident lookup. Includes the ISA, a Python emulator, a C-subset → QASM transpiler, and a QASM → PyTorch GPU compiler.

The full QPU stack — including the CUDA GF(4) attention-similarity kernel, the QuaternaryLinear / QuaternaryMatmul transformer integration layer, the gcc-vs-numpy-vs-QPU benchmark harness, and the papers — lives in the parent project. This package is what you can `pip install` and play with today.

## Install

```bash
pip install qpulite            # CPU emulator only
pip install qpulite[gpu]       # With GPU compiler (requires PyTorch + CUDA)
```

The Python import name is still `qpu`:

```python
import qpu
qpu.run("SET r0, 1\nHALT")
```

## Scope

**What this is:**

- A 33-instruction SIMD ISA (QASM) with native GF(4) arithmetic.
- A CPU emulator (Python + NumPy).
- A QASM → PyTorch tensor-op compiler that runs the same programs on a GPU at >10⁹ ops/sec.
- A C-subset → QASM transpiler, with a small built-in benchmark harness.

**What this isn't:**

- A general-purpose CPU accelerator. The CPU emulator is a Python interpreter; it's a correctness reference, not a speed-of-light path.
- A drop-in replacement for transformer attention. A separate GF(4) attention-similarity CUDA kernel lives at `cell/positronic/qpu_kernels.py` (see "Related" below); on attention workloads it correlates ~0.50 with standard cuBLAS attention, so it's suitable for routing, MoE gating, and retrieval scoring — not for full-precision attention.
- A floating-point engine. Everything is 2-bit GF(4).

## Quick Start

```python
import qpu

# 1. Run QASM directly
result = qpu.run('''
    SET r0, 1
    SET r1, 2
    ADD r0, r1      ; GF(4) XOR — 32 elements at once
    MUL r0, r1      ; GF(4) multiply — 16-byte table lookup
    HALT
''')
print(result.regs[0][:4])
print(f"Cycles: {result.cycles}")

# 2. Transpile a small C subset to QASM
qasm = qpu.from_c('''
    int main() {
        int a = 1;
        int b = 2;
        int c = a * b;
        return c;
    }
''')
print(qasm)

# 3. Compile QASM and run on GPU
result = qpu.run_gpu('''
    SET r0, 1
    FMIX r0, 3, 0
    FMIX r0, 3, 1
    FMIX r0, 3, 2
    FMIX r0, 3, 3
    HALT
''', batch_size=10000)

# 4. Run the built-in benchmark suite
qpu.benchmark()
```

## CLI

```bash
qpu run program.qasm       # Execute a QASM file on the CPU emulator
qpu compile program.c      # Transpile C → QASM and print the result
qpu bench                  # Run the built-in benchmark suite
qpu info                   # Show ISA details and register layout
```

## ISA

| Group | Instructions |
|---|---|
| Core ALU | ADD, MUL, FOLD, UNFOLD, BIND, ROUTE |
| Control flow | JMP, JZ, JNZ, JC, JNC, CALL, RET |
| Memory | LOAD, STORE, VLOAD, VSTORE |
| Arithmetic | CADD, CSUB, CMUL, SHL, SHR, CMP |
| System | SET, ZTEST, FMIX, SYSCALL, HALT |
| Virtual memory | MMUEN, MMUDIS, MMAP, MUNMAP, MALLOC |

**Layout:** 8 vector registers × 32 GF(4) elements (64 bits per register). 4096 GF(4) elements of memory by default.

## Benchmarks

Measured on **RTX 5080 Laptop GPU, CUDA 12.8, PyTorch 2.10.0+cu128**, 2026-05-03.

Full output: `cell/qpu/bench/results/c2qasm_harness_20260503.txt` (in the parent repo). All numbers below come from a single run of `python apps/plinko-transformer/c2qasm.py --bench`.

| Workload | NumPy reference | QPU CPU emulator | QPU GPU (compiled) | GPU vs NumPy |
|---|---|---|---|---|
| vector_xor (1M elements, add-heavy) | 1.52 B ops/s | 8.67 M ops/s | 11.25 B ops/s | **7.4×** |
| mul_heavy (1024-element multiply-accumulate) | 6.24 M ops/s | 9.00 M ops/s | 5.21 B ops/s | **836×** |
| feistel_hash (8-round mixed ops) | 485 K rounds/s | 6.67 M ops/s | 5.36 B ops/s | **see note** |

**Notes:**

- Comparison baseline is **NumPy**, not gcc -O2. The harness can drive a `gcc -O2` C build of the same program, but no MSYS2/mingw toolchain was installed on the measurement machine, so that column reads "N/A" in the receipts. If you want a hard-C baseline, install MSYS2 and re-run.
- The CPU emulator path is a Python interpreter loop; it's slower than NumPy for primitive ops but faster than NumPy on mul-heavy and feistel because it routes everything through the 16-byte GF(4) table that stays in L1.
- The Feistel comparison crosses units (NumPy column is rounds/s, QPU columns are element-XOR ops/s). On a per-round basis the honest GPU/NumPy speedup ranges roughly **5×–670×** by workload. Don't multiply the raw-ops Feistel number by the rounds-second number directly.

### Attention-similarity kernel (separate from this package)

A CUDA GF(4) attention-similarity kernel lives at `cell/positronic/qpu_kernels.py`. On the same machine (`cell/qpu/bench/results/qpu_bench_results.json`, B=4 / seq=128 / d_model=2560):

| Path | Time | vs cuBLAS bmm |
|---|---|---|
| cuBLAS bmm (baseline) | 0.180 ms | 1.00× |
| CUDA GF(4) full pipeline (quantize + pack + score) | 0.606 ms | 0.30× |
| **CUDA GF(4) amortized** (kernel only, weights pre-packed) | **0.066 ms** | **2.74×** |

Bit-exact match against the PyTorch reference. **Correlation with standard attention scores: 0.504** — about half. Use for routing / MoE gating / retrieval scoring, not as a drop-in attention replacement.

A dimension sweep across seq_len ∈ {128, 256, 512, 1024, 2048} × d_model ∈ {1024, 2560, 4096, 8192} is in `qpu_dim_sweep_20260503_014852.json`. Amortized speedup ranges 0.89×–2.06×; the full pipeline never beats cuBLAS.

## How it works

GF(4) is the finite field with four elements {0, 1, 2, 3}. Two operations:

- **Addition = XOR.** Single CPU cycle, no lookup needed.
- **Multiplication = 16-byte table lookup.** The table fits in one cache line and stays in L1.

Every QPU instruction operates on 32 GF(4) elements at once (one 64-bit word). The host OS handles I/O — the QPU is a pure compute substrate.

## Related

This sits in the **Hyperdimensional Computing / Vector Symbolic Architecture** lineage: Kanerva (1988, *Sparse Distributed Memory*), Plate (1995, *Holographic Reduced Representations*), Gayler (2003, *Vector Symbolic Architectures Answer Jackendoff's Challenges*). What's distinctive here:

- Explicit **GF(4)** substrate (vs. binary HDC).
- An **ISA-level abstraction** with a transpiler from a familiar source language.
- Runs on **commodity GPUs** through standard PyTorch — no specialized neuromorphic hardware required.

## License

MIT

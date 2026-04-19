"""
QPU high-level API.

Three entry points:
    qpu.run(source)   — execute QASM assembly, return QPU state
    qpu.jit(fn)       — decorator: compile Python function to GPU-accelerated QASM
    qpu.from_c(source) — transpile C to QASM assembly string
    qpu.benchmark()   — run the standard benchmark suite
"""

import numpy as np
import time
import functools
from typing import Optional, Callable
from .core import QPU, QAssembler, execute
from .gf4 import ADD, MUL


class QPUResult:
    """Result of a QPU execution."""
    __slots__ = ('regs', 'mem', 'cycles', 'zero_flag', 'carry_flag')

    def __init__(self, qpu: QPU):
        self.regs = qpu.regs.copy()
        self.mem = qpu.mem.copy()
        self.cycles = qpu.cycles
        self.zero_flag = qpu.zero_flag
        self.carry_flag = getattr(qpu, 'carry_flag', False)

    def reg(self, n: int) -> np.ndarray:
        """Get register n as numpy array."""
        return self.regs[n]

    def reg_int(self, n: int) -> int:
        """Get register n as base-4 integer (LSB first)."""
        val = 0
        for i in range(len(self.regs[n]) - 1, -1, -1):
            val = val * 4 + int(self.regs[n][i])
        return val

    def __repr__(self):
        return f"QPUResult(cycles={self.cycles}, r0={self.regs[0][:8]}...)"


def run(source: str, mem_init: dict = None, max_cycles: int = 100000) -> QPUResult:
    """Execute QASM assembly source on the QPU emulator.

    Args:
        source: QASM assembly text
        mem_init: optional {address: np.array} to preload memory
        max_cycles: execution limit

    Returns:
        QPUResult with register/memory state after execution
    """
    asm = QAssembler()
    program = asm.assemble(source)
    qpu = QPU()

    if mem_init:
        for addr, data in mem_init.items():
            n = min(len(data), len(qpu.mem) - addr)
            qpu.mem[addr:addr + n] = np.array(data[:n], dtype=np.uint8)

    execute(qpu, program, max_cycles=max_cycles)
    return QPUResult(qpu)


def run_gpu(source: str, batch_size: int = 10000, device: str = 'cuda') -> QPUResult:
    """Execute QASM on GPU via compiled tensor ops.

    Args:
        source: QASM assembly text
        batch_size: number of parallel QPU instances
        device: 'cuda' or 'cpu'

    Returns:
        QPUResult from first batch element
    """
    try:
        import torch
        from .compiler import QASMtoGPU
    except ImportError:
        raise ImportError("GPU execution requires PyTorch: pip install torch")

    asm = QAssembler()
    program = asm.assemble(source)

    compiler = QASMtoGPU(device=device)
    gpu_fn = compiler.compile(program)

    batch_input = torch.randint(0, 4, (batch_size, QPU.WORD_SIZE),
                                dtype=torch.int32, device=device)
    output = gpu_fn(batch_input)

    # Build result from first batch element
    qpu = QPU()
    if hasattr(output, 'cpu'):
        qpu.regs[0] = output[0].cpu().numpy().astype(np.uint8)
    return QPUResult(qpu)


def from_c(source: str) -> str:
    """Transpile C source code to QASM assembly.

    Supports a subset of C: int variables, +, *, ^, &, if/else, while, for.
    All values map to GF(4): + becomes XOR, * becomes table lookup.

    Args:
        source: C source code string

    Returns:
        QASM assembly string
    """
    from .transpiler import C2QASM
    transpiler = C2QASM()
    return transpiler.compile(source)


def jit(fn: Optional[Callable] = None, *, device: str = 'cuda', batch_size: int = 10000):
    """JIT decorator: compile a Python function's QASM body to GPU.

    Usage:
        @qpu.jit
        def my_hash():
            return '''
                SET r0, 1
                SET r1, 2
                FMIX r0, 3, 0
                FMIX r0, 3, 1
                HALT
            '''

        result = my_hash()  # runs on GPU
    """
    def decorator(func):
        _compiled = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            source = func(*args, **kwargs)
            if not isinstance(source, str):
                raise TypeError("@qpu.jit function must return a QASM source string")

            cache_key = source
            if cache_key not in _compiled:
                try:
                    import torch
                    from .compiler import QASMtoGPU
                    asm = QAssembler()
                    program = asm.assemble(source)
                    compiler = QASMtoGPU(device=device)
                    _compiled[cache_key] = {
                        'gpu_fn': compiler.compile(program),
                        'program': program,
                        'device': device,
                        'batch_size': batch_size,
                    }
                except ImportError:
                    # Fallback to CPU emulator
                    asm = QAssembler()
                    _compiled[cache_key] = {
                        'gpu_fn': None,
                        'program': asm.assemble(source),
                    }

            entry = _compiled[cache_key]
            if entry['gpu_fn'] is not None:
                import torch
                batch_input = torch.randint(
                    0, 4, (batch_size, QPU.WORD_SIZE),
                    dtype=torch.int32, device=device
                )
                output = entry['gpu_fn'](batch_input)
                qpu = QPU()
                qpu.regs[0] = output[0].cpu().numpy().astype(np.uint8)
                return QPUResult(qpu)
            else:
                qpu = QPU()
                execute(qpu, entry['program'], max_cycles=100000)
                return QPUResult(qpu)

        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


def benchmark(include_gpu: bool = True):
    """Run the standard QPU benchmark suite.

    Tests: vector_xor, mul_heavy, feistel_hash
    Compares: Python baseline vs QPU emulator vs QPU GPU
    """
    asm = QAssembler()

    benchmarks = {
        'vector_xor': {
            'source': '\n'.join(['SET r0, 1', 'SET r1, 2', 'SET r2, 0'] +
                                ['ADD r2, r0', 'ADD r2, r1'] * 16 + ['HALT']),
            'iters': 10000,
        },
        'mul_heavy': {
            'source': '\n'.join(['SET r0, 1', 'SET r1, 2', 'SET r2, 1'] +
                                ['MUL r2, r0', 'MUL r2, r1'] * 16 +
                                ['FOLD r2, 2', 'HALT']),
            'iters': 10000,
        },
        'feistel_8round': {
            'source': '\n'.join(['SET r0, 1', 'SET r1, 2'] +
                                [f'FMIX r0, 3, {i}' for i in range(8)] + ['HALT']),
            'iters': 10000,
        },
    }

    print("QPU Benchmark Suite v0.1")
    print("=" * 60)

    results = {}
    for name, bench in benchmarks.items():
        program = asm.assemble(bench['source'])
        qpu = QPU()

        # Count ops per run
        qpu.reset()
        execute(qpu, program, max_cycles=100000)
        ops_per_run = qpu.cycles * QPU.WORD_SIZE

        # CPU emulator benchmark
        start = time.perf_counter()
        for _ in range(bench['iters']):
            qpu.reset()
            execute(qpu, program, max_cycles=100000)
        elapsed = time.perf_counter() - start
        cpu_ops = ops_per_run * bench['iters'] / elapsed

        gpu_ops = None
        if include_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    from .compiler import QASMtoGPU
                    compiler = QASMtoGPU(device='cuda')
                    gpu_fn = compiler.compile(program)
                    batch = torch.randint(0, 4, (10000, QPU.WORD_SIZE),
                                          dtype=torch.int32, device='cuda')
                    # Warmup
                    for _ in range(3):
                        gpu_fn(batch)
                    torch.cuda.synchronize()

                    start = time.perf_counter()
                    for _ in range(100):
                        gpu_fn(batch)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    gpu_ops = ops_per_run * 10000 * 100 / elapsed
            except ImportError:
                pass

        results[name] = {'cpu': cpu_ops, 'gpu': gpu_ops}
        cpu_str = f"{cpu_ops:,.0f}"
        gpu_str = f"{gpu_ops:,.0f}" if gpu_ops else "N/A"
        ratio = f"{gpu_ops/cpu_ops:.0f}x" if gpu_ops else "-"
        print(f"  {name:<20} CPU: {cpu_str:<16} GPU: {gpu_str:<16} GPU/CPU: {ratio}")

    print()
    return results

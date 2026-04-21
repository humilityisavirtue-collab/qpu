"""
QPU — Quaternary Processing Unit

A co-processor accelerator using GF(4) arithmetic.
4.5-16.5x faster than gcc -O2 for hashing, crypto, and quantized inference.

Usage:
    import qpu

    # Direct QASM assembly
    result = qpu.run('''
        SET r0, 1
        SET r1, 2
        ADD r0, r1
        HALT
    ''')
    print(result.regs[0][:4])  # [3, 3, 3, 3]

    # JIT decorator — compile Python to QASM, run on GPU
    @qpu.jit
    def feistel_hash(data):
        qpu.fmix(data, generator=3, rounds=8)
        return data

    # C transpilation
    qasm_code = qpu.from_c('''
        int main() {
            int a = 1;
            int b = 2;
            int c = a + b;
            return c;
        }
    ''')
"""

__version__ = "0.1.0"

from .core import QPU, QAssembler, execute
from .gf4 import ADD, MUL, INV, add_vec, mul_vec
from .compiler import QASMtoGPU
from .bootstrap import (
    encode_program, decode_bytecode, execute_bytecode,
    OPCODE_TABLE, OPCODE_TO_NUM
)
from .transpiler import C2QASM
from .api import run, jit, from_c, benchmark
from .agent import QPUAgentLoop, AgentResult, run_agent

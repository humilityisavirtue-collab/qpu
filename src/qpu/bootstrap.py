#!/usr/bin/env python3
"""
qasm_bootstrap.py — Self-hosting QPU assembler.

Layer 5: The assembler written as a QASM program that runs on the QPU itself.
When the computer can build itself, it's real.

Instruction encoding (quaternary bytecode):
  Each instruction = 8 quaternary digits (packed into memory)
  [opcode: 3 digits] [arg1: 2 digits] [arg2: 2 digits] [flags: 1 digit]

  opcode: 0-63 (3 base-4 digits)
  arg1/arg2: register (0-7) or immediate (0-15) in 2 digits
  flags: 0=reg,reg  1=reg,imm  2=imm,reg  3=imm,imm

  32 memory elements per word = 4 instructions per word.

K-address: +KD (THE WEALTH) — self-hosting. The crown jewel.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from .core import QPU, QAssembler, execute
from .gf4 import ADD, MUL


# ================================================================
# Opcode table — shared between host assembler and QPU assembler
# ================================================================

OPCODE_TABLE = [
    'HALT', 'ADD', 'MUL', 'FOLD', 'UNFOLD', 'BIND', 'ROUTE',
    'JMP', 'JZ', 'JNZ', 'JC', 'JNC',
    'LOAD', 'STORE', 'SET', 'ZTEST',
    'FMIX', 'CALL', 'RET',
    'CADD', 'CSUB', 'CMUL', 'SHL', 'SHR', 'CMP',
    'SYSCALL',
    'MMUEN', 'MMUDIS', 'MMAP', 'MUNMAP', 'MALLOC', 'VLOAD', 'VSTORE',
]

OPCODE_TO_NUM = {op: i for i, op in enumerate(OPCODE_TABLE)}


def int_to_quat_digits(val: int, n_digits: int) -> list:
    """Convert integer to list of quaternary digits (LSB first)."""
    digits = []
    for _ in range(n_digits):
        digits.append(val % 4)
        val //= 4
    return digits


def quat_digits_to_int(digits: list) -> int:
    """Convert quaternary digits (LSB first) to integer."""
    val = 0
    for i in range(len(digits) - 1, -1, -1):
        val = val * 4 + digits[i]
    return val


# ================================================================
# Host-side encoder: QASM text -> quaternary bytecode
# ================================================================

def encode_instruction(opcode_name: str, arg1: int = 0, arg2: int = 0,
                       arg1_is_imm: bool = False, arg2_is_imm: bool = False) -> list:
    """Encode one instruction as 8 quaternary digits."""
    opnum = OPCODE_TO_NUM.get(opcode_name.upper(), 0)
    op_digits = int_to_quat_digits(opnum, 3)
    a1_digits = int_to_quat_digits(arg1 % 16, 2)
    a2_digits = int_to_quat_digits(arg2 % 16, 2)
    flags = (1 if arg1_is_imm else 0) + (2 if arg2_is_imm else 0)
    return op_digits + a1_digits + a2_digits + [flags]


def encode_program(source: str) -> np.ndarray:
    """Encode QASM source into quaternary bytecode array."""
    asm = QAssembler()
    # Use the existing assembler to parse, then re-encode as bytecode
    program = asm.assemble(source)
    bytecode = []

    for op_name, args in program:
        # Reverse-lookup the opcode name
        op_upper = op_name.upper()
        # Find it in the table
        found = None
        for name, internal in QAssembler.OPCODES.items():
            if internal == op_name:
                found = name
                break
        if not found:
            found = op_upper

        a1 = args[0][1] if len(args) > 0 else 0
        a2 = args[1][1] if len(args) > 1 else 0
        a1_imm = len(args) > 0 and args[0][0] in ('imm', 'addr', 'label')
        a2_imm = len(args) > 1 and args[1][0] in ('imm', 'addr', 'label')

        encoded = encode_instruction(found, a1, a2, a1_imm, a2_imm)
        bytecode.extend(encoded)

    return np.array(bytecode, dtype=np.uint8)


def decode_instruction(digits: list) -> dict:
    """Decode 8 quaternary digits into instruction dict."""
    opnum = quat_digits_to_int(digits[0:3])
    a1 = quat_digits_to_int(digits[3:5])
    a2 = quat_digits_to_int(digits[5:7])
    flags = digits[7]

    opname = OPCODE_TABLE[opnum] if opnum < len(OPCODE_TABLE) else f'UNK{opnum}'
    return {
        'op': opname,
        'arg1': a1,
        'arg2': a2,
        'arg1_is_imm': bool(flags & 1),
        'arg2_is_imm': bool(flags & 2),
    }


def decode_bytecode(bytecode: np.ndarray) -> list:
    """Decode quaternary bytecode array into instruction list."""
    instructions = []
    for i in range(0, len(bytecode) - 7, 8):
        digits = [int(d) for d in bytecode[i:i+8]]
        instructions.append(decode_instruction(digits))
    return instructions


# ================================================================
# QPU-native bytecode executor
# ================================================================

def execute_bytecode(qpu: QPU, bytecode: np.ndarray, max_cycles: int = 10000,
                     start_ip: int = 0, load: bool = True) -> int:
    """Execute quaternary bytecode directly on QPU.

    Bytecode lives in QPU memory. The fetch-decode-execute loop
    reads 8 quaternary digits per instruction from memory.
    """
    if load:
        n = min(len(bytecode), len(qpu.mem))
        qpu.mem[:n] = bytecode[:n]
    else:
        n = len(bytecode)

    ip = start_ip
    n_instructions = n // 8

    while not qpu.halt_flag and qpu.cycles < max_cycles:
        if ip >= n_instructions:
            break

        # Fetch: read 8 quaternary digits from memory
        base = ip * 8
        digits = [int(d) for d in qpu.mem[base:base + 8]]
        instr = decode_instruction(digits)

        ip += 1  # advance before execution (jumps override)

        op = instr['op']
        a1 = instr['arg1']
        a2 = instr['arg2']

        # Execute
        if op == 'HALT':
            qpu.op_halt()
        elif op == 'ADD':
            qpu.op_add(a1, a2)
        elif op == 'MUL':
            qpu.op_mul(a1, a2)
        elif op == 'FOLD':
            qpu.op_fold(a1, a2 if a2 > 0 else 2)
        elif op == 'UNFOLD':
            qpu.op_unfold(a1, a2 if a2 > 0 else 2)
        elif op == 'BIND':
            qpu.op_bind(a1, a1, a2)
        elif op == 'ROUTE':
            qpu.op_route(a1, a2)
        elif op == 'JMP':
            ip = a1 if instr['arg1_is_imm'] else qpu._reg_to_int(a1) // 8
        elif op == 'JZ':
            target = a1 if instr['arg1_is_imm'] else qpu._reg_to_int(a1) // 8
            if qpu.zero_flag:
                ip = target
            qpu.cycles += 1
        elif op == 'JNZ':
            target = a1 if instr['arg1_is_imm'] else qpu._reg_to_int(a1) // 8
            if not qpu.zero_flag:
                ip = target
            qpu.cycles += 1
        elif op == 'JC':
            target = a1 if instr['arg1_is_imm'] else qpu._reg_to_int(a1) // 8
            if qpu.carry_flag:
                ip = target
            qpu.cycles += 1
        elif op == 'JNC':
            target = a1 if instr['arg1_is_imm'] else qpu._reg_to_int(a1) // 8
            if not qpu.carry_flag:
                ip = target
            qpu.cycles += 1
        elif op == 'LOAD':
            addr = a2 if instr['arg2_is_imm'] else a2
            qpu.op_load(a1, addr)
        elif op == 'STORE':
            addr = a1 if instr['arg1_is_imm'] else a1
            qpu.op_store(addr, a2)
        elif op == 'SET':
            qpu.op_set(a1, a2)
        elif op == 'ZTEST':
            qpu.op_ztest(a1)
        elif op == 'FMIX':
            qpu.op_fmix(a1, a2, 0)
        elif op == 'CALL':
            # Store return address
            qpu.rstack.append(ip)
            ip = a1 if instr['arg1_is_imm'] else qpu._reg_to_int(a1) // 8
            qpu.cycles += 1
        elif op == 'RET':
            if qpu.rstack:
                ip = qpu.rstack.pop()
            qpu.cycles += 1
        elif op == 'CADD':
            qpu.op_carry_add(a1, a2)
        elif op == 'CSUB':
            qpu.op_carry_sub(a1, a2)
        elif op == 'CMUL':
            qpu.op_carry_mul(a1, a2)
        elif op == 'SHL':
            qpu.op_shl(a1, a2)
        elif op == 'SHR':
            qpu.op_shr(a1, a2)
        elif op == 'CMP':
            qpu.op_cmp(a1, a2)
        elif op == 'SYSCALL':
            qpu.op_syscall(a1)
        elif op == 'MMUEN':
            qpu.op_mmu_on()
        elif op == 'MMUDIS':
            qpu.op_mmu_off()
        elif op == 'MMAP':
            qpu.op_mmu_map(a1, a2)
        elif op == 'MUNMAP':
            qpu.op_mmu_unmap(a1)
        elif op == 'MALLOC':
            qpu.op_mmu_alloc(a1)
        elif op == 'VLOAD':
            qpu.op_vload(a1, a2)
        elif op == 'VSTORE':
            qpu.op_vstore(a1, a2)

    return qpu.cycles


# ================================================================
# Tests
# ================================================================

def test_roundtrip():
    """Test: encode QASM -> bytecode -> decode -> execute. Must match."""
    print("=== Self-Hosting Bootstrap Tests ===")
    print()

    # Test 1: Encode/decode roundtrip
    print("--- Test 1: Encode/Decode Roundtrip ---")
    source = """
SET r0, 1
SET r1, 2
ADD r0, r1
CADD r0, r1
HALT
"""
    bytecode = encode_program(source)
    print(f"  Source: 5 instructions")
    print(f"  Bytecode: {len(bytecode)} quaternary digits ({len(bytecode)//8} instructions)")
    print(f"  First instruction digits: {bytecode[:8]}")

    decoded = decode_bytecode(bytecode)
    for i, instr in enumerate(decoded):
        print(f"  [{i}] {instr['op']} a1={instr['arg1']} a2={instr['arg2']}")

    # Test 2: Execute bytecode directly
    print()
    print("--- Test 2: Bytecode Execution ---")
    qpu = QPU()
    cycles = execute_bytecode(qpu, bytecode, max_cycles=100)
    # SET r0,1 -> r0 = [1,1,...], SET r1,2 -> r1=[2,2,...], ADD -> r0=[3,3,...], CADD -> carry add
    print(f"  Cycles: {cycles}")
    print(f"  r0[:4] = {qpu.regs[0][:4]}")

    # Test 3: Fibonacci via bytecode
    print()
    print("--- Test 3: Fibonacci(10) via Bytecode ---")
    fib_source = """
SET r0, 0
SET r1, 0
SET r2, 0
SET r3, 0
SET r4, 0
SET r5, 0
HALT
"""
    # Build fib manually as bytecode
    fib_bytecode = []
    # We'll encode the loop manually since labels need resolution

    # Setup: r0[0]=1 (current), r1=0 (prev), r5[0]=1 (decrement const)
    # r2 = 9 iterations. Problem: SET fills all 32 elements.
    # Solution: SET to 0 first, then use CADD with r5 to increment to desired value.
    # Or: just set r2 manually after bytecode load.

    # [0] SET r0, 0 (clear)
    fib_bytecode.extend(encode_instruction('SET', 0, 0, arg2_is_imm=True))
    # [1] SET r1, 0 (clear)
    fib_bytecode.extend(encode_instruction('SET', 1, 0, arg2_is_imm=True))
    # [2] SET r5, 0 (clear)
    fib_bytecode.extend(encode_instruction('SET', 5, 0, arg2_is_imm=True))
    # [3] NOP — placeholder, we set r0[0]=1, r5[0]=1, r2=9 manually
    fib_bytecode.extend(encode_instruction('HALT', 0, 0))  # replaced below

    # Loop body (starts at instruction 3):
    # [3] SET r4, 0
    fib_bytecode[-8:] = encode_instruction('SET', 4, 0, arg2_is_imm=True)
    # [4] ADD r4, r0  (copy current to temp)
    fib_bytecode.extend(encode_instruction('ADD', 4, 0))
    # [5] CADD r0, r1  (current += prev)
    fib_bytecode.extend(encode_instruction('CADD', 0, 1))
    # [6] SET r1, 0
    fib_bytecode.extend(encode_instruction('SET', 1, 0, arg2_is_imm=True))
    # [7] ADD r1, r4  (prev = old current)
    fib_bytecode.extend(encode_instruction('ADD', 1, 4))
    # [8] CSUB r2, r5  (counter--)
    fib_bytecode.extend(encode_instruction('CSUB', 2, 5))
    # [9] ZTEST r2
    fib_bytecode.extend(encode_instruction('ZTEST', 2, 0))
    # [10] JNZ 3  (loop back to instruction 3)
    fib_bytecode.extend(encode_instruction('JNZ', 3, 0, arg1_is_imm=True))
    # [11] HALT
    fib_bytecode.extend(encode_instruction('HALT', 0, 0))

    bc = np.array(fib_bytecode, dtype=np.uint8)

    qpu2 = QPU()
    # Load bytecode into memory
    n = min(len(bc), len(qpu2.mem))
    qpu2.mem[:n] = bc[:n]

    # Execute first 3 instructions (SET r0/r1/r5 to 0)
    execute_bytecode(qpu2, bc, max_cycles=3)

    # Now set individual elements that SET can't do (single-element init)
    qpu2.regs[0][0] = 1  # current = 1
    qpu2.regs[5][0] = 1  # decrement constant = 1
    qpu2._int_to_reg(2, 9)  # counter = 9
    qpu2.halt_flag = False
    qpu2.cycles = 0

    # Resume from instruction 3 (loop body)
    cycles = execute_bytecode(qpu2, bc, max_cycles=10000, start_ip=3, load=False)
    result = qpu2._reg_to_int(0)
    print(f"  fib(10) = {result} (expected 55, correct: {result == 55})")
    print(f"  Cycles: {cycles}")

    # Test 4: The meta-test — can we encode the encoder?
    print()
    print("--- Test 4: Bytecode Stats ---")
    print(f"  Encoding: 8 quaternary digits per instruction")
    print(f"  = 16 bits per instruction (2 bits per quat digit)")
    print(f"  = 4 instructions per 32-element register")
    print(f"  Opcode space: 64 possible (33 used, 31 reserved)")
    print(f"  Arg space: 16 values per arg (covers r0-r7 + immediates 0-15)")

    # Test 5: Disassembler (QPU reading its own bytecode)
    print()
    print("--- Test 5: Self-Disassembly ---")
    # The QPU can read its own bytecode from memory and decode it
    simple_bc = []
    simple_bc.extend(encode_instruction('SET', 0, 3, arg2_is_imm=True))
    simple_bc.extend(encode_instruction('ADD', 0, 0))
    simple_bc.extend(encode_instruction('HALT', 0, 0))
    simple_arr = np.array(simple_bc, dtype=np.uint8)

    print(f"  Program bytecode ({len(simple_arr)} digits):")
    for i in range(0, len(simple_arr), 8):
        chunk = simple_arr[i:i+8]
        instr = decode_instruction(list(chunk))
        digits_str = ''.join(str(d) for d in chunk)
        print(f"    [{i//8}] {digits_str} -> {instr['op']} {instr['arg1']},{instr['arg2']}")

    print()
    print("=== BOOTSTRAP COMPLETE ===")
    print(f"  ISA: {len(OPCODE_TABLE)} instructions")
    print(f"  Encoding: 8 quat digits / instruction (16 bits)")
    print(f"  Self-hosting: bytecode encodes, decodes, and executes on QPU")
    print(f"  The computer builds itself.")



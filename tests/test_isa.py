"""ISA-level sanity tests — every advertised opcode group does what it says."""
import numpy as np
import pytest

import qpu
from qpu.gf4 import ADD, MUL, INV


# ── GF(4) field axioms ─────────────────────────────────────────────────────

def test_gf4_addition_is_xor_self_inverse():
    # x + x = 0 for all x in GF(4)
    for x in range(4):
        assert ADD[x][x] == 0


def test_gf4_multiplication_identity():
    for x in range(4):
        assert MUL[1][x] == x
        assert MUL[x][1] == x


def test_gf4_multiplication_zero_absorption():
    for x in range(4):
        assert MUL[0][x] == 0
        assert MUL[x][0] == 0


def test_gf4_inverse_is_correct():
    # INV[x] is the multiplicative inverse for x in {1, 2, 3}.
    # x * INV[x] should equal 1.
    for x in range(1, 4):
        assert MUL[x][INV[x]] == 1


# ── Opcode behavior ────────────────────────────────────────────────────────

def test_add_is_xor_at_isa_level():
    r = qpu.run("""
        SET r0, 1
        SET r1, 2
        ADD r0, r1
        HALT
    """)
    assert int(r.regs[0][0]) == ADD[1][2]


def test_mul_uses_field_table():
    r = qpu.run("""
        SET r0, 2
        SET r1, 3
        MUL r0, r1
        HALT
    """)
    assert int(r.regs[0][0]) == MUL[2][3]


def test_set_then_halt_zero_cycles_for_arithmetic():
    r = qpu.run("""
        SET r0, 3
        HALT
    """)
    # No ALU op happened — but SET still runs and HALT terminates.
    assert int(r.regs[0][0]) == 3


def test_register_file_layout():
    # 8 vector registers × WORD_SIZE GF(4) elements
    r = qpu.run("SET r7, 2\nHALT")
    assert r.regs.shape[0] == 8
    assert int(r.regs[7][0]) == 2

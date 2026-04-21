#!/usr/bin/env python3
"""
qasm.py — Quaternary Array Assembly: assembler + emulator.

Every instruction operates on arrays of GF(4) elements.
No scalar mode. Scalar = array of width 1.
32 elements per register (64 bits = one word).

6 RISC opcodes + 4 control flow + 2 memory = 12 total.

K-address: +12S (THE VISION) — "you designed a computer"
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from .gf4 import ADD, MUL, INV, add_vec, mul_vec


# ================================================================
# QPU — Quaternary Processing Unit (emulator)
# ================================================================

class QPU:
    """Quaternary Processing Unit.

    8 vector registers (r0-r7), each holds WORD_SIZE GF(4) elements.
    All ops are array-native — every instruction hits the whole register.
    """

    WORD_SIZE = 32  # elements per register (32 × 2 bits = 64 bits = one word)

    RETURN_STACK_DEPTH = 8  # 8-deep return stack (matches Forth convention)

    def __init__(self, mem_size: int = 4096):
        # 8 vector registers
        self.regs = np.zeros((8, self.WORD_SIZE), dtype=np.uint8)

        # Memory: array of GF(4) elements
        self.mem = np.zeros(mem_size, dtype=np.uint8)

        # Program counter
        self.pc = 0

        # Return stack
        self.rstack: List[int] = []

        # Flags
        self.zero_flag = False   # set when result is all zeros
        self.carry_flag = False  # set on carry/borrow overflow
        self.halt_flag = False

        # Cycle counter
        self.cycles = 0

        # Instruction trace
        self.trace: List[str] = []

        # SPEAK output: phrase string emitted by op_speak (None until spoken)
        self.speak_output: Optional[str] = None
        # CHAIN annotations: list of (stage_id, reg) for pipeline handoff
        self.chain_ops: List[Tuple[int, int]] = []

    def reset(self):
        self.regs[:] = 0
        self.pc = 0
        self.rstack = []
        self.zero_flag = False
        self.carry_flag = False
        self.halt_flag = False
        self.cycles = 0
        self.trace = []
        self.speak_output = None
        self.chain_ops = []

    # --- Core 6 opcodes (array-native) ---

    def op_add(self, rd: int, rs: int):
        """ADD rd, rs — GF(4) XOR all elements simultaneously."""
        self.regs[rd] = ADD[self.regs[rd], self.regs[rs]]
        self.zero_flag = np.all(self.regs[rd] == 0)
        self.cycles += 1

    def op_mul(self, rd: int, rs: int):
        """MUL rd, rs — GF(4) multiply all elements."""
        self.regs[rd] = MUL[self.regs[rd], self.regs[rs]]
        self.zero_flag = np.all(self.regs[rd] == 0)
        self.cycles += 1

    def op_fold(self, rd: int, factor: int = 2):
        """FOLD rd, factor — reduce by pairwise XOR. 32→16→8→4→2→1."""
        v = self.regs[rd].copy()
        n = len(v)
        new_n = n // factor
        if new_n < 1:
            new_n = 1
        result = np.zeros(self.WORD_SIZE, dtype=np.uint8)
        for i in range(new_n):
            acc = v[i * factor]
            for j in range(1, factor):
                if i * factor + j < n:
                    acc = ADD[acc, v[i * factor + j]]
            result[i] = acc
        self.regs[rd] = result
        self.zero_flag = np.all(result == 0)
        self.cycles += 1

    def op_unfold(self, rd: int, factor: int = 2):
        """UNFOLD rd, factor — expand by duplication. 16→32."""
        v = self.regs[rd].copy()
        result = np.zeros(self.WORD_SIZE, dtype=np.uint8)
        src_len = self.WORD_SIZE // factor
        for i in range(src_len):
            for j in range(factor):
                idx = i * factor + j
                if idx < self.WORD_SIZE:
                    result[idx] = v[i]
        self.regs[rd] = result
        self.cycles += 1

    def op_bind(self, rd: int, rs1: int, rs2: int):
        """BIND rd, rs1, rs2 — triple bind: rd = rs1 ⊗ rs2 (key-value association)."""
        self.regs[rd] = MUL[self.regs[rs1], self.regs[rs2]]
        self.zero_flag = np.all(self.regs[rd] == 0)
        self.cycles += 1

    def op_route(self, rd: int, rs: int):
        """ROUTE rd, rs — scatter/gather. rd[i] = rs[rd[i]] (permutation by value)."""
        indices = self.regs[rd] % self.WORD_SIZE
        self.regs[rd] = self.regs[rs][indices]
        self.cycles += 1

    # --- Control flow ---

    def op_jmp(self, addr: int):
        """JMP addr — unconditional jump."""
        self.pc = addr
        self.cycles += 1

    def op_jz(self, addr: int):
        """JZ addr — jump if zero flag set."""
        if self.zero_flag:
            self.pc = addr
        self.cycles += 1

    def op_jnz(self, addr: int):
        """JNZ addr — jump if zero flag not set."""
        if not self.zero_flag:
            self.pc = addr
        self.cycles += 1

    def op_halt(self):
        """HALT — stop execution."""
        self.halt_flag = True
        self.cycles += 1

    # --- Memory ---

    def op_load(self, rd: int, addr: int):
        """LOAD rd, [addr] — load WORD_SIZE elements from memory into register."""
        end = min(addr + self.WORD_SIZE, len(self.mem))
        n = end - addr
        self.regs[rd][:n] = self.mem[addr:end]
        self.cycles += 1

    def op_store(self, addr: int, rs: int):
        """STORE [addr], rs — store register to memory."""
        end = min(addr + self.WORD_SIZE, len(self.mem))
        n = end - addr
        self.mem[addr:end] = self.regs[rs][:n]
        self.cycles += 1

    # --- Immediate ---

    def op_set(self, rd: int, value: int):
        """SET rd, value — fill register with constant GF(4) value."""
        self.regs[rd][:] = value % 4
        self.cycles += 1

    def op_call(self, addr: int):
        """CALL addr — push return address onto return stack, jump to addr."""
        if len(self.rstack) >= self.RETURN_STACK_DEPTH:
            raise RuntimeError(f"Return stack overflow (depth {self.RETURN_STACK_DEPTH})")
        self.rstack.append(self.pc)
        self.pc = addr
        self.cycles += 1

    def op_ret(self):
        """RET — pop return address from return stack, jump to it."""
        if not self.rstack:
            raise RuntimeError("Return stack underflow: RET with empty return stack")
        self.pc = self.rstack.pop()
        self.cycles += 1

    def op_ztest(self, rs: int):
        """ZTEST rs — set zero_flag if ALL elements of rs are zero. 1 cycle instead of 5 FOLDs."""
        self.zero_flag = np.all(self.regs[rs] == 0)
        self.cycles += 1

    def op_speak(self, rd: int):
        """SPEAK rd — GF(4) register → K-address → SpeakSpellCache phrase.

        FOLDs rd to 4 suit scores, picks winning suit + rank, constructs
        K-address (+rankS), looks up in SpeakSpellCache. Result stored in
        self.speak_output (string). Register rd is unchanged.
        """
        SUITS = ["H", "S", "D", "C"]
        v = self.regs[rd].copy()
        # FOLD 32 → 4 suit scores via XOR reduction
        scores = [0, 0, 0, 0]
        for i, elem in enumerate(v):
            scores[i % 4] ^= int(elem)
        winning_suit_idx = int(np.argmax(scores))
        suit = SUITS[winning_suit_idx]
        # Rank: count non-zero elements, map to 1-13
        nonzero = int(np.count_nonzero(v))
        rank = max(1, min(13, nonzero // 2 + 1))
        k_addr = f"+{rank}{suit}"
        # Try SpeakSpellCache; fall back to K-address string on import failure
        try:
            import sys as _sys
            _root = str(__file__).split("apps")[0]
            if _root not in _sys.path:
                _sys.path.insert(0, _root)
            from cell.isochip.speakspell_cache import SpeakSpellCache
            _cache = SpeakSpellCache().build()
            phrase = _cache.lookup(k_addr, "general") or k_addr
        except Exception:
            phrase = k_addr
        self.speak_output = phrase
        self.cycles += 1

    def op_chain(self, rd: int, stage_id: int):
        """CHAIN rd, stage_id — annotate register for pipeline handoff.

        Appends (stage_id, rd) to self.chain_ops. The pipeline runner
        reads chain_ops after execution to route output to the next stage.
        Stage IDs: 0=speak, 1=polish(prose_chips), 2=bone, 3=boneyard.
        Register rd is unchanged.
        """
        self.chain_ops.append((stage_id, rd))
        self.cycles += 1

    # --- Layer 1: Multi-digit arithmetic (carry chain) ---

    def op_carry_add(self, rd: int, rs: int):
        """CADD rd, rs — Quaternary addition with carry propagation.

        Treats register as a 32-digit base-4 number (LSB first).
        rd[0] is least significant digit. Carry propagates left.
        Result: rd = rd + rs as multi-digit quaternary number.
        """
        carry = 0
        for i in range(self.WORD_SIZE):
            total = int(self.regs[rd][i]) + int(self.regs[rs][i]) + carry
            self.regs[rd][i] = total % 4
            carry = total // 4
        self.zero_flag = np.all(self.regs[rd] == 0)
        # Store carry in a flag for overflow detection
        self.carry_flag = carry > 0
        self.cycles += 1

    def op_carry_mul(self, rd: int, rs: int):
        """CMUL rd, rs — Quaternary multiply with carry (schoolbook).

        Treats registers as 32-digit base-4 numbers.
        Result in rd (low half). Overflow truncated.
        For full result, use CMUL_WIDE (stores high half in r7).
        """
        a = self.regs[rd].copy()
        b = self.regs[rs].copy()
        result = np.zeros(self.WORD_SIZE * 2, dtype=np.uint32)

        for i in range(self.WORD_SIZE):
            if b[i] == 0:
                continue
            carry = 0
            for j in range(self.WORD_SIZE):
                if i + j >= self.WORD_SIZE * 2:
                    break
                prod = int(a[j]) * int(b[i]) + int(result[i + j]) + carry
                result[i + j] = prod % 4
                carry = prod // 4

        self.regs[rd] = result[:self.WORD_SIZE].astype(np.uint8)
        # High half into r7 for inspection
        self.regs[7] = result[self.WORD_SIZE:self.WORD_SIZE * 2].astype(np.uint8)
        self.zero_flag = np.all(self.regs[rd] == 0)
        self.cycles += 1

    def op_carry_sub(self, rd: int, rs: int):
        """CSUB rd, rs — Quaternary subtraction with borrow.

        rd = rd - rs as multi-digit base-4 number.
        """
        borrow = 0
        for i in range(self.WORD_SIZE):
            diff = int(self.regs[rd][i]) - int(self.regs[rs][i]) - borrow
            if diff < 0:
                diff += 4
                borrow = 1
            else:
                borrow = 0
            self.regs[rd][i] = diff % 4
        self.zero_flag = np.all(self.regs[rd] == 0)
        self.carry_flag = borrow > 0  # underflow
        self.cycles += 1

    def op_shl(self, rd: int, amount: int = 1):
        """SHL rd, amount — Shift left by N quaternary digits (multiply by 4^N)."""
        if amount >= self.WORD_SIZE:
            self.regs[rd][:] = 0
        else:
            self.regs[rd][amount:] = self.regs[rd][:-amount].copy()
            self.regs[rd][:amount] = 0
        self.zero_flag = np.all(self.regs[rd] == 0)
        self.cycles += 1

    def op_shr(self, rd: int, amount: int = 1):
        """SHR rd, amount — Shift right by N quaternary digits (divide by 4^N)."""
        if amount >= self.WORD_SIZE:
            self.regs[rd][:] = 0
        else:
            self.regs[rd][:-amount] = self.regs[rd][amount:].copy()
            self.regs[rd][-amount:] = 0
        self.zero_flag = np.all(self.regs[rd] == 0)
        self.cycles += 1

    def op_cmp(self, rd: int, rs: int):
        """CMP rd, rs — Compare as multi-digit base-4 numbers. Sets flags."""
        for i in range(self.WORD_SIZE - 1, -1, -1):
            if self.regs[rd][i] > self.regs[rs][i]:
                self.zero_flag = False
                self.carry_flag = False  # rd > rs
                self.cycles += 1
                return
            elif self.regs[rd][i] < self.regs[rs][i]:
                self.zero_flag = False
                self.carry_flag = True  # rd < rs (borrow)
                self.cycles += 1
                return
        self.zero_flag = True  # equal
        self.carry_flag = False
        self.cycles += 1

    # --- Layer 3: Syscall bridge ---

    def op_syscall(self, num: int):
        """SYSCALL num — Trap to host OS.

        Syscall numbers:
          0 = nop
          1 = write(fd=r0[0], buf_addr=r1_as_int, len=r2_as_int)
          2 = read(fd=r0[0], buf_addr=r1_as_int, len=r2_as_int)
          3 = exit(code=r0[0])
          4 = time() -> r0 = unix timestamp as quaternary digits
          5 = random(n) -> r0 = n random GF(4) elements

        Returns result in r0. Error flag in carry_flag.
        """
        if num == 0:
            pass  # nop
        elif num == 1:
            # write: convert memory region to bytes and print
            addr = self._reg_to_int(1)
            length = self._reg_to_int(2)
            data = self.mem[addr:addr + length]
            # Convert quaternary digits to ASCII (each digit 0-3 maps to char)
            text = ''.join(chr(48 + d) for d in data)  # '0'-'3'
            print(text, end='')
            self.regs[0][:] = 0
            self.regs[0][0] = length % 4
        elif num == 3:
            # exit
            self.halt_flag = True
        elif num == 4:
            # time -> quaternary digits
            import time as _time
            ts = int(_time.time())
            self._int_to_reg(0, ts)
        elif num == 5:
            # random
            n = min(self._reg_to_int(1), self.WORD_SIZE)
            self.regs[0][:n] = np.random.randint(0, 4, n, dtype=np.uint8)
        self.cycles += 1

    def _reg_to_int(self, reg: int) -> int:
        """Convert register (base-4 digits, LSB first) to Python int."""
        val = 0
        for i in range(self.WORD_SIZE - 1, -1, -1):
            val = val * 4 + int(self.regs[reg][i])
        return val

    def _int_to_reg(self, reg: int, val: int):
        """Convert Python int to register (base-4 digits, LSB first)."""
        for i in range(self.WORD_SIZE):
            self.regs[reg][i] = val % 4
            val //= 4

    def op_jc(self, addr: int):
        """JC addr — jump if carry flag set."""
        if self.carry_flag:
            self.pc = addr
        self.cycles += 1

    def op_jnc(self, addr: int):
        """JNC addr — jump if carry flag not set."""
        if not self.carry_flag:
            self.pc = addr
        self.cycles += 1

    # --- Layer 4: Virtual Memory (base-4 page tables) ---

    def _init_mmu(self):
        """Lazy-init the MMU. Called on first virtual memory access."""
        if hasattr(self, '_mmu_ready'):
            return
        self._mmu_ready = True
        self.mmu_enabled = False
        # Page size: 32 elements (1 register width = 1 page)
        self.PAGE_SIZE = self.WORD_SIZE  # 32
        # Page table: maps virtual page number -> physical page number
        # Max 128 virtual pages -> 128 physical pages (4096/32 = 128 physical pages)
        self.page_table = {}  # virtual_page -> physical_page
        self.page_flags = {}  # virtual_page -> {'r': bool, 'w': bool, 'x': bool}
        self.next_phys_page = 0
        self.page_fault_flag = False

    def op_mmu_on(self):
        """MMUEN — Enable virtual memory."""
        self._init_mmu()
        self.mmu_enabled = True
        self.cycles += 1

    def op_mmu_off(self):
        """MMUDIS — Disable virtual memory (direct physical access)."""
        self._init_mmu()
        self.mmu_enabled = False
        self.cycles += 1

    def op_mmu_map(self, rd: int, rs: int):
        """MMAP rd, rs — Map virtual page (rd as int) to physical page (rs as int).

        rd = register holding virtual page number (as base-4 int)
        rs = register holding physical page number (as base-4 int)
        Page gets rwx permissions by default.
        """
        self._init_mmu()
        vpage = self._reg_to_int(rd)
        ppage = self._reg_to_int(rs)
        self.page_table[vpage] = ppage
        self.page_flags[vpage] = {'r': True, 'w': True, 'x': True}
        self.cycles += 1

    def op_mmu_unmap(self, rd: int):
        """MUNMAP rd — Unmap virtual page (rd as int)."""
        self._init_mmu()
        vpage = self._reg_to_int(rd)
        self.page_table.pop(vpage, None)
        self.page_flags.pop(vpage, None)
        self.cycles += 1

    def op_mmu_alloc(self, rd: int):
        """MALLOC rd — Allocate a physical page, map it to virtual page in rd.

        Auto-assigns next free physical page. Returns physical page in rd.
        """
        self._init_mmu()
        vpage = self._reg_to_int(rd)
        max_pages = len(self.mem) // self.PAGE_SIZE
        if self.next_phys_page >= max_pages:
            self.page_fault_flag = True
            self.carry_flag = True  # signal error
            self.cycles += 1
            return
        ppage = self.next_phys_page
        self.next_phys_page += 1
        self.page_table[vpage] = ppage
        self.page_flags[vpage] = {'r': True, 'w': True, 'x': True}
        self._int_to_reg(rd, ppage)
        self.page_fault_flag = False
        self.cycles += 1

    def _translate(self, vaddr: int) -> int:
        """Translate virtual address to physical address via page table."""
        if not self.mmu_enabled:
            return vaddr  # pass-through when MMU off

        self._init_mmu()
        vpage = vaddr // self.PAGE_SIZE
        offset = vaddr % self.PAGE_SIZE

        if vpage not in self.page_table:
            self.page_fault_flag = True
            self.carry_flag = True
            return -1  # page fault

        ppage = self.page_table[vpage]
        paddr = ppage * self.PAGE_SIZE + offset
        self.page_fault_flag = False
        return paddr

    def op_vload(self, rd: int, vaddr_reg: int):
        """VLOAD rd, rs — Load from virtual address (rs holds address as base-4 int)."""
        self._init_mmu()
        vaddr = self._reg_to_int(vaddr_reg)
        paddr = self._translate(vaddr)
        if paddr < 0:
            self.cycles += 1
            return
        # Check read permission
        vpage = vaddr // self.PAGE_SIZE
        if self.mmu_enabled and vpage in self.page_flags and not self.page_flags[vpage]['r']:
            self.page_fault_flag = True
            self.carry_flag = True
            self.cycles += 1
            return
        end = min(paddr + self.WORD_SIZE, len(self.mem))
        n = end - paddr
        self.regs[rd][:n] = self.mem[paddr:end]
        self.cycles += 1

    def op_vstore(self, vaddr_reg: int, rs: int):
        """VSTORE rs, rd — Store register to virtual address (rd holds address as base-4 int)."""
        self._init_mmu()
        vaddr = self._reg_to_int(vaddr_reg)
        paddr = self._translate(vaddr)
        if paddr < 0:
            self.cycles += 1
            return
        vpage = vaddr // self.PAGE_SIZE
        if self.mmu_enabled and vpage in self.page_flags and not self.page_flags[vpage]['w']:
            self.page_fault_flag = True
            self.carry_flag = True
            self.cycles += 1
            return
        end = min(paddr + self.WORD_SIZE, len(self.mem))
        n = end - paddr
        self.mem[paddr:end] = self.regs[rs][:n]
        self.cycles += 1

    def op_fmix(self, rd: int, generator: int, layer_idx: int):
        """FMIX rd, generator, layer_idx — Feistel cross-mix.

        Even layer: left half ^= rolled right half
        Odd layer: right half ^= rolled left half
        """
        half = self.WORD_SIZE // 2
        offset = (layer_idx * generator) % half
        if offset == 0:
            offset = 1

        left = self.regs[rd][:half].copy()
        right = self.regs[rd][half:].copy()

        if layer_idx % 2 == 0:
            right_rotated = np.roll(right, -offset)
            new_left = ADD[left, right_rotated]
            self.regs[rd][:half] = new_left
        else:
            left_rotated = np.roll(left, -offset)
            new_right = ADD[right, left_rotated]
            self.regs[rd][half:] = new_right

        self.cycles += 1

    def op_polar(self, rd: int):
        """POLAR rd — PolarQuant-style 2D angular extraction. 32→16 elements.

        Groups 32 GF(4) elements into 16 consecutive pairs (a, b).
        For each pair, extracts the polar quadrant (angle in 4-quadrant GF4 space):
          angle = ((a >> 1) << 1) | (b >> 1)  → {0,1,2,3} = {H,S,D,C}

        This preserves the angular/directional structure of each pair before
        further FOLD reduction, giving better inner-product preservation than
        naive XOR-fold alone.

        TurboQuant pipeline in QASM (rotation → polar → reduce → speak):
          FMIX r0, 7, 0   ; rotate: spread information uniformly
          POLAR r0         ; extract 16 pairwise angles (PolarQuant step)
          FOLD r0, 2       ; reduce 16→8 suit scores
          FOLD r0, 2       ; reduce 8→4 final scores
          SPEAK r0         ; decode winning suit+rank → K-address phrase

        Unused upper 16 slots are zeroed. Register shape stays WORD_SIZE=32
        but only slots [0:16] carry meaningful data after this op.
        """
        v = self.regs[rd].copy()
        result = np.zeros(self.WORD_SIZE, dtype=np.uint8)
        n_pairs = self.WORD_SIZE // 2  # 16
        for i in range(n_pairs):
            a = int(v[i * 2])
            b = int(v[i * 2 + 1])
            # High bits encode polar quadrant (suit quadrant in GF4 space)
            angle = ((a >> 1) << 1) | (b >> 1)
            result[i] = angle
        self.regs[rd] = result
        self.zero_flag = np.all(result == 0)
        self.cycles += 1


# ================================================================
# Assembler — text → program
# ================================================================

class QAssembler:
    """Assemble quaternary array assembly into executable programs."""

    OPCODES = {
        'ADD': 'add', 'MUL': 'mul', 'FOLD': 'fold', 'UNFOLD': 'unfold',
        'BIND': 'bind', 'ROUTE': 'route',
        'JMP': 'jmp', 'JZ': 'jz', 'JNZ': 'jnz', 'HALT': 'halt',
        'LOAD': 'load', 'STORE': 'store', 'SET': 'set',
        'FMIX': 'fmix',  # Feistel mix: FMIX rd, generator, layer_idx
        'CALL': 'call',  # CALL addr — push PC, jump
        'RET': 'ret',    # RET — pop PC, return
        'ZTEST': 'ztest',  # ZTEST rs — set zero flag from full register (1 cycle)
        # Pipeline opcodes
        'SPEAK': 'speak',  # SPEAK rd — GF(4) reg → K-address → SpeakSpellCache phrase
        'CHAIN': 'chain',  # CHAIN rd, stage_id — annotate reg for pipeline handoff
        'POLAR': 'polar',  # POLAR rd — PolarQuant 2D angular extraction, 32→16 pairs
        # Layer 1: Multi-digit arithmetic
        'CADD': 'cadd',  # CADD rd, rs — base-4 add with carry
        'CSUB': 'csub',  # CSUB rd, rs — base-4 sub with borrow
        'CMUL': 'cmul',  # CMUL rd, rs — base-4 multiply (low in rd, high in r7)
        'SHL': 'shl',    # SHL rd, amount — shift left N quaternary digits
        'SHR': 'shr',    # SHR rd, amount — shift right N quaternary digits
        'CMP': 'cmp',    # CMP rd, rs — compare as base-4 numbers, set flags
        # Layer 3: Syscall bridge
        'SYSCALL': 'syscall',  # SYSCALL num — trap to host OS
        # Control flow extensions
        'JC': 'jc',      # JC addr — jump if carry flag set
        'JNC': 'jnc',    # JNC addr — jump if carry flag not set
        # Layer 4: Virtual Memory
        'MMUEN': 'mmuen',    # Enable MMU
        'MMUDIS': 'mmudis',  # Disable MMU
        'MMAP': 'mmap',      # MMAP rd, rs — map vpage(rd) to ppage(rs)
        'MUNMAP': 'munmap',  # MUNMAP rd — unmap vpage(rd)
        'MALLOC': 'malloc',  # MALLOC rd — alloc page, map to vpage(rd)
        'VLOAD': 'vload',    # VLOAD rd, rs — load from virtual address in rs
        'VSTORE': 'vstore',  # VSTORE rd, rs — store rs to virtual address in rd
    }

    def assemble(self, source: str) -> List[Tuple]:
        """Parse assembly source into instruction list."""
        program = []
        labels = {}

        lines = source.strip().split('\n')

        # First pass: collect labels
        pc = 0
        for line in lines:
            line = line.split(';')[0].strip()  # remove comments
            if not line:
                continue
            if line.endswith(':'):
                labels[line[:-1]] = pc
                continue
            pc += 1

        # Second pass: assemble
        for line in lines:
            line = line.split(';')[0].strip()
            if not line or line.endswith(':'):
                continue

            parts = line.replace(',', ' ').split()
            op = parts[0].upper()

            if op not in self.OPCODES:
                raise ValueError(f"Unknown opcode: {op}")

            args = []
            for arg in parts[1:]:
                arg = arg.strip()
                if arg.startswith('r') and arg[1:].isdigit():
                    args.append(('reg', int(arg[1:])))
                elif arg.startswith('[') and arg.endswith(']'):
                    inner = arg[1:-1]
                    if inner.startswith('0x'):
                        args.append(('addr', int(inner, 16)))
                    else:
                        args.append(('addr', int(inner)))
                elif arg in labels:
                    args.append(('label', labels[arg]))
                elif arg.startswith('0x'):
                    args.append(('imm', int(arg, 16)))
                else:
                    try:
                        args.append(('imm', int(arg)))
                    except ValueError:
                        args.append(('label', labels.get(arg, 0)))

            program.append((self.OPCODES[op], args))

        return program


# ================================================================
# Executor
# ================================================================

def execute(qpu: QPU, program: List[Tuple], max_cycles: int = 10000,
            trace: bool = False) -> int:
    """Execute a program on the QPU. Returns cycle count."""
    qpu.pc = 0
    qpu.halt_flag = False

    while not qpu.halt_flag and qpu.cycles < max_cycles:
        if qpu.pc >= len(program):
            break

        op, args = program[qpu.pc]
        qpu.pc += 1  # advance before execution (jmp overrides)

        if trace:
            qpu.trace.append(f"  {qpu.cycles:4d}: {op} {args}")

        # Dispatch
        if op == 'add':
            qpu.op_add(args[0][1], args[1][1])
        elif op == 'mul':
            qpu.op_mul(args[0][1], args[1][1])
        elif op == 'fold':
            factor = args[1][1] if len(args) > 1 else 2
            qpu.op_fold(args[0][1], factor)
        elif op == 'unfold':
            factor = args[1][1] if len(args) > 1 else 2
            qpu.op_unfold(args[0][1], factor)
        elif op == 'bind':
            qpu.op_bind(args[0][1], args[1][1], args[2][1])
        elif op == 'route':
            qpu.op_route(args[0][1], args[1][1])
        elif op == 'jmp':
            qpu.op_jmp(args[0][1])
        elif op == 'jz':
            qpu.op_jz(args[0][1])
        elif op == 'jnz':
            qpu.op_jnz(args[0][1])
        elif op == 'halt':
            qpu.op_halt()
        elif op == 'load':
            qpu.op_load(args[0][1], args[1][1])
        elif op == 'store':
            qpu.op_store(args[0][1], args[1][1])
        elif op == 'set':
            qpu.op_set(args[0][1], args[1][1])
        elif op == 'fmix':
            qpu.op_fmix(args[0][1], args[1][1], args[2][1])
        elif op == 'call':
            qpu.op_call(args[0][1])
        elif op == 'ret':
            qpu.op_ret()
        elif op == 'ztest':
            qpu.op_ztest(args[0][1])
        elif op == 'speak':
            qpu.op_speak(args[0][1])
        elif op == 'chain':
            qpu.op_chain(args[0][1], args[1][1])
        elif op == 'polar':
            qpu.op_polar(args[0][1])
        # Layer 1: Multi-digit arithmetic
        elif op == 'cadd':
            qpu.op_carry_add(args[0][1], args[1][1])
        elif op == 'csub':
            qpu.op_carry_sub(args[0][1], args[1][1])
        elif op == 'cmul':
            qpu.op_carry_mul(args[0][1], args[1][1])
        elif op == 'shl':
            amount = args[1][1] if len(args) > 1 else 1
            qpu.op_shl(args[0][1], amount)
        elif op == 'shr':
            amount = args[1][1] if len(args) > 1 else 1
            qpu.op_shr(args[0][1], amount)
        elif op == 'cmp':
            qpu.op_cmp(args[0][1], args[1][1])
        # Layer 3: Syscall
        elif op == 'syscall':
            qpu.op_syscall(args[0][1])
        # Carry flag jumps
        elif op == 'jc':
            qpu.op_jc(args[0][1])
        elif op == 'jnc':
            qpu.op_jnc(args[0][1])
        # Layer 4: Virtual Memory
        elif op == 'mmuen':
            qpu.op_mmu_on()
        elif op == 'mmudis':
            qpu.op_mmu_off()
        elif op == 'mmap':
            qpu.op_mmu_map(args[0][1], args[1][1])
        elif op == 'munmap':
            qpu.op_mmu_unmap(args[0][1])
        elif op == 'malloc':
            qpu.op_mmu_alloc(args[0][1])
        elif op == 'vload':
            qpu.op_vload(args[0][1], args[1][1])
        elif op == 'vstore':
            qpu.op_vstore(args[0][1], args[1][1])

    return qpu.cycles


# ================================================================
# Demo programs
# ================================================================

PLINKO_INFERENCE = """
; Plinko inference: 32 tokens through 4 layers
; Tokens in memory at 0x000, pegs at 0x100-0x180
; Output to 0x200

LOAD    r0, [0x000]    ; load 32 tokens
LOAD    r1, [0x100]    ; layer 0 pegs
ADD     r0, r1         ; deflect all 32
LOAD    r1, [0x120]    ; layer 1 pegs
ADD     r0, r1         ; deflect all 32
LOAD    r1, [0x140]    ; layer 2 pegs
ADD     r0, r1         ; deflect all 32
LOAD    r1, [0x160]    ; layer 3 pegs
ADD     r0, r1         ; deflect all 32
STORE   [0x200], r0    ; write output
HALT
"""

FOLD_TO_SUIT = """
; Fold a 32-element vector down to 4 suit scores
; Demonstrates FOLD as attention reduction

LOAD    r0, [0x000]    ; load input
FOLD    r0, 2          ; 32 -> 16 (pairwise XOR)
FOLD    r0, 2          ; 16 -> 8
FOLD    r0, 2          ; 8 -> 4  (one value per suit)
STORE   [0x200], r0    ; 4 suit scores
HALT
"""

KEY_VALUE_STORE = """
; BIND/RETRIEVE demo: store and retrieve a value by key
; key in r1, value in r2

SET     r1, 2          ; key = [2,2,2,...]
SET     r2, 3          ; value = [3,3,3,...]
BIND    r0, r1, r2     ; r0 = key * value (bound pair)
; Now retrieve: multiply bound pair by inverse of key
SET     r3, 3          ; inv(2) = 3 in GF(4)
MUL     r0, r3         ; r0 = bound * inv(key) = value
STORE   [0x200], r0    ; should be [3,3,3,...]
HALT
"""

LOOP_CONVERGENCE = """
; Loop: XOR vector with FMIX'd copy of itself until annihilated
; Demonstrates CALL/RET + ZTEST + JNZ
; FMIX permutes, XOR cancels — converges when all pairs cancel

LOAD    r0, [0x000]    ; load input
SET     r2, 0          ; loop counter (layer_idx for FMIX)

loop:
SET     r1, 0
ADD     r1, r0         ; r1 = copy of r0
FMIX    r1, 7, 0       ; permute r1
ADD     r0, r1         ; r0 ^= permuted(r0) — cancellation
ZTEST   r0             ; 1-cycle zero check
JNZ     loop           ; keep going until all zero
STORE   [0x200], r0    ; converged
HALT
"""


# ================================================================
# Benchmark
# ================================================================

def benchmark():
    print("=" * 60)
    print("  QASM — Quaternary Array Assembly")
    print(f"  Word size: {QPU.WORD_SIZE} elements ({QPU.WORD_SIZE * 2} bits)")
    print("=" * 60)

    asm = QAssembler()
    qpu = QPU()

    # --- Program 1: Plinko inference ---
    print("\n--- Program 1: Plinko inference (4 layers x 32 elements) ---")

    # Load test data into memory
    np.random.seed(42)
    qpu.mem[0x000:0x000 + 32] = np.random.randint(0, 4, 32, dtype=np.uint8)
    for layer in range(4):
        addr = 0x100 + layer * 0x20
        qpu.mem[addr:addr + 32] = np.random.randint(0, 4, 32, dtype=np.uint8)

    prog = asm.assemble(PLINKO_INFERENCE)
    print(f"  Instructions: {len(prog)}")

    t0 = time.perf_counter()
    cycles = execute(qpu, prog, trace=True)
    elapsed_us = (time.perf_counter() - t0) * 1e6

    print(f"  Cycles: {cycles}")
    print(f"  Time: {elapsed_us:.1f}us")
    print(f"  Input:  {qpu.mem[0x000:0x000+8]}...")
    print(f"  Output: {qpu.mem[0x200:0x200+8]}...")
    print(f"  Trace:")
    for line in qpu.trace[:5]:
        print(f"  {line}")
    print(f"    ... ({len(qpu.trace)} total)")

    # --- Program 2: Fold to suit ---
    print("\n--- Program 2: Fold to suit scores ---")
    qpu.reset()
    qpu.mem[0x000:0x000 + 32] = np.random.randint(0, 4, 32, dtype=np.uint8)

    prog = asm.assemble(FOLD_TO_SUIT)
    cycles = execute(qpu, prog)
    suit_scores = qpu.mem[0x200:0x204]
    print(f"  Cycles: {cycles}")
    print(f"  Input (32 elements): {qpu.mem[0x000:0x008]}...")
    print(f"  Suit scores (4): {suit_scores}")
    print(f"  Suits: H={suit_scores[0]} S={suit_scores[1]} D={suit_scores[2]} C={suit_scores[3]}")

    # --- Program 3: Key-value bind/retrieve ---
    print("\n--- Program 3: Key-value store (BIND/RETRIEVE) ---")
    qpu.reset()
    prog = asm.assemble(KEY_VALUE_STORE)
    cycles = execute(qpu, prog)
    result = qpu.mem[0x200:0x208]
    expected = 1  # 2 * 3 = 1 in GF(4)... wait: MUL[2,3] = 1, then MUL[1,3] = 3? No.
    # BIND: r0 = MUL[2,3] = 1 for each element
    # Then MUL r0 by 3: MUL[1,3] = 3. So result should be 3.
    # Wait: inv(2) = 3, so MUL[MUL[2,3], 3] = MUL[1, 3] = 3. Yes, should be 3.
    print(f"  Cycles: {cycles}")
    print(f"  Key: 2, Value: 3")
    print(f"  Bound: key*value = {MUL[2,3]}")
    print(f"  Retrieved (bound * inv(key)): {result[:4]}")
    print(f"  Expected: [3,3,3,3]  Correct: {np.all(result[:4] == 3)}")

    # --- Program 4: Convergence loop ---
    print("\n--- Program 4: Convergence loop ---")
    qpu.reset()
    qpu.mem[0x000:0x000 + 32] = np.random.randint(1, 4, 32, dtype=np.uint8)  # non-zero start
    prog = asm.assemble(LOOP_CONVERGENCE)
    cycles = execute(qpu, prog, max_cycles=1000)
    print(f"  Cycles to converge: {cycles}")
    print(f"  Final state: {qpu.regs[0][:8]}...")
    print(f"  All zero: {np.all(qpu.regs[0] == 0)}")

    # --- Speed benchmark ---
    print("\n--- Speed: plinko inference throughput ---")
    prog = asm.assemble(PLINKO_INFERENCE)

    # Run many times
    n_runs = 10000
    t0 = time.perf_counter()
    for _ in range(n_runs):
        qpu.pc = 0
        qpu.halt_flag = False
        qpu.cycles = 0
        execute(qpu, prog)
    elapsed = time.perf_counter() - t0

    inferences_per_sec = n_runs / elapsed
    # Each inference = 32 tokens (one word)
    tokens_per_sec = inferences_per_sec * 32

    print(f"  {n_runs} inferences in {elapsed*1000:.1f}ms")
    print(f"  {inferences_per_sec:,.0f} inferences/sec")
    print(f"  {tokens_per_sec:,.0f} tokens/sec (32 per inference)")
    print(f"  {elapsed/n_runs*1e6:.1f}us per inference")

    # Theoretical: at 1GHz, 10 instructions = 10ns per inference
    print(f"\n  Theoretical (1GHz QPU): {1e9/10:.0f}M inferences/sec")
    print(f"  Theoretical tokens/sec: {1e9/10*32/1e9:.0f}B tokens/sec")



#!/usr/bin/env python3
"""
agent.py — QPU Agentic Loop

Self-routing generation loop using QPU opcodes.

Loop:
  encode(text) → GF4 register
  → FMIX (TurboQuant rotation)
  → POLAR (PolarQuant angular extraction)
  → FOLD × 2 (reduce to 4 suit scores)
  → SPEAK (K-address + SpeakSpellCache phrase)
  → CHAIN → BreathPipeline stage
  → output text → re-encode → repeat

Converges when K-address stabilizes or matches target.
Each iteration is ~7 QPU cycles + pipeline stage cost.

K-address: +7C (THE FORGE)
"""

import sys
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent.parent
for p in [str(_ROOT), str(_ROOT / "cell"), str(_ROOT / "satus")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from .core import QPU, QAssembler, execute
from .gf4 import ADD

# QASM program: TurboQuant pipeline (7 cycles, runs on CPU emulator or GPU)
_AGENT_QASM = """
LOAD r0, [0]
FMIX r0, 7, 0
POLAR r0
FOLD r0, 2
FOLD r0, 2
SPEAK r0
CHAIN r0, 1
HALT
"""

# Stage IDs for CHAIN opcode (must match BreathPipeline stage ordering)
STAGE_SPEAK   = 0   # SpeakSpellCache phrase only (fast path, no BP)
STAGE_POLISH  = 1   # ProseChips polish only
STAGE_BONE    = 2   # BonePool scaffold generation
STAGE_FULL    = 3   # Full BreathPipeline (all 4 stages)

# K-address → BreathPipeline stage routing
# Hearts/low-rank → polish only (conversational, fast)
# Spades/mid-rank → bone scaffold (analytical structure)
# Diamonds/high-rank → full pipeline (material depth)
# Clubs/any → full pipeline (action/generation)
_ADDR_TO_STAGE = {
    "H": STAGE_POLISH,
    "S": STAGE_BONE,
    "D": STAGE_FULL,
    "C": STAGE_FULL,
}


@dataclass
class AgentResult:
    output: str
    k_addr: str
    iterations: int
    converged: bool
    trace: List[Tuple[str, str]] = field(default_factory=list)  # [(k_addr, output), ...]
    elapsed_ms: float = 0.0


def _encode_text(text: str, word_size: int = 32) -> np.ndarray:
    """Encode text string → GF(4) register via chunked XOR-fold.

    Splits text into word_size-char chunks, maps each char to ord(c) % 4,
    then XOR-folds all chunks into one 32-element register.
    Captures all of the text, not just the first 32 chars.
    """
    reg = np.zeros(word_size, dtype=np.uint8)
    if not text:
        return reg
    encoded = np.array([ord(c) % 4 for c in text], dtype=np.uint8)
    # Pad to multiple of word_size
    pad = (word_size - len(encoded) % word_size) % word_size
    if pad:
        encoded = np.concatenate([encoded, np.zeros(pad, dtype=np.uint8)])
    # XOR-fold all chunks into one register
    chunks = encoded.reshape(-1, word_size)
    for chunk in chunks:
        reg = ADD[reg, chunk]
    return reg


def _kaddr_to_stage(k_addr: str) -> int:
    """Map K-address (e.g. '+7S') to BreathPipeline stage_id."""
    for suit_char, stage in _ADDR_TO_STAGE.items():
        if suit_char in k_addr:
            return stage
    return STAGE_POLISH  # default


def _run_bp_stage(stage_id: int, text: str, k_addr: str) -> str:
    """Run one BreathPipeline stage on text. Falls back gracefully if unavailable."""
    try:
        if stage_id == STAGE_POLISH:
            from cell.isochip.prose_chips import pipeline as prose_pipeline
            results = prose_pipeline([text])
            return results[0] if results else text

        elif stage_id in (STAGE_BONE, STAGE_FULL):
            from cell.positronic.breath_pipeline import BreathPipeline
            bp = BreathPipeline()
            suit_map = {"H": "hearts", "S": "spades", "D": "diamonds", "C": "clubs"}
            suit = next((v for k, v in suit_map.items() if k in k_addr), "spades")
            result = bp.generate(text, suit=suit)
            return result.get("text", text) if isinstance(result, dict) else str(result)

    except Exception:
        pass

    return text  # identity fallback — never crashes


def _compile_agent_program() -> list:
    """Assemble the agent QASM program once."""
    asm = QAssembler()
    return asm.assemble(_AGENT_QASM)


# Module-level compiled program (assembled once on import)
_PROGRAM = None

def _get_program() -> list:
    global _PROGRAM
    if _PROGRAM is None:
        _PROGRAM = _compile_agent_program()
    return _PROGRAM


class QPUAgentLoop:
    """Self-routing agentic generation loop using QPU opcodes.

    Each iteration:
      1. Encode current text → GF4 register
      2. Run TurboQuant pipeline (FMIX→POLAR→FOLD→FOLD→SPEAK)
      3. Get K-address + SpeakSpellCache phrase
      4. Route to BreathPipeline stage via CHAIN logic
      5. Get output text from BP stage
      6. Check convergence (K-address stable or matches target)
      7. Loop

    Usage:
        loop = QPUAgentLoop()
        result = loop.run("I feel overwhelmed")
        print(result.output)   # polished response
        print(result.k_addr)   # e.g. "+7H"
        print(result.iterations)

    With target K-address:
        result = loop.run("write me a plan", target="+9D")
    """

    def __init__(self, max_iters: int = 8, verbose: bool = False):
        self.max_iters = max_iters
        self.verbose = verbose
        self.qpu = QPU()
        self.program = _get_program()

    def _classify(self, text: str) -> Tuple[str, str]:
        """Encode text and run QASM pipeline. Returns (k_addr, speak_phrase)."""
        self.qpu.reset()
        reg = _encode_text(text)
        self.qpu.mem[:32] = reg
        execute(self.qpu, self.program)
        k_addr = self.qpu.speak_output or "+1H"
        # Extract just the address part if it's a full phrase
        # SpeakSpellCache returns phrases; we need the K-address from op_speak internals
        # Re-run classification separately to get the clean address
        q2 = QPU()
        q2.regs[0] = reg.copy()
        # Minimal program: just FMIX→POLAR→FOLD→FOLD, read suit scores directly
        SUITS = ["H", "S", "D", "C"]
        from .core import QAssembler as _QA
        _asm = _QA()
        _prog = _asm.assemble("""
FMIX r0, 7, 0
POLAR r0
FOLD r0, 2
FOLD r0, 2
HALT
""")
        q2.regs[0] = reg.copy()
        execute(q2, _prog)
        scores = [int(q2.regs[0][i]) for i in range(4)]
        winning = int(np.argmax(scores))
        nonzero = int(np.count_nonzero(reg))
        rank = max(1, min(13, nonzero // 2 + 1))
        clean_addr = f"+{rank}{SUITS[winning]}"
        return clean_addr, self.qpu.speak_output or clean_addr

    def run(
        self,
        input_text: str,
        target: Optional[str] = None,
        initial_stage: Optional[int] = None,
    ) -> AgentResult:
        """Run the agentic loop on input_text.

        Args:
            input_text: raw text or query to process
            target: optional K-address to converge toward (e.g. "+7H")
            initial_stage: override BreathPipeline stage for first iteration

        Returns:
            AgentResult with output, k_addr, iterations, converged, trace
        """
        t0 = time.perf_counter()
        current_text = input_text
        prev_addr = None
        trace = []

        for i in range(self.max_iters):
            k_addr, phrase = self._classify(current_text)

            if self.verbose:
                print(f"  iter {i+1}: {k_addr} -> {phrase[:60]}")

            trace.append((k_addr, current_text[:80]))

            # Convergence: address stabilized or hit target
            if k_addr == prev_addr:
                elapsed = (time.perf_counter() - t0) * 1000
                return AgentResult(
                    output=current_text,
                    k_addr=k_addr,
                    iterations=i + 1,
                    converged=True,
                    trace=trace,
                    elapsed_ms=elapsed,
                )

            if target and k_addr == target:
                elapsed = (time.perf_counter() - t0) * 1000
                return AgentResult(
                    output=current_text,
                    k_addr=k_addr,
                    iterations=i + 1,
                    converged=True,
                    trace=trace,
                    elapsed_ms=elapsed,
                )

            # Route to BreathPipeline stage
            stage = initial_stage if (i == 0 and initial_stage is not None) else _kaddr_to_stage(k_addr)
            output = _run_bp_stage(stage, current_text, k_addr)

            # If BP returned something useful, use it; else stay with speak phrase
            if output and output != current_text:
                current_text = output
            elif phrase and phrase != k_addr:
                current_text = phrase

            prev_addr = k_addr

        elapsed = (time.perf_counter() - t0) * 1000
        return AgentResult(
            output=current_text,
            k_addr=prev_addr or "+1H",
            iterations=self.max_iters,
            converged=False,
            trace=trace,
            elapsed_ms=elapsed,
        )


def run_agent(text: str, target: Optional[str] = None, max_iters: int = 8, verbose: bool = False) -> AgentResult:
    """One-shot convenience wrapper."""
    return QPUAgentLoop(max_iters=max_iters, verbose=verbose).run(text, target=target)

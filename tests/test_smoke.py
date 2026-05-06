"""Smoke tests — package imports and basic API doesn't blow up."""
import qpu


def test_version_present():
    assert qpu.__version__
    assert isinstance(qpu.__version__, str)


def test_public_api_exports():
    # Things the README quick-start references must be importable.
    for name in ("run", "from_c", "jit", "benchmark", "QPU", "QAssembler"):
        assert hasattr(qpu, name), f"qpu.{name} missing"


def test_run_minimal_program():
    result = qpu.run("""
        SET r0, 1
        SET r1, 2
        ADD r0, r1
        HALT
    """)
    assert result.cycles > 0
    # GF(4) ADD is XOR over the field. 1 XOR 2 = 3.
    assert int(result.regs[0][0]) == 3


def test_from_c_returns_qasm_string():
    qasm = qpu.from_c("""
        int main() {
            int a = 1;
            int b = 2;
            int c = a + b;
            return c;
        }
    """)
    assert isinstance(qasm, str)
    assert "HALT" in qasm.upper() or len(qasm) > 0  # transpiler produced something

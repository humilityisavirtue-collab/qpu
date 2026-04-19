"""QPU command-line interface."""

import argparse
import sys


def main():
    p = argparse.ArgumentParser(prog='qpu', description='Quaternary Processing Unit')
    sub = p.add_subparsers(dest='command')

    # qpu run <file.qasm>
    run_p = sub.add_parser('run', help='Execute a QASM file')
    run_p.add_argument('file', help='QASM source file')
    run_p.add_argument('--trace', action='store_true', help='Show execution trace')
    run_p.add_argument('--max-cycles', type=int, default=100000)

    # qpu bench
    sub.add_parser('bench', help='Run benchmark suite')

    # qpu compile <file.c>
    cc = sub.add_parser('compile', help='Transpile C to QASM')
    cc.add_argument('file', help='C source file')

    # qpu info
    sub.add_parser('info', help='Show ISA and stats')

    args = p.parse_args()

    if args.command == 'run':
        from .api import run
        with open(args.file) as f:
            source = f.read()
        result = run(source, max_cycles=args.max_cycles)
        print(f"Cycles: {result.cycles}")
        for i in range(8):
            if any(result.regs[i] != 0):
                print(f"  r{i}: {result.regs[i][:8]}... (int: {result.reg_int(i)})")

    elif args.command == 'bench':
        from .api import benchmark
        benchmark()

    elif args.command == 'compile':
        from .api import from_c
        with open(args.file) as f:
            source = f.read()
        print(from_c(source))

    elif args.command == 'info':
        from .core import QAssembler, QPU
        ops = sorted(QAssembler.OPCODES.keys())
        print(f"QPU v0.1.0")
        print(f"  ISA: {len(ops)} instructions")
        print(f"  Registers: 8 x {QPU.WORD_SIZE} GF(4) elements")
        print(f"  Word size: {QPU.WORD_SIZE * 2} bits")
        print(f"  Memory: 4096 elements (default)")
        print(f"  Instructions: {', '.join(ops)}")

    else:
        p.print_help()


if __name__ == '__main__':
    main()

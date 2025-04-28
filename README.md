# ID-QAOA
Implementation of Intermittent Driver QAOA (ID QAOA) for solving the Shortest Vector Problem (SVP) using Cirq.

This repository implements a Quantum Approximate Optimization Algorithm (QAOA) for solving instances of the Shortest Vector Problem (SVP). The code builds, simulates, and optimizes a novel QAOA implementation, termed Intermittent Driver QAOA (ID QAOA).

## Features
- **Custom QAOA Circuit Construction** based on Gram matrices encoding lattice basis vectors.
- **Hybrid Classical Optimizer** combining Grid Search, CMA-ES, and Adam optimization.
- **Energy Evaluation** from the full quantum wavefunction.
- **Visualization** of probability distributions and energy spectra.

## Dependencies
- Cirq
- NumPy
- SymPy
- SciPy
- Matplotlib
- cma
- PyTorch

Install all dependencies via:

```bash
pip install cirq numpy sympy scipy matplotlib cma torch

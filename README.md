# Equilibrium Propagation — a working tutorial

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

A clean, runnable PyTorch implementation of **Equilibrium Propagation (EP)** — a way to train a neural network **without backpropagation** — with a tutorial that trains it on MNIST and visualizes what the network is doing. It's the teaching companion to my first-author paper, which applies EP to leukemia detection ([arXiv:2601.18710](https://arxiv.org/abs/2601.18710)).

---

## The problem: backprop is powerful but biologically and physically awkward

Backpropagation is how essentially every modern network learns, but it has properties that make it a poor fit for two important settings:

- **It isn't biologically plausible.** Backprop needs separate machinery to carry exact gradients backward through the same weights used in the forward pass (the "weight transport problem"), plus a global, synchronized backward sweep. Brains don't appear to do this ([Lillicrap et al., *Nat. Rev. Neurosci.* 2020, "Backpropagation and the brain"](https://www.nature.com/articles/s41583-020-0277-3)).
- **It's hard to put in physical hardware.** Neuromorphic and analog accelerators promise orders-of-magnitude lower energy than GPUs, but they can't easily implement a global backward pass — they're good at *relaxing to a steady state*, not at routing gradients ([Kendall et al., 2020, "Training end-to-end analog neural networks with equilibrium propagation"](https://arxiv.org/abs/2006.01981)).

So a real research question: **can a network learn with only local information and no backward pass, and still be competitive?**

## The idea: learn from the difference between two equilibria

Equilibrium Propagation ([Scellier & Bengio, *Front. Comput. Neurosci.* 2017](https://www.frontiersin.org/articles/10.3389/fncom.2017.00024/full)) answers yes. The network is an energy-based system that settles to a steady state, and learning happens in two phases:

1. **Free phase.** Clamp the input and let the network relax to an energy minimum `s⁰` — no labels involved.
2. **Nudged phase.** Gently pull the output toward the correct label (strength β) and let it settle to a new equilibrium `sᵝ`.

Each weight is then updated from the **local** difference between the two equilibria it connects,

```
ΔWᵢⱼ  ∝  (1/β) · ( sᵢᵝ sⱼᵝ − sᵢ⁰ sⱼ⁰ )
```

No global gradient, no weight transport — every term a synapse needs is available at its own two endpoints. Remarkably, EP's updates approximate the same gradient backprop would compute, which is why it can match backprop's accuracy in the regimes where it applies.

## Why it matters / where it's going

This isn't a backprop replacement for training frontier models today; it's the reference paradigm for a different bet: **energy-efficient, local, on-device learning**. The relevance is growing on three fronts — neuromorphic and analog hardware (learning that runs *in physics* rather than on a GPU), edge devices that must adapt without a datacenter, and the scientific question of how biological learning works. For a research portfolio it's also a concrete demonstration of fluency with the *foundations* of learning algorithms, not just the application of `loss.backward()`.

## What's here (technical breakdown)

This is the **largest part of the repo** — a from-scratch EP system, not a wrapper:

| File | What it implements |
|---|---|
| `model_pytorch.py` | The energy-based network: layer state variables, the global energy `E(s)`, and the fixed-point **relaxation dynamics** (gradient descent on energy until the state stops moving). |
| `equilibrium_propagation_pytorch.py` | The two-phase EP training step — free-phase relaxation, nudged-phase relaxation, and the local contrastive weight update above. |
| `external_world_pytorch.py` | Data plumbing: loads MNIST and presents it as clamped boundary conditions on the input units. |
| `train_model_pytorch.py` | The training loop + evaluation on handwritten digits. |
| `gui_pytorch.py` | A live visualizer of the network's unit activations as it relaxes. |
| `Equilibrium_Propagation_Colab.ipynb` | A zero-setup notebook walkthrough of the whole thing. |

**Skills it exercises:** implementing an iterative fixed-point solver and reasoning about its convergence; deriving and coding a non-backprop learning rule from an energy function; PyTorch tensor mechanics without `autograd` doing the work for you; and the experimental discipline of checking that a local rule actually trains (MNIST accuracy) rather than just runs.

## Quickstart

```bash
pip install -r requirements.txt
python train_model_pytorch.py          # train EP on MNIST, prints accuracy per epoch
python gui_pytorch.py                   # watch the network relax to equilibrium
```

Or open `Equilibrium_Propagation_Colab.ipynb` in Colab for a no-install tour.

## Read next

- `SIMPLE_EXPLANATION.md` — EP explained from scratch, no equations required.
- `APPLICATIONS.md` / `ADVANCED_APPLICATIONS.md` — where local learning rules are headed.
- The paper: **Analyzing Images of Blood Cells with Quantum Machine Learning Methods** ([arXiv:2601.18710](https://arxiv.org/abs/2601.18710)) — EP reaching 86.4% on leukemia detection with no backprop.

## References

- Scellier, B. & Bengio, Y. (2017). *Equilibrium Propagation: bridging the gap between energy-based models and backpropagation.* Front. Comput. Neurosci. 11:24.
- Lillicrap, T. et al. (2020). *Backpropagation and the brain.* Nature Reviews Neuroscience 21, 335–346.
- Kendall, J. et al. (2020). *Training end-to-end analog neural networks with equilibrium propagation.* arXiv:2006.01981.

## License

MIT — see [LICENSE](LICENSE). Author: **Azra Bano**.

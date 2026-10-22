# Sequential Models

A hands-on study of sequential neural-network architectures, from recurrent neural networks to Transformers.

The goal is to understand each architecture mathematically, implement it, train it on real data, and evaluate it with reproducible experiments.

## Repository map

```text
sequential-models/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   └── notation.md
├── datasets/
│   └── README.md
├── models/
│   ├── README.md
│   ├── 01_rnn/
│   ├── 02_lstm/
│   ├── 03_gru/
│   ├── 04_encoder_decoder/
│   ├── 05_attention/
│   └── 06_transformer/
└── results/
    └── README.md
```

Model folders will be added as we start each chapter; we won't create empty notebooks or placeholder code.

## What's inside a model chapter?

Each chapter will organize the material by purpose, adding files when they're useful:

- `README.md` — architecture overview, dataset, run instructions, and results.
- `notebooks/` or clearly named `.ipynb` files — explanation, derivations, implementation, training, and evaluation in Jupyter.
- `mathematics/` — derivation notes in Markdown and, when useful, a PDF.
- `src/` — reusable implementation and data-loading modules.
- `tests/` — checks for shapes, numerical correctness, and behavior.
- `experiments/` — configuration or scripts only if the experiment becomes complex enough to need them.

The NumPy-from-scratch and PyTorch versions will both be included where educationally useful. Not every chapter needs every subfolder.

## Dataset plan

- **IMDb movie reviews:** shared classification dataset for RNN, LSTM, and GRU.
- **English–French parallel sentences:** encoder–decoder and attention.
- **Tiny Shakespeare:** small text corpus for Transformer language modeling.

Dataset links, loading examples, license/source notes, and instructions for accessing data from Jupyter are maintained in [datasets/README.md](datasets/README.md). Downloaded datasets are stored locally and excluded from Git.

## Setup

Clone the repository, open a terminal in the repository root, and run:

```bash
python -m pip install -r requirements.txt
jupyter notebook
```

If PyTorch is already installed and working with your GPU, do not reinstall it unnecessarily. Use the official PyTorch installation selector if you later need to configure a specific CUDA build.

## Working principles

- Use Jupyter for learning, implementation, training, and visualizations.
- Keep reusable code in Python modules when it becomes useful.
- Reuse the same dataset and split for fair RNN/LSTM/GRU comparisons.
- Keep training, validation, and test data separate.
- Document dataset sources and licenses.
- Do not commit raw datasets, credentials, or large checkpoints.
- Record actual experiment results only after running the experiments.

## Current status

**Repository foundation complete.** The RNN chapter is next. Notebooks, implementations, tests, and results will be added as we work through each architecture.

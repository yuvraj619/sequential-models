# Sequential Models

A hands-on study of sequential neural-network architectures, from recurrent neural networks to Transformers.

The project emphasizes mathematical understanding, implementation, reproducible experiments, and evaluation on real datasets.

## Roadmap

1. **RNN** — NumPy implementation and manual backpropagation through time, followed by a PyTorch implementation.
2. **LSTM** — gated recurrent memory.
3. **GRU** — gated recurrence and comparison with RNN/LSTM.
4. **Encoder–Decoder** — sequence-to-sequence learning.
5. **Attention** — attention-based sequence-to-sequence learning.
6. **Transformer** — self-attention and a small language-modeling experiment.

## Working principles

- Use Jupyter notebooks for explanations, implementation, and experiments.
- Keep reusable model and data-loading code in Python modules when it becomes useful.
- Use real datasets with documented sources and licenses.
- Reuse the same dataset and split for fair RNN/LSTM/GRU comparisons.
- Keep datasets, trained checkpoints, and generated artifacts out of Git unless there is a clear reason to include a small artifact.
- Record actual results only after running experiments; do not commit placeholder metrics.

## Planned datasets

- **IMDb movie reviews:** shared starting task for RNN, LSTM, and GRU sentiment classification.
- **English–French parallel sentences:** encoder–decoder and attention experiments.
- **Tiny Shakespeare:** small text corpus for Transformer language modeling.

Dataset sources, licensing notes, and loading instructions will be documented before use. The repository will not contain downloaded datasets themselves.

## Setup

Use a Python environment that has Jupyter installed. From the repository root:

```bash
python -m pip install -r requirements.txt
jupyter notebook
```

If you already have a working PyTorch/CUDA environment, avoid reinstalling PyTorch unnecessarily.

## Repository layout

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
│   └── 01_rnn/
│       └── README.md
└── results/
    └── README.md
```

## Status

**Current stage:** repository foundation. The RNN chapter is the first planned implementation. Model notebooks, source files, tests, and experiment results will be added as work begins.

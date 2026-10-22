# Model Chapters

Each architecture has its own chapter folder. We will add the folders and implementation files progressively as we begin each model.

## Planned chapters

| Folder | Architecture | Main training task |
|---|---|---|
| `01_rnn/` | Recurrent Neural Network | IMDb sentiment classification |
| `02_lstm/` | Long Short-Term Memory | IMDb sentiment classification |
| `03_gru/` | Gated Recurrent Unit | IMDb sentiment classification |
| `04_encoder_decoder/` | Encoder–Decoder | English-to-French translation |
| `05_attention/` | Attention | English-to-French translation |
| `06_transformer/` | Transformer | Tiny Shakespeare language modeling |

RNN, LSTM, and GRU will use the same data split and evaluation protocol to make their comparison meaningful.

## What belongs inside a chapter?

A chapter may contain the following sections. Add only what is needed for that architecture:

```text
01_rnn/
├── README.md
├── 01_rnn_from_scratch.ipynb
├── 02_rnn_pytorch.ipynb
├── mathematics/
│   ├── derivation.md
│   └── derivation.pdf       # optional, when useful
├── src/
│   ├── numpy_model.py
│   ├── pytorch_model.py
│   └── data.py
└── tests/
    └── test_rnn.py
```

### Purpose of each section

- **Jupyter notebooks:** derivations, tensor-shape checks, implementation, training, evaluation, plots, and discussion.
- **mathematics/:** searchable Markdown derivations and an optional PDF copy of polished notes.
- **src/:** reusable model and dataset-loading code, once code is mature enough to reuse.
- **tests/:** shape checks, numerical gradient checks where relevant, and model behavior tests.
- **README.md:** goals, dataset source, how to run the chapter, and actual measured results.

NumPy implementations and manual gradients are especially useful for understanding RNN, LSTM, and GRU internals. PyTorch implementations will show the practical training workflow. For later models, we will choose implementation depth based on what best teaches the architecture.

The notebook is the main learning document; the other files support reuse, testing, and clear documentation. We will not create empty placeholder files.

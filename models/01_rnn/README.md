# 01 — Recurrent Neural Network (RNN)

This is the first model chapter. Implementation files will be added as the work begins.

## Planned work

1. Explain the recurrent update and tensor shapes.
2. Derive the forward pass step by step.
3. Derive the loss and backpropagation through time.
4. Implement a small RNN in NumPy and verify gradients numerically.
5. Train an RNN in PyTorch on IMDb sentiment classification.
6. Evaluate on held-out test data and record actual results.
7. Add focused tests for shapes and numerical behavior.

## Planned notebooks

- `01_rnn_from_scratch.ipynb` — mathematical derivation, NumPy implementation, and manual gradient checks.
- `02_rnn_pytorch.ipynb` — PyTorch implementation, training, and evaluation.

## Planned data interface

A data-loading module may be added when implementation begins. It should document the source, preprocessing, vocabulary/tokenization, sequence truncation, and split strategy.

## Completion checklist

- [ ] Mathematical derivation and notation are documented.
- [ ] NumPy implementation passes numerical gradient checks.
- [ ] PyTorch implementation trains reproducibly.
- [ ] Validation is used for model selection.
- [ ] Test metrics are calculated on held-out data.
- [ ] Runtime, parameter count, and relevant hardware details are recorded.
- [ ] Tests and run instructions are included.

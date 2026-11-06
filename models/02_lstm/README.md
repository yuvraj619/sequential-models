# 02 — Long Short-Term Memory (LSTM)

This is the second model chapter. It follows the vanilla RNN chapter and develops a modern LSTM with a forget gate.

## Chapter structure

- `dataset/` — dataset source and download instructions.
- `mathematics/` — mathematical derivation of the LSTM forward pass, loss, backpropagation through time (BPTT), and parameter updates.
- `01_lstm_from_scratch.ipynb` — planned NumPy implementation and numerical gradient verification.

## Planned work

1. Review the LSTM architecture and tensor dimensions.
2. Implement the forward pass in NumPy.
3. Implement the loss calculation.
4. Implement backpropagation through time (BPTT) manually.
5. Verify analytical gradients using numerical gradient checking.
6. Train the NumPy LSTM on IMDb sentiment classification.
7. Evaluate the model and record actual results.

## Mathematical derivation

The derivation is available in:

- [LSTM mathematical derivation](mathematics/derivation.md)

It covers the modern LSTM architecture with input, forget, and output gates; candidate memory; cell-state and hidden-state updates; sequence-level classification; cross-entropy; BPTT; shared parameter gradients; and gradient-descent updates.

## Dataset

The LSTM chapter will use the IMDb movie-review sentiment dataset, matching the RNN chapter so that the architectures can be compared on the same task and data source.

See [dataset/README.md](dataset/README.md) for download and loading instructions.

## Completion checklist

- [x] Mathematical derivation documented.
- [ ] NumPy forward pass implemented.
- [ ] NumPy BPTT implemented.
- [ ] Analytical gradients checked numerically.
- [ ] Model trained on IMDb.
- [ ] Validation and held-out test evaluation completed.
- [ ] Actual metrics and experiment details documented.

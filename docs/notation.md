# Mathematical Notation and Tensor Shapes

This file records conventions shared by the model chapters. Extend it when a convention is introduced.

## General conventions

- `B`: batch size
- `T`: sequence length / number of time steps
- `D`: input feature dimension per time step
- `H`: hidden-state dimension
- `V`: vocabulary size
- `C`: number of classes

## Common shapes

| Object | Typical shape | Meaning |
|---|---|---|
| Batched sequence | `(B, T, D)` | Batch, time, input features |
| Hidden state | `(B, H)` | One hidden vector per sequence in the batch |
| Class logits | `(B, C)` | Unnormalized class scores |
| Token logits | `(B, T, V)` | Vocabulary scores at each time step |

Exact conventions may differ by implementation. Every notebook should state tensor shapes explicitly, especially when using PyTorch modules with a `batch_first` option.

## Notation rules

- Use uppercase letters for matrices, e.g. `W_xh`.
- Use lowercase letters for vectors, e.g. `h_t`.
- Use subscripts for time steps, e.g. `x_t` and `h_t`.
- State whether a vector is a row or column vector whenever it matters to a derivation.
- Distinguish logits from probabilities.

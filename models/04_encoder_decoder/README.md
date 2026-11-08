# 04 — LSTM Encoder–Decoder

This chapter studies sequence-to-sequence learning using the encoder–decoder architecture introduced by Sutskever, Vinyals, and Le (2014).

The project will build an English-to-French translation model using two LSTMs:
- an **encoder** that reads the source sentence and produces a fixed-dimensional representation;
- a **decoder** that generates the target sentence autoregressively, conditioned on the encoded representation.

The implementation will begin with a single-layer NumPy model to make the mathematics and data flow explicit. The original paper's multi-layer architecture and training choices will be documented separately from our simplified implementation.

## Dataset

The task uses an English–French parallel sentence dataset. See [dataset/README.md](dataset/README.md) for the selected source, download instructions, and expected file format.

## Chapter structure

- `dataset/` — dataset source, download instructions, and preprocessing notes.
- `mathematics/` — derivation of the sequence-to-sequence objective, encoder, decoder, teacher forcing, and backpropagation.
- `src/` — reusable NumPy implementation of the encoder, decoder, and complete sequence-to-sequence model.
- `01_encoder_decoder_from_scratch.ipynb` — derivation, implementation, training, and evaluation in Jupyter.

## Planned work

1. Study the original sequence-to-sequence formulation in Sutskever et al. (2014).
2. Derive the encoder and decoder equations using the LSTM cell developed in the previous chapter.
3. Define the source and target vocabularies, special tokens, and sequence handling.
4. Implement the encoder in NumPy.
5. Implement the autoregressive decoder in NumPy.
6. Connect both components into a sequence-to-sequence model.
7. Implement teacher-forced training and sequence-level cross-entropy.
8. Implement inference with greedy decoding, then compare with beam search.
9. Evaluate translations on a held-out test split and document actual results.

## Architecture scope

The first implementation will use the original fixed-vector encoder–decoder formulation **without attention**. Attention will be studied in the next chapter.

We will use a single encoder layer and a single decoder layer initially. This is an educational simplification; it is not a reproduction of the original paper's four-layer LSTM configuration.

## References

1. Sutskever, I., Vinyals, O., & Le, Q. V. (2014). *Sequence to Sequence Learning with Neural Networks*. Advances in Neural Information Processing Systems. https://arxiv.org/abs/1409.3215

## Completion checklist

- [ ] Original paper and model formulation reviewed.
- [ ] Mathematical derivation documented.
- [ ] English–French dataset documented and prepared.
- [ ] NumPy encoder implemented.
- [ ] NumPy decoder implemented.
- [ ] Encoder–decoder integration implemented.
- [ ] Training and inference implemented.
- [ ] Held-out evaluation completed and actual metrics recorded.

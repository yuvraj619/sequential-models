# Datasets

This folder documents dataset sources and loading procedures. Downloaded datasets belong in the ignored local path `datasets/data/` and should not be committed to Git.

## Planned datasets

### IMDb movie reviews

- **Purpose:** binary sentiment classification for RNN, LSTM, and GRU.
- **Plan:** use a documented train/test split and reserve validation data from the training partition.
- **Record before use:** source URL, dataset version, license/terms, preprocessing, and split details.

### English–French parallel sentences

- **Purpose:** sequence-to-sequence translation for encoder–decoder and attention chapters.
- **Plan:** choose a manageable subset, document provenance and licensing, and use separate train/validation/test data.

### Tiny Shakespeare

- **Purpose:** small text corpus for a Transformer next-token prediction demonstration.
- **Plan:** document source, license, preprocessing, and the split strategy.

## Data handling rules

- Do not upload downloaded datasets or credentials.
- Keep train, validation, and test sets separate.
- Fit learned preprocessing (such as vocabulary construction or normalization) on training data only, unless the dataset's established protocol specifies otherwise.
- Fix and record random seeds where applicable.
- Record the dataset version and any filtering or truncation.

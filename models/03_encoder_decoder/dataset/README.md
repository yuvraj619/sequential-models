# English–French Parallel Dataset

This chapter uses the **English–French sentence-pair dataset** distributed by ManyThings.org in tab-separated Anki format.

## Source

- Dataset page: https://www.manythings.org/anki/
- Direct download: https://www.manythings.org/anki/fra-eng.zip
- Expected archive: `fra-eng.zip`
- Expected extracted file: `fra.txt`

Each row contains an English sentence, its French translation, and optionally additional attribution or metadata fields. The file is tab-separated.

## Download

Download and extract `fra-eng.zip`, then place `fra.txt` in this folder:

```text
models/03_encoder_decoder/dataset/fra.txt
```

Do not commit the raw dataset to Git. Keep the download instructions here so the dataset can be obtained locally when running the notebook.

## Expected loading format

The first two tab-separated columns are the English source sentence and French target sentence. Load them explicitly:

```python
import pandas as pd

data = pd.read_csv(
    "models/03_encoder_decoder/dataset/fra.txt",
    sep="\t",
    header=None,
    usecols=[0, 1],
    names=["english", "french"],
    on_bad_lines="skip"
)

print(data.head())
print(data.shape)
```

Check the downloaded file's actual column layout before training; the source may include optional attribution fields after the sentence pair.

## Project preprocessing plan

The notebook will document and apply preprocessing consistently:

1. Normalize and clean the English and French text without discarding meaningful characters.
2. Add explicit start-of-sequence and end-of-sequence tokens to target sentences.
3. Build source and target vocabularies from the training split only.
4. Reserve distinct IDs for padding, unknown, start, and end tokens.
5. Convert sentences to integer sequences.
6. Pad batches while retaining sequence lengths or masks.
7. Split sentence pairs into training, validation, and held-out test sets before fitting vocabularies.

English and French will have separate vocabularies. The target decoder must learn when to emit the end-of-sequence token.

## Dataset size and scope

The full download contains many sentence pairs and may include short conversational examples as well as longer sentences. For the first NumPy implementation, we may use a documented subset to keep training practical. Any filtering or subset selection must be performed reproducibly and reported in the notebook.

## Attribution and usage

The ManyThings page provides the dataset download and source information. Review the dataset page and the included attribution information before redistributing derived data. This repository will link to the source rather than redistribute the raw file.

## Important note about GitHub

Do not commit `fra.txt` or the ZIP archive. Store the dataset locally; only commit code, documentation, and small derived artifacts that are appropriate to distribute.

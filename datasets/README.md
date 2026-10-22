# Datasets: sources and Jupyter access

This file lists the real datasets planned for the project and explains how to load them. **We store the download/source links in GitHub, not large copies of the datasets.** The actual data is downloaded or cached on your computer when a notebook runs.

## 1. IMDb movie reviews — RNN, LSTM, and GRU

- Dataset page: https://huggingface.co/datasets/stanfordnlp/imdb
- Library: Hugging Face Datasets (`datasets` Python package)
- Task: binary sentiment classification.
- Reuse the same split and preprocessing for RNN, LSTM, and GRU comparisons.

Install the loader package in the same Python environment used by Jupyter:

```python
%pip install datasets
```

Then in a notebook:

```python
from datasets import load_dataset

imdb = load_dataset("stanfordnlp/imdb")
train_data = imdb["train"]
test_data = imdb["test"]

print(train_data[0])
print("Train rows:", len(train_data))
print("Test rows:", len(test_data))
```

Create a validation split from the training portion only; keep the official test split untouched until final evaluation.

## 2. English–French parallel sentences — Encoder–Decoder and Attention

- Dataset/source page: https://www.manythings.org/anki/
- Direct English–French text file: https://www.manythings.org/anki/fra-eng.zip
- Task: translate a source sentence into a target sentence.

Download the archive from the source page or direct link, extract it locally, and point the notebook's data-loading code at the extracted text file. Review the source's terms and attribution before redistribution. **Do not commit the archive or extracted corpus to this repository.**

## 3. Tiny Shakespeare — Transformer

- Source repository: https://github.com/karpathy/char-rnn
- Text file: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
- Task: next-character prediction on a small text corpus.

A notebook can download the text to a local ignored data directory:

```python
from pathlib import Path
from urllib.request import urlretrieve

data_dir = Path("../../datasets/data/tiny_shakespeare")
data_dir.mkdir(parents=True, exist_ok=True)

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
file_path = data_dir / "input.txt"

if not file_path.exists():
    urlretrieve(url, file_path)

text = file_path.read_text(encoding="utf-8")
print(text[:500])
print("Characters:", len(text))
```

Record the source, license/terms, and train/validation/test split used in the experiment before publishing results.

## Where files live on your computer

After cloning this repository, use the repository-relative local path:

```text
sequential-models/
└── datasets/
    └── data/                 # local downloads; ignored by Git
        ├── english_french/
        └── tiny_shakespeare/
```

The `datasets/data/` directory is intentionally excluded by `.gitignore`. The IMDb loader typically caches downloaded files in Hugging Face's local cache unless you explicitly set a cache directory.

## Good experimental practice

- Keep train, validation, and test sets separate.
- Fit learned preprocessing (such as a vocabulary) on training data only.
- Fix and record random seeds where applicable.
- Document versions, filtering, sequence length, tokenization, and split details.
- Check source terms and licensing; a public download link does not automatically mean the dataset can be republished freely.
- Never commit private data, credentials, or large downloaded datasets.

# IMDb Dataset

This folder is for the IMDb sentiment dataset used by the NumPy vanilla RNN project.

## Download

Download the CSV from the dataset page:
https://zenodo.org/records/7928582

Look for `IMDB Dataset.csv`.

## Use it locally

Place the downloaded CSV in this folder. Load it with Pandas:

```python
import pandas as pd

df = pd.read_csv("models/01_rnn/dataset/IMDB Dataset.csv")

print(df.head())
print(df.shape)
```

The CSV contains 50,000 labeled reviews. Before training, we will create training and validation splits from the training portion and keep the test data separate.

## Important note about GitHub

The CSV is large (around 66 MB). GitHub recommends keeping individual files below 50 MB and blocks regular Git files larger than 100 MB. To keep this repository lightweight, do not commit the dataset CSV directly unless we explicitly decide to use Git LFS. The download link and loading instructions are kept here so the dataset can be retrieved when needed.

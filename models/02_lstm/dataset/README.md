# IMDb Dataset

This folder is for the IMDb sentiment dataset used by the NumPy LSTM project.

The LSTM chapter will use the same dataset source as the vanilla RNN chapter to make the comparison between architectures consistent.

## Download

Download the CSV from the dataset page:

https://zenodo.org/records/7928582

Look for `IMDB Dataset.csv`.

## Use it locally

Place the downloaded CSV in this folder. Load it with Pandas:

```python
import pandas as pd

df = pd.read_csv("models/02_lstm/dataset/IMDB Dataset.csv")

print(df.head())
print(df.shape)
```

The CSV contains 50,000 labeled reviews. The training/validation split and held-out test data should follow the same strategy used in the RNN chapter.

## Important note about GitHub

The CSV is large (around 66 MB). To keep this repository lightweight, do not commit the dataset CSV directly unless we explicitly decide to use Git LFS. Keep the download link and loading instructions here instead.

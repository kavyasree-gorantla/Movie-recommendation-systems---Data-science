# Movie-recommendation-systems---Data-science
A hybrid movie recommendation engine combining Content-Based TF-IDF and Collaborative SVD to predict user preferences.
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')
/kaggle/input/datasets/kavyasreegorantla/movie-recommendations-and-ratings/ratings.csv (1)/ratings.csv
/kaggle/input/datasets/kavyasreegorantla/movie-recommendations-and-ratings/mymoviedb.csv (1)/mymoviedb.csv
/kaggle/input/datasets/disham993/9000-movies-dataset/mymoviedb.csv
/kaggle/input/datasets/luisreimberg/ratingscsv/ratings.csv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# ==========================
# Load Dataset
# ==========================
kaggle_path = "/kaggle/input"

if os.path.exists(kaggle_path):
    dataset_folder = os.listdir(kaggle_path)[0]
    movies_path = os.path.join(
        kaggle_path,
        dataset_folder,
        "mymoviedb.csv (1).zip"
    )
else:
    movies_path = "mymoviedb.csv (1).zip"

try:
    # Read ZIP file
    movies_df = pd.read_csv(movies_path, compression="zip")
    print("Dataset loaded successfully!")
except Exception:
    try:
        # Read normal CSV if available
        movies_df = pd.read_csv("mymoviedb.csv")
        print("Loaded local CSV successfully!")
    except Exception as e:
        print("Dataset not found.")
        print(e)
        raise

# ==========================
# Initial Shape
# ==========================
print("Initial shape:", movies_df.shape)

# ==========================
# Fill Missing Values
# ==========================
movies_df["genre"] = movies_df["genre"].fillna("Unknown")
movies_df["overview"] = movies_df["overview"].fillna("")

# ==========================
# One-Hot Encode Genre
# ==========================
if movies_df["genre"].astype(str).str.contains(",").any():
    genres_encoded = movies_df["genre"].str.get_dummies(sep=", ")
else:
    genres_encoded = pd.get_dummies(
        movies_df["genre"],
        prefix="genre"
    )

# ==========================
# TF-IDF on Overview
# ==========================
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

tfidf_matrix = tfidf.fit_transform(movies_df["overview"])

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=[
        f"text_{word}"
        for word in tfidf.get_feature_names_out()
    ]
)

# ==========================
# Combine Features
# ==========================
metadata_features = movies_df.drop(
    columns=["genre", "overview"],
    errors="ignore"
)

metadata_features.reset_index(drop=True, inplace=True)
genres_encoded.reset_index(drop=True, inplace=True)
tfidf_df.reset_index(drop=True, inplace=True)

final_features_df = pd.concat(
    [metadata_features, genres_encoded, tfidf_df],
    axis=1
)

# ==========================
# Output
# ==========================
print("\n--- Feature Engineering Complete ---")
print("Genres encoded shape:", genres_encoded.shape)
print("TF-IDF shape:", tfidf_df.shape)
print("Final feature matrix shape:", final_features_df.shape)

print("\nFeature Matrix Preview:")
print(final_features_df.head())
Dataset not found.
[Errno 2] No such file or directory: 'mymoviedb.csv'
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_16/95777654.py in <cell line: 0>()
     21     # Read ZIP file
---> 22     movies_df = pd.read_csv(movies_path, compression="zip")
     23     print("Dataset loaded successfully!")

/usr/local/lib/python3.12/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 

/usr/local/lib/python3.12/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 

/usr/local/lib/python3.12/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 

/usr/local/lib/python3.12/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,

/usr/local/lib/python3.12/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    793             # ReadBuffer[bytes], WriteBuffer[bytes]]"
--> 794             handle = _BytesZipFile(
    795                 handle, ioargs.mode, **compression_args  # type: ignore[arg-type]

/usr/local/lib/python3.12/dist-packages/pandas/io/common.py in __init__(self, file, mode, archive_name, **kwargs)
   1036         # base class "_BufferedWriter" defined the type as "BytesIO")
-> 1037         self.buffer: zipfile.ZipFile = zipfile.ZipFile(  # type: ignore[assignment]
   1038             file, mode, **kwargs

/usr/lib/python3.12/zipfile/__init__.py in __init__(self, file, mode, compression, allowZip64, compresslevel, strict_timestamps, metadata_encoding)
   1351                 try:
-> 1352                     self.fp = io.open(file, filemode)
   1353                 except OSError:

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/datasets/mymoviedb.csv (1).zip'

During handling of the above exception, another exception occurred:

FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_16/95777654.py in <cell line: 0>()
     25     try:
     26         # Read normal CSV if available
---> 27         movies_df = pd.read_csv("mymoviedb.csv")
     28         print("Loaded local CSV successfully!")
     29     except Exception as e:

/usr/local/lib/python3.12/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.12/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.12/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.12/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.12/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'mymoviedb.csv'
import os

for root, dirs, files in os.walk("/kaggle/input"):
    print(root)
    for file in files:
        print("   ", file)
import os

for root, dirs, files in os.walk("/kaggle/input"):
    for file in files:
        print(os.path.join(root, file))
import pandas as pd

movies_df = pd.read_csv("/kaggle/input/movie-recommendations-and-ratings/mymoviedb.csv")
print(movies_df.head())
import os

for root, dirs, files in os.walk("/kaggle/input"):
    for file in files:
        print(os.path.join(root, file))
import os

path = "/kaggle/input/datasets/kavyasreegorantla/movie-recommendations-and-ratings/mymoviedb.csv (1)"

print(os.listdir(path))
import pandas as pd

movies_df = pd.read_csv("/kaggle/input/datasets/kavyasreegorantla/movie-recommendations-and-ratings/mymoviedb.csv (1)/mymoviedb.csv")

print("Dataset loaded successfully!")
print(movies_df.shape)
movies_df.head()
file_path = "/kaggle/input/datasets/kavyasreegorantla/movie-recommendations-and-ratings/mymoviedb.csv (1)/mymoviedb.csv"

with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
    for i in range(5):
        print(f.readline())
import pandas as pd

file_path = "/kaggle/input/datasets/kavyasreegorantla/movie-recommendations-and-ratings/mymoviedb.csv (1)/mymoviedb.csv"

movies_df = pd.read_csv(
    file_path,
    engine="python",
    encoding="utf-8"
)

print("Dataset loaded successfully!")
print(movies_df.shape)
print(movies_df.head())
movies_df.columns = movies_df.columns.str.lower()
movies_df["genre"] = movies_df["genre"].fillna("Unknown")
movies_df["overview"] = movies_df["overview"].fillna("")
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
file_path = "/kaggle/input/datasets/kavyasreegorantla/movie-recommendations-and-ratings/mymoviedb.csv (1)/mymoviedb.csv"

movies_df = pd.read_csv(
    file_path,
    engine="python",
    encoding="utf-8"
)

print("Dataset Loaded Successfully!")
print(movies_df.shape)
movies_df.head()
movies_df.columns = movies_df.columns.str.lower()

movies_df.head()
movies_df.isnull().sum()
movies_df["genre"] = movies_df["genre"].fillna("Unknown")

movies_df["overview"] = movies_df["overview"].fillna("")

movies_df["title"] = movies_df["title"].fillna("Unknown")
movies_df["combined_features"] = (
    movies_df["genre"] + " " +
    movies_df["overview"]
)

movies_df[["title","combined_features"]].head()
tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(
    movies_df["combined_features"]
)

print(tfidf_matrix.shape)
cosine_sim = cosine_similarity(tfidf_matrix)

print(cosine_sim.shape)
indices = pd.Series(
    movies_df.index,
    index=movies_df["title"]
).drop_duplicates()

indices.head()
def recommend_movies(title, cosine_sim=cosine_sim):

    title = title.strip()

    if title not in indices:
        return ["Movie not found!"]

    idx = indices[title]

    similarity_scores = list(enumerate(cosine_sim[idx]))

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similarity_scores = similarity_scores[1:11]

    movie_indices = [i[0] for i in similarity_scores]

    return movies_df["title"].iloc[movie_indices].tolist()
recommend_movies("Spider-Man: No Way Home")
recommend_movies("The Batman")
movie = "Spider-Man: No Way Home"

recommendations = recommend_movies(movie)

print("Recommended Movies:\n")

for i, rec in enumerate(recommendations,1):
    print(f"{i}. {rec}")
 

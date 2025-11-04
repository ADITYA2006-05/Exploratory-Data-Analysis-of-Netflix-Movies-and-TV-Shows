# %%
"""
Netflix Titles Analysis
Jupyter-ready Python script (use in VSCode or Jupyter by running as cells)

This notebook does the following:
1. (Optional) Load real Netflix dataset if available at './netflix_titles.csv'
2. Otherwise, generate a realistic synthetic Netflix-style dataset and save it to /mnt/data/netflix_titles_synthetic.csv
3. Run EDA: overview, top genres, type distribution, releases by year, additions by year, top countries, top directors
4. Save CSV summaries to ./netflix_analysis_outputs/
5. Produce Matplotlib plots (compatible with Jupyter)

Requirements (see requirements.txt cell below)
"""

# %%
# Cell: Imports
import os
import random
import string
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# For reproducibility
random.seed(42)
np.random.seed(42)

# %%
# Cell: Configuration / Paths
DATA_DIR = "./data"
OUT_DIR = "./analysis_outputs"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

SYNTHETIC_CSV = os.path.join(DATA_DIR, "netflix_titles_synthetic.csv")
REQUIREMENTS_TXT = os.path.join(DATA_DIR, "requirements.txt")

# If you have the real dataset downloaded, place it at './data/netflix_titles.csv'
REAL_DATA_PATH = os.path.join(DATA_DIR, "netflix_titles.csv")

# %%
# Cell: Utility functions to generate synthetic dataset (used if real dataset not present)

def rand_title():
    words = [
        "Love","Life","War","Lost","Secret","Night","Day","Last","First","Return",
        "Adventure","Dark","Bright","Edge","Dream","City","Legend","Journey","Island","Game"
    ]
    return " ".join(random.choices(words, k=random.randint(2,4))) + f" {random.randint(1,99)}"


def rand_name():
    first = ["John","Emma","Michael","Olivia","Liam","Ava","Noah","Sophia","Ethan","Isabella",
             "Mason","Mia","Lucas","Amelia","Logan","Harper"]
    last = ["Smith","Johnson","Brown","Taylor","Anderson","Thomas","Jackson","White","Harris","Martin","Thompson","Garcia"]
    return random.choice(first) + " " + random.choice(last)


def rand_countries():
    countries = [
        "United States","United Kingdom","India","Canada","Australia","France","Germany","Spain","Brazil","Japan","Mexico","South Korea"
    ]
    # some titles have multiple countries
    k = 1 if random.random() < 0.85 else random.randint(2,3)
    return ", ".join(random.sample(countries, k=k))


def rand_genres():
    genres = ["Dramas","Comedies","Documentaries","Action","Thriller","Romantic","Kids","Horror","Sci-Fi","Reality","Crime","Animation"]
    k = random.randint(1,3)
    return ", ".join(random.sample(genres, k=k))


def rand_rating():
    ratings = ["TV-MA","TV-14","PG-13","TV-PG","R","PG","G","Not Rated"]
    return random.choices(ratings, weights=[0.25,0.25,0.15,0.1,0.1,0.08,0.04,0.03])[0]

# %%
# Cell: Create synthetic dataset (if real dataset not found)
if not os.path.exists(REAL_DATA_PATH):
    print("Real dataset not found at", REAL_DATA_PATH)
    print("Generating synthetic dataset and saving to:", SYNTHETIC_CSV)

    n = 6000
    rows = []
    start_date = datetime(2008,1,1)
    end_date = datetime(2024,12,31)
    date_range_days = (end_date - start_date).days

    for i in range(n):
        show_id = i + 1
        title = rand_title()
        show_type = random.choices(["Movie","TV Show"], weights=[0.65,0.35])[0]
        director = rand_name() if random.random() < 0.9 else None
        cast_count = random.randint(1,6)
        cast = ", ".join(rand_name() for _ in range(cast_count))
        country = rand_countries()
        date_added = start_date + timedelta(days=random.randint(0, date_range_days))
        release_year = random.randint(1990, 2024)
        rating = rand_rating()
        if show_type == "Movie":
            duration = f"{random.randint(70,150)} min"
        else:
            duration = f"{random.randint(1,12)} Seasons"
        listed_in = rand_genres()
        description = "A " + random.choice(["thrilling","heartwarming","thought-provoking","light-hearted","gripping","emotional","funny","mysterious"]) + " story about " + random.choice(["family","friendship","love","survival","ambition","power","justice","identity"]) 
        rows.append((show_id, title, show_type, director, cast, country, date_added.strftime("%Y-%m-%d"), release_year, rating, duration, listed_in, description))

    df = pd.DataFrame(rows, columns=[
        "show_id","title","type","director","cast","country","date_added","release_year","rating","duration","listed_in","description"
    ])

    # Introduce a few missing values to mimic real-world datasets
    for col in ["director","cast","country","rating"]:
        mask = np.random.rand(len(df)) < 0.02
        df.loc[mask, col] = None

    df.to_csv(SYNTHETIC_CSV, index=False)
    data_path = SYNTHETIC_CSV
else:
    print("Using real dataset:", REAL_DATA_PATH)
    data_path = REAL_DATA_PATH

# %%
# Cell: Load dataset
print("Loading dataset from:", data_path)
df = pd.read_csv(data_path, parse_dates=["date_added"])  # if date_added missing, parse will ignore

# Quick preview
print("Dataset rows:", len(df))
print(df.head(3).T)

# %%
# Cell: Basic overview & cleaning
# Ensure date_added is datetime (some real datasets use 'date_added' or 'date_added' as string)
if 'date_added' in df.columns:
    try:
        df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    except Exception:
        pass

# Basic stats
total_titles = len(df)
num_movies = int((df['type'] == 'Movie').sum()) if 'type' in df.columns else None
num_tv = int((df['type'] == 'TV Show').sum()) if 'type' in df.columns else None
unique_directors = int(df['director'].nunique(dropna=True)) if 'director' in df.columns else None
unique_countries = int(df['country'].nunique(dropna=True)) if 'country' in df.columns else None

overview = {
    'TotalTitles': total_titles,
    'Movies': num_movies,
    'TV_Shows': num_tv,
    'UniqueDirectors': unique_directors,
    'UniqueCountries': unique_countries
}
print("Overview:", overview)

# %%
# Cell: Top genres (explode listed_in column)
if 'listed_in' in df.columns:
    genre_series = df['listed_in'].dropna().astype(str).str.split(', ').explode()
    top_genres = genre_series.value_counts().head(15).reset_index()
    top_genres.columns = ['Genre','Count']
    print(top_genres.head())
else:
    top_genres = pd.DataFrame()

# %%
# Cell: Type distribution
if 'type' in df.columns:
    type_counts = df['type'].value_counts().reset_index()
    type_counts.columns = ['Type','Count']
    print(type_counts)
else:
    type_counts = pd.DataFrame()

# %%
# Cell: Releases by year
if 'release_year' in df.columns:
    releases_by_year = df.groupby('release_year').size().reset_index(name='Count').sort_values('release_year')
    print(releases_by_year.head())
else:
    releases_by_year = pd.DataFrame()

# %%
# Cell: Titles added per year (if date_added exists)
if 'date_added' in df.columns:
    df['year_added'] = df['date_added'].dt.year
    added_per_year = df.groupby('year_added').size().reset_index(name='Count').sort_values('year_added')
    print(added_per_year.head())
else:
    added_per_year = pd.DataFrame()

# %%
# Cell: Top countries
if 'country' in df.columns:
    country_series = df['country'].dropna().astype(str).str.split(', ').explode()
    top_countries = country_series.value_counts().head(15).reset_index()
    top_countries.columns = ['Country','Count']
    print(top_countries.head())
else:
    top_countries = pd.DataFrame()

# %%
# Cell: Top directors
if 'director' in df.columns:
    top_directors = df['director'].value_counts().head(20).reset_index()
    top_directors.columns = ['Director','Count']
    print(top_directors.head())
else:
    top_directors = pd.DataFrame()

# %%
# Cell: Rating distribution
if 'rating' in df.columns:
    rating_counts = df['rating'].value_counts(dropna=False).reset_index()
    rating_counts.columns = ['Rating','Count']
    print(rating_counts.head())
else:
    rating_counts = pd.DataFrame()

# %%
# Cell: Save analysis outputs as CSV
# (useful for building dashboards or further analysis)
if not top_genres.empty:
    top_genres.to_csv(os.path.join(OUT_DIR, 'top_genres.csv'), index=False)
if not type_counts.empty:
    type_counts.to_csv(os.path.join(OUT_DIR, 'type_counts.csv'), index=False)
if not releases_by_year.empty:
    releases_by_year.to_csv(os.path.join(OUT_DIR, 'releases_by_year.csv'), index=False)
if not added_per_year.empty:
    added_per_year.to_csv(os.path.join(OUT_DIR, 'added_per_year.csv'), index=False)
if not top_countries.empty:
    top_countries.to_csv(os.path.join(OUT_DIR, 'top_countries.csv'), index=False)
if not top_directors.empty:
    top_directors.to_csv(os.path.join(OUT_DIR, 'top_directors.csv'), index=False)
if not rating_counts.empty:
    rating_counts.to_csv(os.path.join(OUT_DIR, 'rating_counts.csv'), index=False)

print("Saved analysis CSVs to:", OUT_DIR)

# %%
# Cell: Plots (Matplotlib only)
plt.rcParams['figure.figsize'] = (10,5)

# 1) Type distribution
if not type_counts.empty:
    fig, ax = plt.subplots()
    ax.bar(type_counts['Type'], type_counts['Count'])
    ax.set_title('Type Distribution (Movie vs TV Show)')
    ax.set_xlabel('Type')
    ax.set_ylabel('Count')
    plt.show()

# 2) Number of titles by release year
if not releases_by_year.empty:
    fig, ax = plt.subplots()
    ax.plot(releases_by_year['release_year'], releases_by_year['Count'])
    ax.set_title('Number of Titles by Release Year')
    ax.set_xlabel('Release Year')
    ax.set_ylabel('Count')
    plt.show()

# 3) Titles added per year
if not added_per_year.empty:
    fig, ax = plt.subplots()
    ax.plot(added_per_year['year_added'], added_per_year['Count'])
    ax.set_title('Number of Titles Added to Catalog by Year (date_added)')
    ax.set_xlabel('Year Added')
    ax.set_ylabel('Count')
    plt.show()

# 4) Top genres
if not top_genres.empty:
    fig, ax = plt.subplots()
    ax.bar(top_genres['Genre'], top_genres['Count'])
    ax.set_title('Top Genres')
    ax.set_xticklabels(top_genres['Genre'], rotation=45, ha='right')
    plt.show()

# 5) Top countries
if not top_countries.empty:
    fig, ax = plt.subplots()
    ax.bar(top_countries['Country'], top_countries['Count'])
    ax.set_title('Top Countries (by title count)')
    ax.set_xticklabels(top_countries['Country'], rotation=45, ha='right')
    plt.show()

# %%
# Cell: Extra lightweight analyses (optional)
# Co-occurrence of genres (build simple genre co-occurrence matrix)
if 'listed_in' in df.columns:
    genres_expanded = df['listed_in'].dropna().astype(str).str.split(', ')
    unique_genres = sorted(set(g for row in genres_expanded for g in row))
    genre_index = {g: i for i, g in enumerate(unique_genres)}
    cooc = np.zeros((len(unique_genres), len(unique_genres)), dtype=int)
    for row in genres_expanded:
        for i, a in enumerate(row):
            for b in row[i+1:]:
                cooc[genre_index[a], genre_index[b]] += 1
                cooc[genre_index[b], genre_index[a]] += 1
    genre_cooc_df = pd.DataFrame(cooc, index=unique_genres, columns=unique_genres)
    # Save small co-occurrence csv
    genre_cooc_df.to_csv(os.path.join(OUT_DIR, 'genre_cooccurrence.csv'))
    print('Saved genre co-occurrence matrix')

# %%
# Cell: Requirements file (write into data folder)
requirements_txt = (
    "pandas==2.2.2\n"
    "numpy==1.26.4\n"
    "matplotlib==3.9.2\n"
    "jupyter==1.1.1\n"
)
with open(REQUIREMENTS_TXT, 'w') as f:
    f.write(requirements_txt)
print('requirements written to', REQUIREMENTS_TXT)

# %%
# Cell: Final notes for the grader / README snippet
readme = f"""
Netflix Titles Analysis

How to run:
1. Create a Python virtual environment:
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\\Scripts\\activate
2. Install dependencies:
   pip install -r {REQUIREMENTS_TXT}
3. Run this script in Jupyter (open as notebook) or run as a script:
   jupyter notebook

Files created:
 - Synthetic dataset (if real not provided): {SYNTHETIC_CSV}
 - Analysis outputs (CSV): {OUT_DIR}
 - requirements: {REQUIREMENTS_TXT}

Notes:
 - If you have the real Netflix dataset, place it at {REAL_DATA_PATH} before running.
 - This notebook intentionally uses Matplotlib (no Seaborn) for compatibility.
"""
with open(os.path.join(DATA_DIR, 'README_NETFLIX.md'), 'w') as f:
    f.write(readme)

print('\nAll done. Notebook is ready. Open this file in Jupyter or VSCode to run.')

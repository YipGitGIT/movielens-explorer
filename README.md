# MovieLens Explorer

MovieLens Explorer is an explainable movie recommendation and discovery tool built on the MovieLens Latest Small dataset.

## Project Goal

This project combines a relational database layer and a graph-based relationship layer to support:

- personalized movie recommendation
- explainable recommendation evidence
- related-movie exploration
- rating and genre analytics

## Dataset

I use the MovieLens Latest Small dataset, which includes:

- movies
- ratings
- tags
- external movie links

Core files:

- `movies.csv`
- `ratings.csv`
- `tags.csv`
- `links.csv`

## System Design

### Relational Layer

I use SQLite to store structured movie data.

Main tables:

- `users`
- `movies`
- `genres`
- `movie_genres`
- `ratings`
- `tags`
- `links`

This layer supports:

- joins
- aggregation
- filtering
- descriptive statistics

### Graph Layer

I model graph relationships using graph-based logic in Python.

Main node/edge ideas:

- `User -> Movie` via ratings
- `Movie -> Genre`
- `Movie -> Tag`
- `Movie -> Movie` similarity

This layer supports:

- related-movie exploration
- similar-user discovery
- explainable recommendation paths

## Main Features

### 1. Recommendations
Users can:
- choose an existing user profile, or
- select their own favorite movies

The system then recommends movies and explains why they were recommended.

### 2. Movie Explorer
Users can select one movie and view related movies based on:
- overlapping high-rating users
- shared genres
- shared tags

### 3. SQL Analytics
The app provides:
- top movies
- genre statistics
- release-year rating trends

### 4. Schema Visualization
The app includes both:
- relational schema diagram
- graph schema diagram

## Tech Stack

- Python
- Streamlit
- SQLite
- pandas
- matplotlib
- networkx

## How to Run

```bash
python -m streamlit run app.py

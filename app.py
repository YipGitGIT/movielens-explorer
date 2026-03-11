from pathlib import Path
import math
import re
import sqlite3
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MovieLens Explorer", layout="wide")

APP_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_CANDIDATES = [APP_DIR, Path.cwd(), Path("/mnt/data")]
DB_PATH = APP_DIR / "movielens.db"


def find_file(name: str) -> Path:
    for base in DATA_CANDIDATES:
        path = base / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {name} in {DATA_CANDIDATES}")


def extract_year(title: str):
    m = re.search(r"\((\d{4})\)\s*$", str(title))
    return int(m.group(1)) if m else None


@st.cache_data
def load_raw_data():
    movies = pd.read_csv(find_file("movies.csv"))
    ratings = pd.read_csv(find_file("ratings.csv"))
    tags = pd.read_csv(find_file("tags.csv"))
    links = pd.read_csv(find_file("links.csv"))
    movies["year"] = movies["title"].apply(extract_year)
    return movies, ratings, tags, links


@st.cache_resource
def init_sqlite():
    movies, ratings, tags, links = load_raw_data()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()

    cur.execute("drop table if exists users")
    cur.execute("drop table if exists movies")
    cur.execute("drop table if exists genres")
    cur.execute("drop table if exists movie_genres")
    cur.execute("drop table if exists ratings")
    cur.execute("drop table if exists tags")
    cur.execute("drop table if exists links")

    cur.execute("create table users (user_id integer primary key)")
    cur.execute("create table movies (movie_id integer primary key, title text, year integer)")
    cur.execute("create table genres (genre_id integer primary key autoincrement, genre_name text unique)")
    cur.execute("create table movie_genres (movie_id integer, genre_id integer, primary key (movie_id, genre_id))")
    cur.execute("create table ratings (user_id integer, movie_id integer, rating real, timestamp integer)")
    cur.execute("create table tags (user_id integer, movie_id integer, tag text, timestamp integer)")
    cur.execute("create table links (movie_id integer primary key, imdb_id integer, tmdb_id integer)")

    users = pd.DataFrame({"user_id": sorted(ratings["userId"].unique())})
    users.to_sql("users", conn, if_exists="append", index=False)

    movies_sql = movies[["movieId", "title", "year"]].rename(columns={"movieId": "movie_id"})
    movies_sql.to_sql("movies", conn, if_exists="append", index=False)

    ratings_sql = ratings.rename(columns={"userId": "user_id", "movieId": "movie_id"})
    ratings_sql.to_sql("ratings", conn, if_exists="append", index=False)

    tags_sql = tags.rename(columns={"userId": "user_id", "movieId": "movie_id"})
    tags_sql.to_sql("tags", conn, if_exists="append", index=False)

    links_sql = links.rename(columns={"movieId": "movie_id", "imdbId": "imdb_id", "tmdbId": "tmdb_id"})
    links_sql.to_sql("links", conn, if_exists="append", index=False)

    genre_set = set()
    movie_genre_rows = []
    for _, row in movies.iterrows():
        movie_id = int(row["movieId"])
        values = str(row["genres"]).split("|") if pd.notna(row["genres"]) else []
        values = [g.strip() for g in values if g and g != "(no genres listed)"]
        for g in values:
            genre_set.add(g)
            movie_genre_rows.append((movie_id, g))

    genre_df = pd.DataFrame({"genre_name": sorted(genre_set)})
    genre_df.to_sql("genres", conn, if_exists="append", index=False)

    genre_lookup = pd.read_sql_query("select genre_id, genre_name from genres", conn)
    genre_map = dict(zip(genre_lookup["genre_name"], genre_lookup["genre_id"]))
    movie_genres_df = pd.DataFrame(
        [{"movie_id": movie_id, "genre_id": genre_map[genre_name]} for movie_id, genre_name in movie_genre_rows]
    )
    movie_genres_df.to_sql("movie_genres", conn, if_exists="append", index=False)

    cur.execute("create index if not exists idx_ratings_user on ratings(user_id)")
    cur.execute("create index if not exists idx_ratings_movie on ratings(movie_id)")
    cur.execute("create index if not exists idx_tags_movie on tags(movie_id)")
    cur.execute("create index if not exists idx_movie_genres_movie on movie_genres(movie_id)")
    cur.execute("create index if not exists idx_movie_genres_genre on movie_genres(genre_id)")
    conn.commit()
    return conn


@st.cache_data
def build_support_objects():
    movies, ratings, tags, links = load_raw_data()

    movie_title = dict(zip(movies["movieId"], movies["title"]))
    movie_year = dict(zip(movies["movieId"], movies["year"]))
    title_to_movie_id = dict(zip(movies["title"], movies["movieId"]))

    movie_genres = {}
    for _, row in movies.iterrows():
        values = str(row["genres"]).split("|") if pd.notna(row["genres"]) else []
        values = [g.strip() for g in values if g and g != "(no genres listed)"]
        movie_genres[int(row["movieId"])] = values

    user_ratings = defaultdict(dict)
    movie_ratings = defaultdict(dict)
    for _, row in ratings.iterrows():
        user_id = int(row["userId"])
        movie_id = int(row["movieId"])
        rating = float(row["rating"])
        user_ratings[user_id][movie_id] = rating
        movie_ratings[movie_id][user_id] = rating

    movie_tags = defaultdict(list)
    for _, row in tags.iterrows():
        movie_tags[int(row["movieId"])] .append(str(row["tag"]).strip())

    movie_tag_counts = {}
    for movie_id, values in movie_tags.items():
        normalized = [v.lower() for v in values if v]
        movie_tag_counts[movie_id] = Counter(normalized)

    movie_stats = (
        ratings.groupby("movieId")
        .agg(avg_rating=("rating", "mean"), n_ratings=("rating", "count"))
        .reset_index()
    )
    movie_stats["weighted_score"] = movie_stats["avg_rating"] * (1 + movie_stats["n_ratings"].clip(upper=250) / 250)
    movie_stats_map = {
        int(row["movieId"]): {
            "avg_rating": float(row["avg_rating"]),
            "n_ratings": int(row["n_ratings"]),
            "weighted_score": float(row["weighted_score"]),
        }
        for _, row in movie_stats.iterrows()
    }

    title_options = movies[["movieId", "title", "year"]].sort_values(["title", "year"], ascending=[True, True]).reset_index(drop=True)

    return {
        "movies": movies,
        "ratings": ratings,
        "tags": tags,
        "links": links,
        "movie_title": movie_title,
        "movie_year": movie_year,
        "title_to_movie_id": title_to_movie_id,
        "movie_genres": movie_genres,
        "user_ratings": user_ratings,
        "movie_ratings": movie_ratings,
        "movie_tags": movie_tags,
        "movie_tag_counts": movie_tag_counts,
        "movie_stats_map": movie_stats_map,
        "title_options": title_options,
    }


def cosine_on_overlap(a: dict, b: dict):
    common = sorted(set(a).intersection(b))
    if len(common) < 3:
        return 0.0
    a_mean = sum(a[movie_id] for movie_id in common) / len(common)
    b_mean = sum(b[movie_id] for movie_id in common) / len(common)
    numerator = sum((a[movie_id] - a_mean) * (b[movie_id] - b_mean) for movie_id in common)
    a_denom = math.sqrt(sum((a[movie_id] - a_mean) ** 2 for movie_id in common))
    b_denom = math.sqrt(sum((b[movie_id] - b_mean) ** 2 for movie_id in common))
    if a_denom == 0 or b_denom == 0:
        return 0.0
    return numerator / (a_denom * b_denom)


def top_genres_for_profile(profile_ratings: dict, data, min_rating: float = 4.0):
    counter = Counter()
    for movie_id, rating in profile_ratings.items():
        if rating >= min_rating:
            counter.update(data["movie_genres"].get(movie_id, []))
    return counter


def similar_users_from_profile(profile_ratings: dict, data, limit: int = 15):
    sims = []
    for other_id, other_ratings in data["user_ratings"].items():
        sim = cosine_on_overlap(profile_ratings, other_ratings)
        if sim > 0:
            overlap = len(set(profile_ratings).intersection(other_ratings))
            sims.append((other_id, sim, overlap))
    sims.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return sims[:limit]


def recommend_from_profile(profile_ratings: dict, data, top_k: int = 10):
    watched = set(profile_ratings)
    sims = similar_users_from_profile(profile_ratings, data, limit=40)
    if not sims:
        return pd.DataFrame(columns=["movieId", "title", "predicted_score", "avg_rating", "n_ratings", "reason"]), []

    genre_pref = top_genres_for_profile(profile_ratings, data)
    candidate_scores = defaultdict(float)
    similarity_weight = defaultdict(float)
    supporter_count = defaultdict(int)
    supporters = defaultdict(list)

    for other_id, sim, overlap in sims:
        for movie_id, rating in data["user_ratings"][other_id].items():
            if movie_id in watched or rating < 4.0:
                continue
            movie_stats = data["movie_stats_map"].get(movie_id, {"avg_rating": 0.0, "n_ratings": 0, "weighted_score": 0.0})
            popularity_bonus = min(movie_stats["n_ratings"], 100) / 500
            genre_bonus = 0.0
            for genre_name in data["movie_genres"].get(movie_id, []):
                genre_bonus += 0.08 * genre_pref.get(genre_name, 0)
            score = sim * rating + genre_bonus + popularity_bonus
            candidate_scores[movie_id] += score
            similarity_weight[movie_id] += sim
            supporter_count[movie_id] += 1
            supporters[movie_id].append((other_id, sim, rating))

    rows = []
    for movie_id, raw_score in candidate_scores.items():
        if similarity_weight[movie_id] <= 0:
            continue
        normalized = raw_score / similarity_weight[movie_id]
        movie_stats = data["movie_stats_map"].get(movie_id, {"avg_rating": 0.0, "n_ratings": 0, "weighted_score": 0.0})
        related_genres = data["movie_genres"].get(movie_id, [])
        matching_genres = [genre_name for genre_name in related_genres if genre_pref.get(genre_name, 0) > 0]
        strongest_supporters = sorted(supporters[movie_id], key=lambda x: x[1], reverse=True)[:2]
        strongest_supporter_text = ", ".join([f"User {user_id}" for user_id, _, _ in strongest_supporters])
        reason_parts = []
        if matching_genres:
            reason_parts.append("shared genres: " + ", ".join(matching_genres[:3]))
        if supporter_count[movie_id] > 0:
            reason_parts.append(f"liked by {supporter_count[movie_id]} similar users")
        if strongest_supporter_text:
            reason_parts.append(f"top supporters: {strongest_supporter_text}")
        tag_counter = data["movie_tag_counts"].get(movie_id, Counter())
        if tag_counter:
            reason_parts.append("tags: " + ", ".join([tag for tag, _ in tag_counter.most_common(3)]))
        rows.append(
            {
                "movieId": movie_id,
                "title": data["movie_title"].get(movie_id, str(movie_id)),
                "predicted_score": round(normalized, 3),
                "avg_rating": round(movie_stats["avg_rating"], 3),
                "n_ratings": movie_stats["n_ratings"],
                "reason": " | ".join(reason_parts),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result, sims
    result = result.sort_values(["predicted_score", "avg_rating", "n_ratings", "title"], ascending=[False, False, False, True]).head(top_k)
    return result, sims


def build_profile_from_selected_titles(selected_titles: list[str], assumed_rating: float, data):
    profile = {}
    for title in selected_titles:
        movie_id = data["title_to_movie_id"].get(title)
        if movie_id is not None:
            profile[int(movie_id)] = float(assumed_rating)
    return profile


def movie_similarity_candidates(movie_id: int, data, top_k: int = 10):
    target_high_raters = {user_id for user_id, rating in data["movie_ratings"].get(movie_id, {}).items() if rating >= 4.0}
    target_genres = set(data["movie_genres"].get(movie_id, []))
    target_tags = set(tag.lower() for tag in data["movie_tags"].get(movie_id, []))
    rows = []

    for other_movie_id in data["movie_title"]:
        if other_movie_id == movie_id:
            continue
        other_high_raters = {user_id for user_id, rating in data["movie_ratings"].get(other_movie_id, {}).items() if rating >= 4.0}
        other_genres = set(data["movie_genres"].get(other_movie_id, []))
        other_tags = set(tag.lower() for tag in data["movie_tags"].get(other_movie_id, []))

        user_overlap = len(target_high_raters & other_high_raters)
        genre_overlap = len(target_genres & other_genres)
        tag_overlap = len(target_tags & other_tags)

        if user_overlap == 0 and genre_overlap == 0 and tag_overlap == 0:
            continue

        other_stats = data["movie_stats_map"].get(other_movie_id, {"avg_rating": 0.0, "n_ratings": 0, "weighted_score": 0.0})
        score = 0.55 * user_overlap + 1.5 * genre_overlap + 0.75 * tag_overlap + 0.01 * min(other_stats["n_ratings"], 100)
        reason_parts = []
        if genre_overlap:
            reason_parts.append("shared genres")
        if user_overlap:
            reason_parts.append(f"{user_overlap} overlapping high-rating users")
        if tag_overlap:
            reason_parts.append("shared tags")

        rows.append(
            {
                "movieId": other_movie_id,
                "title": data["movie_title"][other_movie_id],
                "score": round(score, 3),
                "avg_rating": round(other_stats["avg_rating"], 3),
                "n_ratings": other_stats["n_ratings"],
                "reason": " | ".join(reason_parts),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["score", "avg_rating", "n_ratings", "title"], ascending=[False, False, False, True]).head(top_k)


def build_movie_graph(center_movie_id: int, data, n_neighbors: int = 8):
    graph = nx.Graph()
    center_title = data["movie_title"].get(center_movie_id, str(center_movie_id))
    graph.add_node(center_title, kind="center")
    neighbors = movie_similarity_candidates(center_movie_id, data, top_k=n_neighbors)
    for _, row in neighbors.iterrows():
        neighbor_title = row["title"]
        graph.add_node(neighbor_title, kind="neighbor")
        graph.add_edge(center_title, neighbor_title, weight=float(row["score"]))
    return graph, neighbors


def plot_movie_graph(graph: nx.Graph):
    fig, ax = plt.subplots(figsize=(11, 7))
    pos = nx.spring_layout(graph, seed=7)
    weights = [graph[u][v]["weight"] for u, v in graph.edges()]
    widths = [1.2 + 0.15 * w for w in weights]
    node_sizes = [2600 if graph.nodes[node].get("kind") == "center" else 1800 for node in graph.nodes]
    nx.draw_networkx(graph, pos=pos, ax=ax, with_labels=True, width=widths, node_size=node_sizes, font_size=9)
    ax.set_axis_off()
    return fig


def build_relational_schema_graph():
    graph = nx.DiGraph()
    tables = ["users", "movies", "genres", "movie_genres", "ratings", "tags", "links"]
    for table_name in tables:
        graph.add_node(table_name)
    edges = [
        ("ratings", "users"),
        ("ratings", "movies"),
        ("tags", "users"),
        ("tags", "movies"),
        ("movie_genres", "movies"),
        ("movie_genres", "genres"),
        ("links", "movies"),
    ]
    graph.add_edges_from(edges)
    return graph


def build_graph_schema_graph():
    graph = nx.DiGraph()
    graph.add_edge("User", "Movie", label="RATED")
    graph.add_edge("Movie", "Genre", label="IN_GENRE")
    graph.add_edge("Movie", "Tag", label="HAS_TAG")
    graph.add_edge("Movie", "Movie", label="SIMILAR_TO")
    return graph


def plot_schema_graph(graph: nx.DiGraph, title: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    pos = nx.spring_layout(graph, seed=11)
    nx.draw_networkx(graph, pos=pos, ax=ax, arrows=True, node_size=2600, font_size=10)
    edge_labels = {(u, v): d.get("label", "") for u, v, d in graph.edges(data=True)}
    if any(edge_labels.values()):
        nx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=edge_labels, font_size=9, ax=ax)
    ax.set_title(title)
    ax.set_axis_off()
    return fig


def sql_top_movies(conn, min_ratings: int = 30, limit: int = 10):
    query = f"""
    select
        m.title,
        round(avg(r.rating), 3) as avg_rating,
        count(*) as n_ratings
    from ratings r
    join movies m on r.movie_id = m.movie_id
    group by m.movie_id, m.title
    having count(*) >= {int(min_ratings)}
    order by avg(r.rating) desc, count(*) desc, m.title asc
    limit {int(limit)}
    """
    return pd.read_sql_query(query, conn)


def sql_genre_stats(conn):
    query = """
    select
        g.genre_name,
        round(avg(r.rating), 3) as avg_rating,
        count(*) as n_ratings,
        count(distinct r.movie_id) as n_movies
    from ratings r
    join movie_genres mg on r.movie_id = mg.movie_id
    join genres g on mg.genre_id = g.genre_id
    group by g.genre_name
    order by n_ratings desc, avg_rating desc
    """
    return pd.read_sql_query(query, conn)


def sql_year_stats(conn, min_movies: int = 10):
    query = f"""
    select
        m.year,
        count(distinct m.movie_id) as n_movies,
        round(avg(r.rating), 3) as avg_rating,
        count(*) as n_ratings
    from movies m
    join ratings r on m.movie_id = r.movie_id
    where m.year is not null
    group by m.year
    having count(distinct m.movie_id) >= {int(min_movies)}
    order by m.year asc
    """
    return pd.read_sql_query(query, conn)


def sql_user_profile(conn, user_id: int):
    query = f"""
    select
        m.title,
        round(r.rating, 2) as rating,
        r.timestamp
    from ratings r
    join movies m on r.movie_id = m.movie_id
    where r.user_id = {int(user_id)}
    order by r.rating desc, m.title asc
    """
    return pd.read_sql_query(query, conn)


def sql_favorite_genres(conn, user_id: int):
    query = f"""
    select
        g.genre_name,
        round(avg(r.rating), 3) as avg_rating,
        count(*) as n_ratings
    from ratings r
    join movie_genres mg on r.movie_id = mg.movie_id
    join genres g on mg.genre_id = g.genre_id
    where r.user_id = {int(user_id)}
    group by g.genre_name
    order by avg_rating desc, n_ratings desc, g.genre_name asc
    """
    return pd.read_sql_query(query, conn)


def render_home(data):
    st.title("MovieLens Explorer")
    st.subheader("An explainable movie recommendation and discovery tool")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Movies", f"{len(data['movie_title']):,}")
    c2.metric("Users", f"{len(data['user_ratings']):,}")
    c3.metric("Ratings", f"{len(data['ratings']):,}")
    c4.metric("Tags", f"{len(data['tags']):,}")

    st.markdown(
        """
        **What this app demonstrates**

        - a relational layer for joins, aggregation, filtering, and statistics
        - a graph layer for relationship exploration and explainable recommendations
        - a movie recommendation workflow using ratings, genres, and tags

        **Core pages**

        - **SQL Analytics**: top movies, genre-level statistics, year trends
        - **Recommendations**: recommend movies from an existing user profile or from selected favorite movies
        - **Movie Explorer**: inspect one movie and visualize related movies as a graph
        - **Schema**: show relational and graph schema diagrams for the report and slides
        """
    )

    st.markdown("### Quick start")
    st.markdown(
        """
        1. Open **Recommendations**.
        2. Try **Custom Input** and pick three to five movies you like.
        3. Generate recommendations and inspect the explanation column.
        4. Open **Movie Explorer** to visualize local movie relationships.
        """
    )


def render_sql_analytics(conn):
    st.title("SQL Analytics")

    controls_left, controls_right = st.columns(2)
    min_ratings = controls_left.slider("Minimum ratings for top-movie table", 5, 200, 30, 5)
    limit = controls_right.slider("Rows in top-movie table", 5, 25, 10, 1)

    top_movies = sql_top_movies(conn, min_ratings=min_ratings, limit=limit)
    genre_stats = sql_genre_stats(conn)
    year_stats = sql_year_stats(conn, min_movies=10)

    left, right = st.columns(2)
    left.subheader("Top movies")
    left.dataframe(top_movies, use_container_width=True, hide_index=True)

    right.subheader("Genre statistics")
    right.dataframe(genre_stats, use_container_width=True, hide_index=True)

    if not genre_stats.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        top = genre_stats.head(10).sort_values("n_ratings")
        ax.barh(top["genre_name"], top["n_ratings"])
        ax.set_xlabel("Number of ratings")
        ax.set_ylabel("Genre")
        ax.set_title("Top genres by rating count")
        st.pyplot(fig)

    if not year_stats.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(year_stats["year"], year_stats["avg_rating"])
        ax.set_xlabel("Year")
        ax.set_ylabel("Average rating")
        ax.set_title("Average rating by release year")
        st.pyplot(fig)


def render_recommendations(conn, data):
    st.title("Recommendations")
    mode = st.radio("Recommendation mode", ["Existing User", "Custom Input"], horizontal=True)

    if mode == "Existing User":
        user_ids = sorted(data["user_ratings"].keys())
        user_id = st.selectbox("Select a user", user_ids)
        profile_ratings = data["user_ratings"].get(int(user_id), {})
        profile_df = sql_user_profile(conn, int(user_id))
        favorite_genres = sql_favorite_genres(conn, int(user_id))

        top_k = st.slider("Number of recommendations", 3, 15, 8, 1)
        recs, sims = recommend_from_profile(profile_ratings, data, top_k=top_k)

        c1, c2 = st.columns(2)
        c1.subheader(f"User {user_id} rating history")
        c1.dataframe(profile_df.head(15), use_container_width=True, hide_index=True)

        c2.subheader("Favorite genres")
        c2.dataframe(favorite_genres.head(10), use_container_width=True, hide_index=True)

    else:
        title_options = data["title_options"]["title"].tolist()
        selected_titles = st.multiselect(
            "Select movies you like",
            options=title_options,
            default=[
                title for title in [
                    "Toy Story (1995)",
                    "Star Wars: Episode IV - A New Hope (1977)",
                    "Matrix, The (1999)",
                ] if title in title_options
            ],
        )
        assumed_rating = st.slider("Assumed rating for selected favorites", 3.5, 5.0, 4.5, 0.5)
        top_k = st.slider("Number of recommendations", 3, 15, 8, 1)
        profile_ratings = build_profile_from_selected_titles(selected_titles, assumed_rating, data)
        genre_counter = top_genres_for_profile(profile_ratings, data)
        favorite_genres = pd.DataFrame(
            [{"genre_name": key, "count": value} for key, value in genre_counter.most_common()]
        )
        profile_df = pd.DataFrame(
            [{"title": title, "assumed_rating": assumed_rating} for title in selected_titles]
        )
        recs, sims = recommend_from_profile(profile_ratings, data, top_k=top_k)

        c1, c2 = st.columns(2)
        c1.subheader("Custom profile")
        c1.dataframe(profile_df, use_container_width=True, hide_index=True)

        c2.subheader("Inferred favorite genres")
        c2.dataframe(favorite_genres, use_container_width=True, hide_index=True)

    st.subheader("Recommended movies")
    if recs.empty:
        st.warning("No recommendation candidates were generated. Try selecting more movies or a different user.")
    else:
        st.dataframe(recs, use_container_width=True, hide_index=True)

    st.subheader("Most similar users")
    sim_df = pd.DataFrame(sims, columns=["userId", "similarity", "overlap"])
    if sim_df.empty:
        st.info("No similar users with enough overlap were found.")
    else:
        st.dataframe(sim_df, use_container_width=True, hide_index=True)


def render_movie_explorer(data):
    st.title("Movie Explorer")
    title_options = data["title_options"]
    selected_title = st.selectbox("Select a movie", title_options["title"].tolist())
    movie_id = int(title_options.loc[title_options["title"] == selected_title, "movieId"].iloc[0])

    stats = data["movie_stats_map"].get(movie_id, {"avg_rating": 0.0, "n_ratings": 0, "weighted_score": 0.0})
    genres = data["movie_genres"].get(movie_id, [])
    tag_counter = data["movie_tag_counts"].get(movie_id, Counter())

    c1, c2, c3 = st.columns(3)
    c1.metric("Average rating", f"{stats['avg_rating']:.3f}")
    c2.metric("Number of ratings", f"{stats['n_ratings']:,}")
    c3.metric("Year", data["movie_year"].get(movie_id) if data["movie_year"].get(movie_id) else "N/A")

    st.markdown(f"**Title:** {selected_title}")
    st.markdown(f"**Genres:** {', '.join(genres) if genres else 'N/A'}")
    st.markdown(f"**Top tags:** {', '.join([tag for tag, _ in tag_counter.most_common(8)]) if tag_counter else 'N/A'}")

    graph, neighbors = build_movie_graph(movie_id, data)
    st.subheader("Related movies")
    if neighbors.empty:
        st.info("No related movies were found.")
    else:
        st.dataframe(neighbors, use_container_width=True, hide_index=True)
        fig = plot_movie_graph(graph)
        st.pyplot(fig)


def render_schema_page():
    st.title("Database Design")
    st.markdown(
        """
        ### Relational schema
        - users(user_id)
        - movies(movie_id, title, year)
        - genres(genre_id, genre_name)
        - movie_genres(movie_id, genre_id)
        - ratings(user_id, movie_id, rating, timestamp)
        - tags(user_id, movie_id, tag, timestamp)
        - links(movie_id, imdb_id, tmdb_id)

        ### Graph schema
        - (:User)-[:RATED {rating, timestamp}]->(:Movie)
        - (:Movie)-[:IN_GENRE]->(:Genre)
        - (:Movie)-[:HAS_TAG]->(:Tag)
        - (:Movie)-[:SIMILAR_TO {score}]->(:Movie)
        """
    )

    left, right = st.columns(2)
    relational_graph = build_relational_schema_graph()
    graph_graph = build_graph_schema_graph()

    left.pyplot(plot_schema_graph(relational_graph, "Relational schema"))
    right.pyplot(plot_schema_graph(graph_graph, "Graph schema"))

    st.markdown(
        """
        ### Design rationale
        - The relational layer is used for normalized storage, joins, filtering, aggregations, and statistics.
        - The graph layer is used for relationship exploration, neighbor search, and explainable recommendations.
        - The application combines both: SQL prepares clean structured data, and graph logic exposes multi-hop relationships.
        """
    )


def render_about_page():
    st.title("Project Framing")
    st.markdown(
        """
        **Project title**

        MovieLens Explorer: An Explainable Hybrid Recommendation System Using Relational and Graph Data

        **Problem statement**

        Traditional recommendation systems often return suggestions without explaining why the user should trust them. This project builds a movie exploration and recommendation tool that combines relational queries with graph-based relationship discovery to provide both recommendations and interpretable evidence.

        **Main use cases**

        - personalized movie recommendation
        - explanation of recommendation evidence
        - similar-user discovery
        - related-movie exploration
        - genre-level analytics
        """
    )


def main():
    conn = init_sqlite()
    data = build_support_objects()

    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Recommendations", "Movie Explorer", "SQL Analytics", "Schema", "About"],
    )

    if page == "Home":
        render_home(data)
    elif page == "Recommendations":
        render_recommendations(conn, data)
    elif page == "Movie Explorer":
        render_movie_explorer(data)
    elif page == "SQL Analytics":
        render_sql_analytics(conn)
    elif page == "Schema":
        render_schema_page()
    else:
        render_about_page()


main()

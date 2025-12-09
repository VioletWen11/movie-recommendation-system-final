# =========================
# 🎬 Streamlit Movie Recommender App
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="🎬 Movie Recommendation System", layout="wide")

# =========================
# 1) Load Data
# =========================
DATA_DIR = "./ml-latest-small"

@st.cache_data
def load_data():
    movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    # extract year
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)").fillna("")
    # one-hot genres
    genres_dummies = movies["genres"].str.get_dummies(sep="|")
    movies_encoded = pd.concat(
        [movies[["movieId", "title", "genres", "year"]], genres_dummies], axis=1
    )
    return movies, ratings, movies_encoded

movies, ratings, movies_encoded = load_data()
st.sidebar.success("✅ Data successfully loaded!")

# =========================
# 2) Prepare Features
# =========================
user_counts  = ratings["userId"].value_counts()
movie_counts = ratings["movieId"].value_counts()

# popularity/activity thresholds
active_users   = user_counts[user_counts >= 20].index
popular_movies = movie_counts[movie_counts >= 10].index

filtered_ratings = ratings[
    (ratings["userId"].isin(active_users)) &
    (ratings["movieId"].isin(popular_movies))
]

# genre similarity (only for popular movies)
genre_features = movies_encoded.set_index("movieId").drop(columns=["title", "genres", "year"])
genre_features = genre_features.loc[genre_features.index.intersection(popular_movies)]
genre_sim = cosine_similarity(genre_features)
genre_sim_df = pd.DataFrame(genre_sim, index=genre_features.index, columns=genre_features.index)

# rating matrix (for shape / global mean in predictor)
R = filtered_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
R_dense  = R.values
movie_ids = R.columns.tolist()

# =========================
# 3) Helpers
# =========================
def predict_user_ratings(user_rating_vec, sim_df, movie_id_order, k=20):
    """Item-based CF using provided similarity matrix (here: genre-based)."""
    pred = np.zeros_like(user_rating_vec, dtype=float)
    # global mean over known ratings in training matrix
    existing = R_dense[R_dense > 0]
    global_mean = existing.mean() if existing.size else 3.5

    for i, mid in enumerate(movie_id_order):
        if user_rating_vec[i] > 0:
            pred[i] = user_rating_vec[i]
            continue

        # similarity to all items in same order
        if mid not in sim_df.index:
            pred[i] = global_mean
            continue

        sim_to_i   = sim_df.loc[mid, movie_id_order].reindex(movie_id_order).fillna(0).values
        rated_mask = user_rating_vec > 0
        if not np.any(rated_mask):
            pred[i] = global_mean
            continue

        weights = sim_to_i * rated_mask
        if np.all(weights == 0):
            pred[i] = user_rating_vec[rated_mask].mean()
            continue

        top_idx       = np.argsort(weights)[-k:][::-1]
        top_weights   = weights[top_idx]
        top_ratings   = user_rating_vec[top_idx]
        if np.sum(np.abs(top_weights)) == 0:
            pred[i] = user_rating_vec[rated_mask].mean()
        else:
            pred[i] = float(np.dot(top_weights, top_ratings) / np.sum(np.abs(top_weights)))

    return pred

def recommend_by_selected_titles(selected_titles, movies_df, sim_df, top_n=10):
    """Average similarity to user-selected titles. Safely ignore non-popular items."""
    if not selected_titles:
        return pd.Series(dtype=float)

    selected_ids_all = movies_df.loc[movies_df["title"].isin(selected_titles), "movieId"]
    selected_ids = [mid for mid in selected_ids_all if mid in sim_df.index]

    if not selected_ids:
        st.warning("⚠️ Selected movies are not in the popular set used for similarity. Please pick others.")
        return pd.Series(dtype=float)

    # average similarity
    scores = sim_df[selected_ids].mean(axis=1)
    scores = scores.drop(index=selected_ids, errors="ignore")  # do not recommend what user selected
    return scores.sort_values(ascending=False).head(top_n)

# =========================
# 4) Layout – Sidebar
# =========================
st.title("🎬 Movie Recommendation System")
st.markdown("**A hybrid system combining Genre-Based and Collaborative Filtering approaches.**")

with st.sidebar:
    st.header("👤 User Settings")
    mode  = st.radio("Choose mode:", ["Genre-based (no login)", "Simulated User (Collaborative Filtering)"])
    top_n = st.slider("Top-N recommendations:", 5, 20, 10)

# -------------------------
# 5) Mode A: Genre-based (WITH charts)
# -------------------------
if mode == "Genre-based (no login)":
    st.subheader("🎞️ Choose a few movies you like")
    selected = st.multiselect(
        "Select from the list below:",
        options=movies["title"].tolist(),
        default=["Toy Story (1995)", "Jumanji (1995)"]
    )

    if "recs_gb" not in st.session_state:
        st.session_state.recs_gb = None

    if st.button("✨ Generate Recommendations"):
        recs = recommend_by_selected_titles(selected, movies, genre_sim_df, top_n=top_n)
        if not recs.empty:
            details = movies.set_index("movieId").loc[recs.index, ["title", "year", "genres"]]
            st.session_state.recs_gb = pd.DataFrame({
                "Title":   details["title"].values,
                "Year":    details["year"].values,
                "Genres":  details["genres"].values,
                "Similarity": recs.values
            })

    if st.session_state.recs_gb is not None:
        st.write("### 🎯 Recommendation Results")
        st.dataframe(st.session_state.recs_gb, use_container_width=True)

        # ✔️ Charts only for Genre-based mode
        show_chart = st.checkbox("📊 Show similarity analytics")
        if show_chart:
            # Bar: similarity
            fig, ax = plt.subplots(figsize=(8, 4))
            plot_df = st.session_state.recs_gb.sort_values("Similarity", ascending=True)
            sns.barplot(x="Similarity", y="Title", data=plot_df, palette="crest", ax=ax)
            ax.set_title("Top Recommended Movies by Similarity")
            st.pyplot(fig)

            # Hist: year distribution
            year_series = st.session_state.recs_gb["Year"].replace("", np.nan).dropna()
            if not year_series.empty:
                fig2, ax2 = plt.subplots(figsize=(6, 3))
                sns.histplot(year_series.astype(int), bins=10, color="#EC7063", ax=ax2)
                ax2.set_title("Distribution of Recommended Movies by Year")
                ax2.set_xlabel("Release Year")
                ax2.set_ylabel("Count")
                st.pyplot(fig2)

# -------------------------
# 6) Mode B: Simulated User (NO charts)
# -------------------------
else:
    st.subheader("👤 Simulate a User: rate some movies you’ve watched")

    # init states
    if "sample_movies" not in st.session_state:
        st.session_state.sample_movies = np.random.choice(movies["title"].tolist(), 8, replace=False)
    if "user_ratings" not in st.session_state:
        st.session_state.user_ratings = {}
    if "search_done" not in st.session_state:
        st.session_state.search_done = False

    # --- 🔍 Search (on top) ---
    st.markdown("### 🔍 Search & Rate a Movie")
    query = st.text_input(
        "Type a movie name:",
        value="" if st.session_state.search_done else st.session_state.get("last_query", "")
    )

    if query:
        st.session_state.last_query = query
        matched = movies[movies["title"].str.contains(query, case=False, na=False)].head(5)

        if not matched.empty and not st.session_state.search_done:
            for _, row in matched.iterrows():
                title = row["title"]
                if title in st.session_state.user_ratings:
                    st.markdown(f"📽️ *{title}* — ⭐ **{st.session_state.user_ratings[title]:.1f}/5** (rated)")
                else:
                    rating = st.slider(
                        f"Your rating for “{title}”:",
                        1.0, 5.0, 3.0, 0.5, key=f"search_{title}"
                    )
                    if rating != 3.0:  # changed by user
                        st.session_state.user_ratings[title] = rating
                        st.toast(f"✅ Rated {title}: {rating:.1f} stars")
                        st.session_state.search_done = True
                        st.rerun()
        elif st.session_state.search_done:
            st.success("🎬 Rating saved! You can search another movie.")
            st.session_state.search_done = False
            st.session_state.last_query = ""

    # --- 🔄 Refresh batch (does NOT clear ratings) ---
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔄 Refresh Batch"):
            rated_titles = set(st.session_state.user_ratings.keys())
            all_titles   = movies["title"].tolist()
            candidates   = [t for t in all_titles if t not in rated_titles]
            pool = candidates if len(candidates) >= 8 else all_titles
            st.session_state.sample_movies = np.random.choice(pool, 8, replace=False)
            st.toast("✅ New batch loaded!")
    with col2:
        st.caption("Rated movies are saved and used for recommendation.")

    # --- ⭐ Already Rated Movies (card style) ---
    if st.session_state.user_ratings:
        st.markdown("### ⭐ Already Rated Movies")
        rated_items = list(st.session_state.user_ratings.items())
        cols = st.columns(4)
        for i, (title, rating) in enumerate(rated_items):
            with cols[i % 4]:
                st.markdown(
                    f"""
                    <div style="background-color:#f8f9fa; border-radius:12px; padding:12px; margin:6px; border:1px solid #eee;">
                        🎬 <b>{title}</b><br>
                        ⭐ <b style="color:#E67E22;">{rating:.1f}</b>/5
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # --- 🎞️ Rate current batch ---
    st.markdown("### 🎞️ Rate These Movies")
    for title in st.session_state.sample_movies:
        if title in st.session_state.user_ratings:
            st.markdown(f"🎬 *{title}* — ⭐ **{st.session_state.user_ratings[title]:.1f}/5** (rated)")
        else:
            val = st.slider(
                f"Your rating for “{title}”:",
                1.0, 5.0, 3.0, 0.5, key=f"slider_{title}"
            )
            if val != 3.0:
                st.session_state.user_ratings[title] = val
                st.toast(f"✅ Rated {title}: {val:.1f} stars")
                st.rerun()

    # --- 🔮 Generate recommendations (table only, no charts) ---
    if st.button("🔮 Generate Recommendations"):
        rated_dict = {k: v for k, v in st.session_state.user_ratings.items() if v > 0}
        if not rated_dict:
            st.warning("Please rate at least one movie.")
        else:
            # build new user's rating vector aligned to movie_ids (popular set)
            user_vec = np.zeros(len(movie_ids), dtype=float)
            for title, rating in rated_dict.items():
                mids = movies.loc[movies["title"] == title, "movieId"].values
                if len(mids) == 0:
                    continue
                mid = mids[0]
                if mid in movie_ids:
                    user_vec[movie_ids.index(mid)] = rating

            preds = predict_user_ratings(user_vec, genre_sim_df, movie_ids, k=30)
            pred_series = pd.Series(preds, index=movie_ids)

            # remove seen items; take top N
            seen_ids = [movies.loc[movies["title"] == t, "movieId"].values[0]
                        for t in rated_dict.keys() if len(movies.loc[movies["title"] == t, "movieId"].values) > 0]
            pred_series = pred_series.drop(index=seen_ids, errors="ignore")
            pred_series = pred_series.sort_values(ascending=False).head(top_n)

            details = movies.set_index("movieId").loc[pred_series.index, ["title", "year", "genres"]]
            rec_df = pd.DataFrame({
                "Title": details["title"].values,
                "Year": details["year"].values,
                "Genres": details["genres"].values,
                "Predicted Rating": pred_series.values
            })
            st.write("### 🎯 Recommendations Based on Your Ratings")

            st.dataframe(rec_df, use_container_width=True)

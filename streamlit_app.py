import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from gensim.models.doc2vec import Doc2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Load data and model
df = pd.read_csv("amz_total_data_limited.csv")
model = Doc2Vec.load("doc2vec_model.bin")

# Preprocessing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Clean data
df["title"] = df["title"].fillna("").astype(str)
df["categoryName"] = df["categoryName"].fillna("Unknown")

# Title to index mapping
product_titles = df['title'].tolist()
title_to_index = {title: idx for idx, title in enumerate(product_titles)}

# Streamlit app title
st.title("🛍️ Product Recommender")

# -------------------------------
# 📁 Category Selection (Main Area)
# -------------------------------

st.subheader("📁 Select Product Categories")

# Top categories by popularity
top_categories = (
    df.groupby("categoryName")["boughtInLastMonth"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .index.tolist()
)
all_categories = sorted(df["categoryName"].unique())

# Multiselect for categories
selected_categories = st.multiselect(
    "You can select multiple categories:",
    options=all_categories,
)

# Fallback to top categories
if not selected_categories:
    selected_categories = top_categories
    st.caption("No categories selected — showing top popular categories.")

# Show selected/unselected categories with visual diff
st.write("### 🧾 Selected Categories:")
selected_cols = st.columns(len(selected_categories) if selected_categories else 1)
for i, cat in enumerate(selected_categories):
    with selected_cols[i]:
        st.success(cat)

unselected = [c for c in all_categories if c not in selected_categories]
if unselected:
    st.write("### 📦 Other Categories:")
    unselected_cols = st.columns(min(5, len(unselected)))
    for i, cat in enumerate(unselected):
        with unselected_cols[i % len(unselected_cols)]:
            st.info(cat)

# -------------------------------
# 💲 Price Range Filter
# -------------------------------

price_min = float(df["price"].min())
price_max = float(df["price"].max())
price_range = st.slider("💰 Price Range", min_value=price_min, max_value=price_max, value=(price_min, price_max))

# -------------------------------
# 🔍 Product Selection (Optional)
# -------------------------------

st.subheader("🎯 Choose a Product (Optional)")
selected_title = st.selectbox("Search Product Title", ["None"] + product_titles)
product_index = title_to_index.get(selected_title) if selected_title != "None" else -1

# -------------------------------
# 🧼 Preprocessing
# -------------------------------

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]
    return tokens

# -------------------------------
# 🔁 Recommendation Logic
# -------------------------------

def recommend(product_index=-1, top_n=20, fromvalue=None, tovalue=None, category_list=None):
    full_df = df.copy()

    if product_index != -1:
        inferred_vector = model.dv[str(product_index)]
        similars = model.dv.most_similar([inferred_vector], topn=500)

        sim_df = pd.DataFrame(similars, columns=["model_index", "similarity"])
        sim_df["model_index"] = sim_df["model_index"].astype(int)

        full_df["model_index"] = full_df.index
        merged = sim_df.merge(full_df, on="model_index", how="left")
        merged = merged[merged["categoryName"].isin(category_list)]
    else:
        merged = full_df[full_df["categoryName"].isin(category_list)]

    # Filter by price
    merged["price"] = pd.to_numeric(merged["price"], errors="coerce")
    merged = merged[(merged["price"] >= fromvalue) & (merged["price"] <= tovalue)]

    # Ranking fields
    merged["stars"] = pd.to_numeric(merged["stars"], errors="coerce").fillna(0)
    merged["reviews"] = pd.to_numeric(merged["reviews"], errors="coerce").fillna(0)
    merged["boughtInLastMonth"] = pd.to_numeric(merged["boughtInLastMonth"], errors="coerce").fillna(0)
    merged["reviews_log"] = np.log1p(merged["reviews"])

    if product_index != -1:
        merged["rank_score"] = (
            merged["similarity"] * 0.5 +
            merged["stars"] * 0.2 +
            merged["reviews_log"] * 0.1 +
            merged["boughtInLastMonth"] * 0.1
        )
    else:
        merged["rank_score"] = (
            merged["stars"] * 0.4 +
            merged["reviews_log"] * 0.3 +
            merged["boughtInLastMonth"] * 0.3
        )

    result = merged.sort_values("rank_score", ascending=False).head(top_n)

    return result[[
        "title", "stars", "reviews", "boughtInLastMonth", "rank_score",
        "productURL", "categoryName", "imgUrl", "price"
    ]]

# -------------------------------
# 🔁 Show Doc2Vec Similar Products
# -------------------------------

if product_index != -1:
    st.markdown("### 🔗 Most Similar Products")
    similar_recs = recommend(
        product_index=product_index,
        fromvalue=price_range[0],
        tovalue=price_range[1],
        category_list=selected_categories
    )

    for _, row in similar_recs.iterrows():
        cols = st.columns([1, 3])
        with cols[0]:
            st.image(row['imgUrl'], width=100)
        with cols[1]:
            st.markdown(f"**{row['title']}**")
            st.markdown(f"⭐ {row['stars']} | 💬 {int(row['reviews'])} reviews | 🛒 {int(row['boughtInLastMonth'])} bought")
            st.markdown(f"💲 **{row['price']}**")
            st.markdown(f"[🔗 View Product]({row['productURL']})")
        st.markdown("---")

# -------------------------------
# 🧠 Category-based Recommendations
# -------------------------------

st.markdown("### 🧠 Category-based Recommendations")

category_recs = recommend(
    product_index=-1,
    fromvalue=price_range[0],
    tovalue=price_range[1],
    category_list=selected_categories
)

if not category_recs.empty:
    for _, row in category_recs.iterrows():
        cols = st.columns([1, 3])
        with cols[0]:
            st.image(row['imgUrl'], width=100)
        with cols[1]:
            st.markdown(f"**{row['title']}**")
            st.markdown(f"⭐ {row['stars']} | 💬 {int(row['reviews'])} reviews | 🛒 {int(row['boughtInLastMonth'])} bought")
            st.markdown(f"💲 **{row['price']}**")
            st.markdown(f"[🔗 View Product]({row['productURL']})")
        st.markdown("---")
else:
    st.warning("No products found for the selected filters.")

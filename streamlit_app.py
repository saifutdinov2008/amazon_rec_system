import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from gensim.models.doc2vec import Doc2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import random

# Ensure NLTK data is available
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Load data and model
df = pd.read_csv("amz_total_data_limited.csv")
model = Doc2Vec.load("doc2vec_model.bin")

lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words("english")

st.title("🛍️ Product Recommender")

# --- CATEGORY SELECTION AS BUTTONS ---
categories = sorted(df['categoryName'].dropna().unique())

if "selected_categories" not in st.session_state:
    st.session_state.selected_categories = []

def toggle_category(cat):
    if cat in st.session_state.selected_categories:
        st.session_state.selected_categories.remove(cat)
    else:
        st.session_state.selected_categories.append(cat)

cols = st.columns(5)
for i, cat in enumerate(categories):
    col = cols[i % 5]
    is_selected = cat in st.session_state.selected_categories
    label = f"{cat}"
    if is_selected:
        if col.button(label, key=f"cat_{i}"):
            toggle_category(cat)
    else:
        if col.button(label, key=f"cat_{i}"):
            toggle_category(cat)

selected_categories = st.session_state.selected_categories

# Show top popular categories if none selected
if not selected_categories:
    popular_cats = df['categoryName'].value_counts().nlargest(5).index.tolist()
    selected_categories = popular_cats
    st.write(f"Showing top popular categories (auto-selected): {', '.join(selected_categories)}")
else:
    st.markdown(f"**Selected Categories:** {', '.join(selected_categories)}")

# Price range slider
price_min = float(df["price"].min())
price_max = float(df["price"].max())
price_range = st.slider("💲 Price Range", min_value=price_min, max_value=price_max,
                        value=(price_min, price_max))

# Product selectbox
product_titles = df['title'].fillna("").astype(str).tolist()
title_to_index = {title: idx for idx, title in enumerate(product_titles)}

selected_title = st.selectbox("🔍 Search by Product", ["None"] + product_titles)

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]
    return tokens

def recommend(product_index=-1, top_n=20, fromvalue=None, tovalue=None, category_list=None):
    full_df = df.copy()

    if product_index != -1:
        inferred_vector = model.dv[str(product_index)]
        similars = model.dv.most_similar([inferred_vector], topn=500)

        sim_df = pd.DataFrame(similars, columns=["model_index", "similarity"])
        sim_df["model_index"] = sim_df["model_index"].astype(int)

        full_df["model_index"] = full_df.index
        merged = sim_df.merge(full_df, on="model_index", how="left")

        # Filter by same category as target product
        target_category = df.loc[product_index, "categoryName"]
        merged = merged[merged["categoryName"] == target_category]

    else:
        merged = full_df[full_df["categoryName"].isin(category_list)]

    # Filter by price
    merged["price"] = pd.to_numeric(merged["price"], errors="coerce")
    merged = merged[(merged["price"] >= fromvalue) & (merged["price"] <= tovalue)]

    # Add scores
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

    if product_index != -1:
        return result[[
            "title", "stars", "reviews", "boughtInLastMonth", "similarity", "rank_score",
            "productURL", "categoryName", "imgUrl", "price"
        ]]
    else:
        return result[[
            "title", "stars", "reviews", "boughtInLastMonth", "rank_score",
            "productURL", "categoryName", "imgUrl", "price"
        ]]

def calculate_mrr(model, df, queries, top_n=20):
    reciprocal_ranks = []

    for q_idx in queries:
        inferred_vector = model.dv[str(q_idx)]
        similars = model.dv.most_similar([inferred_vector], topn=top_n)

        query_category = df.loc[q_idx, "categoryName"]

        rank = None
        for i, (rec_idx_str, _) in enumerate(similars, start=1):
            rec_idx = int(rec_idx_str)
            rec_category = df.loc[rec_idx, "categoryName"]
            if rec_category == query_category:
                rank = i
                break

        if rank:
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)

    return np.mean(reciprocal_ranks)

# Show MRR metric
st.subheader("📈 Model Quality Metrics")
sample_queries = random.sample(range(len(df)), 50)
mrr_score = calculate_mrr(model, df, sample_queries, top_n=20)
st.metric("Mean Reciprocal Rank (MRR) @ 20", f"{mrr_score:.3f}")

if selected_title != "None":
    product_idx = title_to_index[selected_title]

    selected_product = df.loc[product_idx]
    st.subheader(f"🛒 Selected Product: **{selected_product['title']}**")
    cols = st.columns([1, 3])
    with cols[0]:
        st.image(selected_product['imgUrl'], width=120)
    with cols[1]:
        st.markdown(f"⭐ {selected_product['stars']} | 💬 {int(selected_product['reviews'])} reviews | 🛒 {int(selected_product['boughtInLastMonth'])} bought")
        st.markdown(f"💲 **{selected_product['price']}**")
        st.markdown(f"[🔗 View Product]({selected_product['productURL']})")
    st.markdown("---")

    recs = recommend(product_index=product_idx, fromvalue=price_range[0], tovalue=price_range[1])
    st.subheader(f"🔁 Products similar to: **{selected_title}**")

    if not recs.empty:
        avg_sim = recs["similarity"].mean()
        top_sim = recs["similarity"].max()
        st.metric("🔗 Avg Similarity (Top 20)", f"{avg_sim:.3f}")
        st.metric("⭐ Top Similarity Score", f"{top_sim:.3f}")

        for _, row in recs.iterrows():
            cols = st.columns([1, 3])
            with cols[0]:
                st.image(row['imgUrl'], width=120)
            with cols[1]:
                st.markdown(f"**{row['title']}**")
                st.markdown(f"⭐ {row['stars']} | 💬 {int(row['reviews'])} reviews | 🛒 {int(row['boughtInLastMonth'])} bought")
                st.markdown(f"💲 **{row['price']}**")
                st.markdown(f"[🔗 View Product]({row['productURL']})")
            st.markdown("---")

elif selected_categories:
    recs = recommend(product_index=-1, category_list=selected_categories,
                     fromvalue=price_range[0], tovalue=price_range[1])
    st.subheader("📁 Recommendations from selected categories")
    if not recs.empty:
        for _, row in recs.iterrows():
            cols = st.columns([1, 3])
            with cols[0]:
                st.image(row['imgUrl'], width=120)
            with cols[1]:
                st.markdown(f"**{row['title']}**")
                st.markdown(f"⭐ {row['stars']} | 💬 {int(row['reviews'])} reviews | 🛒 {int(row['boughtInLastMonth'])} bought")
                st.markdown(f"💲 **{row['price']}**")
                st.markdown(f"[🔗 View Product]({row['productURL']})")
            st.markdown("---")

else:
    st.info("Select a product or choose categories above to get recommendations.")

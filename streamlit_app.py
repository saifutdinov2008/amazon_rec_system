import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from gensim.models.doc2vec import Doc2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data is available
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('punkt_tab')


# Load data and model
df = pd.read_csv("amz_total_data_limited.csv")
model = Doc2Vec.load("doc2vec_model.bin")

# Preprocessing tools
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words("english")

# Sidebar: Select product or category
st.title("🛍️ Product Recommender")

st.sidebar.header("🔧 Filter Options")
product_titles = df['title'].fillna("").astype(str).tolist()
title_to_index = {title: idx for idx, title in enumerate(product_titles)}

selected_title = st.sidebar.selectbox("🔍 Search by Product", ["None"] + product_titles)
categories = sorted(df['categoryName'].dropna().unique())
selected_categories = st.sidebar.multiselect("📁 Or Select Categories", categories)

# Price range
price_min = float(df["price"].min())
price_max = float(df["price"].max())
price_range = st.sidebar.slider("💲 Price Range", min_value=price_min, max_value=price_max, value=(0.0, 13376.0))

# Clean text
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]
    return tokens

# Recommendation logic
def recommend(product_index=-1, top_n=20, fromvalue=None, tovalue=None, category_list=None):
    full_df = df.copy()

    if product_index != -1:
        inferred_vector = model.dv[str(product_index)]
        similars = model.dv.most_similar([inferred_vector], topn=500)

        sim_df = pd.DataFrame(similars, columns=["model_index", "similarity"])
        sim_df["model_index"] = sim_df["model_index"].astype(int)

        full_df["model_index"] = full_df.index
        merged = sim_df.merge(full_df, on="model_index", how="left")

        # Optional: Filter by same category
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

    return result[[
        "title", "stars", "reviews", "boughtInLastMonth", "rank_score",
        "productURL", "categoryName", "imgUrl", "price"
    ]]

# Run logic
if selected_title != "None":
    product_idx = title_to_index[selected_title]
    recs = recommend(product_index=product_idx, fromvalue=price_range[0], tovalue=price_range[1])
    st.subheader(f"🔁 Products similar to: **{selected_title}**")
elif selected_categories:
    recs = recommend(product_index=-1, category_list=selected_categories,
                     fromvalue=price_range[0], tovalue=price_range[1])
    st.subheader("📁 Recommendations from selected categories")
else:
    st.info("Select a product or choose categories from the sidebar to get recommendations.")
    recs = None

# Display recommendations
if recs is not None and not recs.empty:
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

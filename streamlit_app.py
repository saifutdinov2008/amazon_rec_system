import streamlit as st
import re
import nltk
import numpy as np
import pandas as pd
import matplotlib as plt
from tqdm import tqdm_notebook
from ast import literal_eval
from pymystem3 import Mystem
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

nltk.download('stopwords')


# Load dataset
df = pd.read_csv("amz_total_data_limited.csv")



# Clean and prepare title list for dropdown
product_title_list = df['title'].fillna("").tolist()
title_to_index = {title: idx for idx, title in enumerate(product_title_list)}

# Top categories
categories = [
    "Baby", "Luggage  Travel Gear", "Handmade Home Décor", "Handmade Kitchen  Dining",
    "Perfume  Cologne", "Beauty", "Men's Shoes", "Men's Watches", "Handmade Jewellery",
    "Boys", "Men's Jewelry", "Women's Watches", "Uniforms, Work  Safety", "Women's Shoes",
    "Shaving  Hair Removal Products", "Bath  Body", "Electronics", "Salon  Spa Equipment",
    "Grocery", "Beauty Tools  Accessories", "Men", "Nail Polish  Nail Decoration Products",
    "Women's Handbags", "Women"
]

# App title
st.title("🛍️ Product Recommender")

# Sidebar filters
st.sidebar.header("Filter Options")

# Product selector
selected_product_title = st.sidebar.selectbox("🔍 Or Search by Product", options=["None"] + product_title_list)

# Category selector (if product is not chosen)
selected_categories = st.sidebar.multiselect("📁 Or Select Categories", options=categories)

# Price filter
price_min, price_max = float(df["price"].min()), float(df["price"].max())
price_range = st.sidebar.slider("💲 Price Range", min_value=0.0, max_value=float(price_max),
                                value=(10.0, 70.0), step=1.0)


# init lemmatizer to avoid slow performance
mystem = Mystem() 

def word_tokenize_clean(doc: str, stop_words: list):
    '''
    tokenize from string to list of words
    '''

    # split into lower case word tokens \w lemmatization
    tokens = list(set(mystem.lemmatize(doc.lower())))
  
    # remove tokens that are not alphabetic (including punctuation) and not a stop word
    tokens = [word for word in tokens if word.isalpha() and not word in stop_words \
              not in list(punctuation)]
    return tokens

tags_corpus = df['title'].values
tags_corpus = [re.sub('[!/()0-9]', '', str(x)) for x in tags_corpus]
stop_words = stopwords.words('english')

tags_doc = [TaggedDocument(words = word_tokenize_clean(D, stop_words), tags = [str(i)]) for i, D in enumerate(tags_corpus)]

VEC_SIZE = 50 # длина вектора для каждого фильма
ALPHA = .02 # параметр обучения модели
MIN_ALPHA = .00025 # параметр обучения модели
MIN_COUNT = 4 # минимальное количество вхождений слова в словарь
EPOCHS = 20 # количество тренировок

# инициализировать модель
model = Doc2Vec(vector_size = VEC_SIZE,
                alpha = ALPHA, 
                min_alpha = MIN_ALPHA,
                min_count = MIN_COUNT,
                dm = 0)

# создайте словарь из всех документов по тегам
model.build_vocab(tags_doc)

# обучить модель
model.train(tags_doc,
            total_examples = model.corpus_count,
            epochs = EPOCHS)


def recommend(product_index: int = -1, top_n: int = 20, fromvalue: float = None, tovalue: float = None, category_list: list = None):
    if product_index != -1:   
        # Step 1: Get embeddings
        products_embeddings = model.dv[str(product_index)]

        # Step 2: Get broader pool of similar products
        similars = model.dv.most_similar(positive=[products_embeddings], topn=500)
        similar_df = pd.DataFrame(similars, columns=['model_index', 'model_score'])
        similar_df['model_index'] = similar_df['model_index'].astype(int)

        # Step 3: Merge with product metadata
        full_df = df.copy()
        full_df['model_index'] = full_df.index
        merged = pd.merge(similar_df, full_df, on='model_index', how='left')

        # Step 4: Filter by same category
        target_category = df.loc[product_index, 'categoryName']
        merged = merged[merged['categoryName'] == target_category]

        # Step 5: Convert numeric fields
        merged['price'] = pd.to_numeric(merged['price'], errors='coerce')
        merged['stars'] = pd.to_numeric(merged['stars'], errors='coerce').fillna(0)
        merged['reviews'] = pd.to_numeric(merged['reviews'], errors='coerce').fillna(0)
        merged['boughtInLastMonth'] = pd.to_numeric(merged['boughtInLastMonth'], errors='coerce').fillna(0)

        # Step 6: Optional price filter
        if fromvalue is not None and tovalue is not None:
            merged = merged[(merged['price'] >= fromvalue) & (merged['price'] <= tovalue)]

        # Step 7: Add reviews_log
        merged['reviews_log'] = np.log1p(merged['reviews'])

        # Step 8: Rank scoring
        merged['rank_score'] = (
            merged['model_score'] * 0.5 +
            merged['stars'] * 0.2 +
            merged['reviews_log'] * 0.1 +
            merged['boughtInLastMonth'] * 0.1
        )

        # Step 9: Sort and return
        result = merged.sort_values(by='rank_score', ascending=False).head(top_n)

        return result[[
            'title', 'stars', 'reviews', 'boughtInLastMonth',
            'rank_score', 'productURL', 'categoryName', 'imgUrl', 'price'
        ]]
    
    else:
        # fallback for cold-start (no product index) — based on category
        full_df = df.copy()
        filtered = full_df[full_df['categoryName'].isin(category_list)]

        filtered['price'] = pd.to_numeric(filtered['price'], errors='coerce')
        filtered['stars'] = pd.to_numeric(filtered['stars'], errors='coerce').fillna(0)
        filtered['reviews'] = pd.to_numeric(filtered['reviews'], errors='coerce').fillna(0)
        filtered['boughtInLastMonth'] = pd.to_numeric(filtered['boughtInLastMonth'], errors='coerce').fillna(0)
        filtered['reviews_log'] = np.log1p(filtered['reviews'])

        if fromvalue is not None and tovalue is not None:
            filtered = filtered[(filtered['price'] >= fromvalue) & (filtered['price'] <= tovalue)]

        filtered['rank_score'] = (
            filtered['stars'] * 0.4 +
            filtered['reviews_log'] * 0.3 +
            filtered['boughtInLastMonth'] * 0.3
        )

        result = filtered.sort_values(by='rank_score', ascending=False).head(top_n)

        return result[[
            'title', 'stars', 'reviews', 'boughtInLastMonth',
            'rank_score', 'productURL', 'categoryName', 'imgUrl', 'price'
        ]]


# Main logic
if selected_product_title != "None":
    product_index = title_to_index[selected_product_title]
    recommendations = recommend(product_index=product_index)
    st.subheader(f"🔁 Products similar to: **{selected_product_title}**")
elif selected_categories:
    recommendations = recommend(product_index=-1, category_list=selected_categories,
                                 fromvalue=price_range[0], tovalue=price_range[1])
    st.subheader("📁 Recommendations from selected categories")
else:
    st.info("Select a product or choose categories from the sidebar to get recommendations.")
    recommendations = None

# Display recommendations
if recommendations is not None and not recommendations.empty:
    for _, row in recommendations.iterrows():
        cols = st.columns([1, 3])
        with cols[0]:
            st.image(row['imgUrl'], width=120)
        with cols[1]:
            st.markdown(f"**{row['title']}**")
            st.markdown(f"⭐ {row['stars']} | 💬 {int(row['reviews'])} reviews | 🛒 {int(row['boughtInLastMonth'])} bought")
            st.markdown(f"💲 **{row['price']}**")
            st.markdown(f"[🔗 View Product]({row['productURL']})")
        st.markdown("---")

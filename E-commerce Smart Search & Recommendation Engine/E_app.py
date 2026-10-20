import streamlit as st
import pandas as pd
from pathlib import Path
import streamlit as st
import sys
import os
from pathlib import Path



current_dir = os.getcwd()
sys.path.append(current_dir)
# 2. Now import your logic file
import E_utlis as utils

# --- 1. Page settings and dark CSS ---
st.set_page_config(page_title="Smart E-Commerce", page_icon="🛒", layout="wide")

# Custom CSS for Dark Theme and Beautiful Product Cards
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #121212;
        color: #E0E0E0;
    }
    
    /* Product Card Styling */
    .product-card {
        background-color: #1E1E1E;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #333;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0, 191, 165, 0.2);
        border-color: #00BFA5;
    }
    
    /* Text Styling */
    .product-title {
        color: #FFFFFF;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .product-desc {
        color: #A0A0A0;
        font-size: 0.9rem;
        margin-bottom: 15px;
        line-height: 1.5;
    }
    
    /* Prices and Badges */
    .price-container {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 15px;
    }
    .discount-price {
        color: #00BFA5;
        font-size: 1.4rem;
        font-weight: bold;
    }
    .actual-price {
        color: #757575;
        text-decoration: line-through;
        font-size: 1rem;
    }
    .badge {
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .badge-discount {
        background-color: #D32F2F;
        color: white;
    }
    .badge-rating {
        background-color: #FFB300;
        color: #212121;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. Load data and model (use cache to make it fast) ---
utils.setup_nltk()

@st.cache_data
def fetch_data():
    return utils.load_data()

@st.cache_resource
def fetch_model(df):
    return utils.build_model(df)

df = fetch_data()
vectorizer, tfidf_matrix = fetch_model(df)

# --- 3. Function to draw the product card ---
def render_product_card(row):
    # Make description short if it is too long
    desc = str(row['about_product'])[:150] + "..." if len(str(row['about_product'])) > 150 else str(row['about_product'])
    
    html = f"""
    <div class="product-card">
        <div class="product-title">{row['product_name']}</div>
        <div class="price-container">
            <span class="discount-price">{row['discounted_price']}</span>
            <span class="actual-price">{row['actual_price']}</span>
            <span class="badge badge-discount">Save {row['discount_percentage']}</span>
            <span class="badge badge-rating">⭐ {row['rating']}</span>
        </div>
        <div class="product-desc">{desc}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# --- 4. Main App UI ---
st.title("🛒 AI E-Commerce Search & Recommend")

tab1, tab2 = st.tabs(["🔍 Smart Search", "💡 Auto-Suggest & Recommend"])

# First Tab: Search
with tab1:
    search_query = st.text_input("What are you looking for? (e.g., 'fast charging cable iphone 13')")
    
    if search_query:
        # Call the search function from utils file
        top_indices, similarities = utils.get_search_results(search_query, vectorizer, tfidf_matrix, top_n=3)
        
        # Check if we have good results
        if similarities[top_indices[0]] > 0:
            st.subheader(f"🎯 Search Results for: '{search_query}'")
            cols = st.columns(3)
            
            # Show the top 3 search results
            for i, idx in enumerate(top_indices):
                with cols[i]:
                    render_product_card(df.iloc[idx])
            
            st.divider()
            
            # Get recommendations for the first search result
            best_match_idx = top_indices[0]
            st.subheader(f"✨ Because you searched for this, you might also like:")
            rec_indices = utils.get_recommendations(best_match_idx, tfidf_matrix, top_n=4)
            
            # Show 4 recommended products
            rec_cols = st.columns(4)
            for i, idx in enumerate(rec_indices):
                with rec_cols[i]:
                    render_product_card(df.iloc[idx])
        else:
            st.warning("No matching products found. Try different keywords.")

# Second Tab: Auto-Suggest
with tab2:
    st.markdown("Use the dropdown below. As you type, it will suggest products automatically.")
    
    # Put all product names in a list for the select box
    product_list = df['product_name'].tolist()
    selected_product = st.selectbox("Type a product name:", [""] + product_list)
    
    if selected_product:
        st.subheader("💡 Similar Products Recommended for You:")
        
        # Find the index of the selected product
        product_idx = df[df['product_name'] == selected_product].index[0]
        
        # Call recommendation function
        rec_indices = utils.get_recommendations(product_idx, tfidf_matrix, top_n=4)
        
        # Show the recommendations
        cols = st.columns(4)
        for i, idx in enumerate(rec_indices):
            with cols[i]:
                render_product_card(df.iloc[idx])
import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
current_dir = os.getcwd()
sys.path.append(current_dir)
import glob

def setup_nltk():
    """Download NLTK files only one time."""
    nltk.download('stopwords', quiet=True)

def clean_ecommerce_text(content: str) -> str:
    """Clean the text but keep numbers and models."""
    # Keep english letters and numbers, then make it lower case
    cleaned = re.sub('[^a-zA-Z0-9]', ' ', str(content)).lower()
    
    # Remove stop words like 'in', 'the', 'and'
    stop_words = set(stopwords.words('english'))
    cleaned = [word for word in cleaned.split() if word not in stop_words]
    
    return ' '.join(cleaned)

def load_data() -> pd.DataFrame:
    """Load the dataset safely by searching for it anywhere nearby."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Try direct path first
    file_path = os.path.join(current_dir, 'amazon.csv')
    
    # 2. If not found, search recursively in the current directory and parent folders
    if not os.path.exists(file_path):
        search_pattern = os.path.join(current_dir, '**', 'amazon.csv')
        found_files = glob.glob(search_pattern, recursive=True)
        
        if found_files:
            file_path = found_files[0]
        else:
            # Look for any csv file as a fallback
            all_csvs = glob.glob(os.path.join(current_dir, '**', '*.csv'), recursive=True)
            if all_csvs:
                file_path = all_csvs[0]
            else:
                raise FileNotFoundError("Could not find 'amazon.csv' anywhere in the project folders!")
                
    df = pd.read_csv(file_path)
    df = df.drop_duplicates(subset=['product_name','about_product']).reset_index(drop=True)
    
    # Fill missing values with empty space to avoid errors
    df['product_name'] = df['product_name'].fillna('')
    df['about_product'] = df['about_product'].fillna('')
    
    # Combine the name and description, then apply the cleaning function
    df['product_content'] = df['product_name'] + " " + df['about_product']
    df['clean_content'] = df['product_content'].apply(clean_ecommerce_text)
    
    return df

def build_model(df: pd.DataFrame):
    """Build the TF-IDF matrix."""
    # Ignore very common words (90%) and very rare words (less than 2 times)
    vectorizer = TfidfVectorizer(max_df=0.90, min_df=2)
    tfidf_matrix = vectorizer.fit_transform(df['clean_content'])
    return vectorizer, tfidf_matrix

def get_search_results(query: str, vectorizer, tfidf_matrix, top_n: int = 3):
    """Get the best matching products for the user search."""
    # Convert user query to vector
    query_vec = vectorizer.transform([clean_ecommerce_text(query)])
    
    # Calculate similarity
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Sort and get the best results
    top_indices = similarities.argsort()[-top_n:][::-1]
    return top_indices, similarities

def get_recommendations(target_idx: int, tfidf_matrix, top_n: int = 4):
    """Get similar products to recommend to the user."""
    target_vec = tfidf_matrix[target_idx]
    similarities = cosine_similarity(target_vec, tfidf_matrix).flatten()
    
    # Skip the first result because it is the exact same product [1:]
    top_indices = similarities.argsort()[-(top_n + 1):][::-1][1:]
    return top_indices
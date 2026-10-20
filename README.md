# 🛒 AI-Powered E-Commerce Search & Recommendation Engine

An intelligent e-commerce product search and recommendation system built with Python, Natural Language Processing (NLP), and Streamlit. The application leverages TF-IDF vectorization and cosine similarity to deliver precise product matching and context-aware recommendations based on product metadata and descriptions.

---

## 🌟 Key Features

- **🔍 Smart Semantic Search:** Delivers accurate product search results by analyzing product titles and detailed descriptions using TF-IDF vectorization and Cosine Similarity.
- **💡 Automated Product Recommendations:** Generates relevant product suggestions based on user search queries or explicitly selected items.
- **⚡ Custom NLP Pipeline:** Includes automated text preprocessing with `NLTK` to clean data, filter stop words, and normalize text while preserving product numbers and specifications.
- **🖥️ Modern Streamlit Interface:** Interactive web interface with dark mode styling, custom product cards, price savings tags, rating badges, and tabbed navigation.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Web Framework:** Streamlit
- **Machine Learning & NLP:** Scikit-Learn (`TfidfVectorizer`, `cosine_similarity`), NLTK
- **Data Manipulation:** Pandas

---

## 📂 Project Structure

```text
.
├── E_app.py          # Main Streamlit application and UI layout
├── E_utlis.py        # Core NLP utilities, dataset loading, and search/recommendation algorithms
├── amazon.csv        # Product dataset
└── requirements.txt  # Required Python dependencies

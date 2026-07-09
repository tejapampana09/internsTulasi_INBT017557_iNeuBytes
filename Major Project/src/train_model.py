import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, '../data/movies_enriched.csv')
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run data_prep.py first.")
        return
        
    print("Loading enriched dataset...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} movies.")
    
    # Fill missing values
    df['title'] = df['title'].fillna('Unknown')
    df['overview'] = df['overview'].fillna('')
    df['genres'] = df['genres'].fillna('')
    df['vote_average'] = df['vote_average'].fillna(0.0)
    
    # Feature engineering for content-based matching
    print("Extracting features...")
    df['clean_title'] = df['title'].apply(clean_text)
    df['features'] = df['genres'] + " " + df['overview']
    df['clean_features'] = df['features'].apply(clean_text)
    
    # TF-IDF Vectorization
    print("Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(df['clean_features'])
    
    # Calculate cosine similarity matrix
    print("Calculating Cosine Similarity matrix...")
    similarity = cosine_similarity(tfidf_matrix)
    
    # Build a lookup dictionary: movie_id -> list of top 10 recommended movie indices
    print("Pre-computing top-10 recommendations lookup table...")
    recommendations_lookup = {}
    
    for idx in range(len(df)):
        # Sort similarity scores and get top indices (excluding itself)
        sim_scores = list(enumerate(similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Select top 10 similar movies (skipping the first one which is itself)
        top_similar = [i for i, score in sim_scores[1:11]]
        
        movie_id = int(df.iloc[idx]['id'])
        recommendations_lookup[movie_id] = top_similar
        
    # Serialize the models
    model_dir = os.path.join(base_dir, '../model')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'recommendation_model.pkl')
    
    # Save lookup dictionary & lightweight movie metadata
    print("Saving serialized model objects...")
    
    # Keep only required columns in metadata for the web app to minimize payload size
    metadata_df = df[['id', 'title', 'genres', 'overview', 'vote_average']].copy()
    
    # Save lookup dictionary and metadata DataFrame
    with open(model_path, 'wb') as f:
        pickle.dump({
            'metadata': metadata_df,
            'lookup': recommendations_lookup,
            'vectorizer': vectorizer
        }, f)
        
    print("Model trained and saved successfully.")

if __name__ == "__main__":
    main()

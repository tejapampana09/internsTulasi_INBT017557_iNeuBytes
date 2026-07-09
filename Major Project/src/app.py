import os
import pickle
import difflib
from flask import Flask, request, jsonify

app = Flask(__name__, static_folder='../static', static_url_path='')

# Global variable to hold loaded model artifacts
model_artifacts = None

def load_model():
    global model_artifacts
    model_path = os.path.join(os.path.dirname(__file__), '../model/recommendation_model.pkl')
    
    if not os.path.exists(model_path):
        print(f"WARNING: Model file not found at {model_path}. Please train the model first.")
        return False
        
    try:
        with open(model_path, 'rb') as f:
            model_artifacts = pickle.load(f)
        print("Model artifacts successfully loaded in memory.")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Load model once at server startup
load_model()

@app.route('/')
def home():
    # Serves the landing page of our web app
    return app.send_static_file('index.html')

@app.route('/health', methods=['GET'])
def health():
    # Required health endpoint
    if model_artifacts is not None:
        return jsonify({
            "status": "OK",
            "model": "TF-IDF Content-Based Filtering",
            "dataset_movies_count": len(model_artifacts['metadata']),
            "message": "Full-stack Recommendation service is running."
        }), 200
    else:
        return jsonify({
            "status": "ERROR",
            "message": "Model artifacts not loaded."
        }), 500

@app.route('/movies', methods=['GET'])
def get_movies():
    # Helper endpoint to provide autocomplete suggestions to the frontend
    if model_artifacts is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    metadata = model_artifacts['metadata']
    movies_list = [{"id": int(row['id']), "title": str(row['title'])} for _, row in metadata.iterrows()]
    return jsonify(movies_list), 200

@app.route('/recommend', methods=['POST'])
def recommend():
    # Required recommendation endpoint
    if model_artifacts is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    try:
        data = request.get_json(force=True, silent=True)
        if not data or 'movie_title' not in data:
            return jsonify({
                "error": "Bad Request",
                "message": "Invalid input format. Request body must be JSON and contain 'movie_title'."
            }), 400
            
        query_title = data['movie_title'].strip()
        if not query_title:
            return jsonify({
                "error": "Validation Error",
                "message": "Movie title cannot be empty."
            }), 400
            
        metadata = model_artifacts['metadata']
        lookup = model_artifacts['lookup']
        
        # Match using difflib for robust close-match search
        all_titles = metadata['title'].tolist()
        close_matches = difflib.get_close_matches(query_title, all_titles, n=1, cutoff=0.5)
        
        if not close_matches:
            return jsonify({
                "error": "Not Found",
                "message": f"No matches found for '{query_title}'. Please try another search term."
            }), 404
            
        matched_title = close_matches[0]
        # Find row corresponding to matched title
        matched_row = metadata[metadata['title'] == matched_title].iloc[0]
        matched_id = int(matched_row['id'])
        
        # Get recommendations
        recommended_indices = lookup.get(matched_id, [])
        recommendations = []
        
        for idx in recommended_indices:
            rec_row = metadata.iloc[idx]
            recommendations.append({
                "id": int(rec_row['id']),
                "title": str(rec_row['title']),
                "genres": str(rec_row['genres']),
                "overview": str(rec_row['overview']),
                "vote_average": float(rec_row['vote_average'])
            })
            
        return jsonify({
            "query_submitted": query_title,
            "matched_title": matched_title,
            "recommendations_count": len(recommendations),
            "recommendations": recommendations
        }), 200
        
    except Exception as e:
        # Robust error handling: prevent server crash and return clean JSON
        return jsonify({
            "error": "Internal Server Error",
            "message": f"An error occurred while processing your request: {str(e)}"
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

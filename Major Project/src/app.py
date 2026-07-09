import os
import pickle
import difflib
import requests
import json
from flask import Flask, request, jsonify

app = Flask(__name__, static_folder='../static', static_url_path='')

# Global variable to hold loaded model artifacts
model_artifacts = None
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

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
    # Required health endpoint with live diagnostics
    gemini_active = GEMINI_API_KEY is not None and len(GEMINI_API_KEY.strip()) > 0
    openrouter_active = OPENROUTER_API_KEY is not None and len(OPENROUTER_API_KEY.strip()) > 0
    gemini_test_status = "N/A"
    gemini_test_error = "N/A"
    openrouter_test_status = "N/A"
    openrouter_test_error = "N/A"
    
    if gemini_active:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
            res = requests.post(url, json={"contents": [{"parts": [{"text": "Hello"}]}]}, timeout=5.0)
            gemini_test_status = res.status_code
            if res.status_code != 200:
                gemini_test_error = res.text[:200]
        except Exception as e:
            gemini_test_error = str(e)
            
    if openrouter_active:
        try:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }
            res = requests.post(url, json={
                "model": "meta-llama/llama-3-8b-instruct:free",
                "messages": [{"role": "user", "content": "Hello"}]
            }, headers=headers, timeout=5.0)
            openrouter_test_status = res.status_code
            if res.status_code != 200:
                openrouter_test_error = res.text[:200]
        except Exception as e:
            openrouter_test_error = str(e)
            
    if model_artifacts is not None:
        return jsonify({
            "status": "OK",
            "model_fallback": "SentenceTransformer Semantic Search",
            "gemini_api_integrated": gemini_active,
            "gemini_test_status": gemini_test_status,
            "gemini_test_error": gemini_test_error,
            "openrouter_api_integrated": openrouter_active,
            "openrouter_test_status": openrouter_test_status,
            "openrouter_test_error": openrouter_test_error,
            "dataset_movies_count": len(model_artifacts['metadata']),
            "message": "Full-stack hybrid recommendation service is running."
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
    movies_list = [{"id": int(row['id']), "title": str(row['title'])} for row in metadata]
    return jsonify(movies_list), 200

def get_gemini_recommendations(movie_title):
    # Calls Google's Gemini API for dynamic LLM movie recommendations
    if not GEMINI_API_KEY:
        return None
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    prompt = (
        f"Recommend 10 movies similar to the movie \"{movie_title}\". "
        "You must return a JSON array of objects. Each object in the array must contain the keys: "
        "\"id\" (integer, starting from 1), \"title\" (string), \"genres\" (JSON array of strings), "
        "\"overview\" (string), and \"vote_average\" (float from 1.0 to 10.0)."
    )
    
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "responseMimeType": "application/json"
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=15.0)
        if response.status_code == 200:
            res_data = response.json()
            generated_text = res_data['candidates'][0]['content']['parts'][0]['text']
            recommendations = json.loads(generated_text)
            
            # Ensure the output is indeed a list of recommendations
            if isinstance(recommendations, list):
                return recommendations
            elif isinstance(recommendations, dict) and 'recommendations' in recommendations:
                return recommendations['recommendations']
        print(f"Gemini API returned status code {response.status_code}: {response.text}")
        return None
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return None

def get_pollinations_recommendations(movie_title):
    # Calls Pollinations AI whitelisted text API for keyless LLM recommendations
    url = "https://text.pollinations.ai/"
    
    prompt = (
        f"Recommend 10 movies similar to the movie \"{movie_title}\". "
        "You must return ONLY a JSON array of objects, with no markdown code block formatting (no ```json or ```). "
        "Each object in the array must contain the keys: "
        "\"id\" (integer, starting from 1), \"title\" (string), \"genres\" (JSON array of strings), "
        "\"overview\" (string), and \"vote_average\" (float from 1.0 to 10.0)."
    )
    
    payload = {
        "messages": [
            {"role": "system", "content": "You are a movie recommendation assistant. You return ONLY a raw JSON array of recommendation objects, with no markdown formatting, backticks, or conversational introduction."},
            {"role": "user", "content": prompt}
        ]
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30.0)
        if response.status_code == 200:
            generated_text = response.text.strip()
            
            # Clean markdown code blocks if present
            if "```" in generated_text:
                parts = generated_text.split("```")
                # find the part that has JSON
                for part in parts:
                    part_clean = part.strip()
                    if part_clean.startswith("json"):
                        part_clean = part_clean[4:].strip()
                    if part_clean.startswith("["):
                        generated_text = part_clean
                        break
            
            recommendations = json.loads(generated_text)
            if isinstance(recommendations, list):
                return recommendations
        print(f"Pollinations AI returned status code {response.status_code}: {response.text}")
        return None
    except Exception as e:
        print(f"Error calling Pollinations AI: {e}")
        return None

def get_openrouter_recommendations(movie_title):
    # Calls OpenRouter API for Llama-3-8B-Instruct recommendations
    if not OPENROUTER_API_KEY:
        return None
        
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    prompt = (
        f"Recommend 10 movies similar to the movie \"{movie_title}\". "
        "You must return ONLY a JSON array of objects, with no markdown code block formatting (no ```json or ```). "
        "Each object in the array must contain the keys: "
        "\"id\" (integer, starting from 1), \"title\" (string), \"genres\" (JSON array of strings), "
        "\"overview\" (string), and \"vote_average\" (float from 1.0 to 10.0)."
    )
    
    payload = {
        "model": "meta-llama/llama-3-8b-instruct:free",
        "messages": [
            {"role": "system", "content": "You are a movie recommendation assistant. You return ONLY a raw JSON array of recommendation objects. Do not include markdown formatting or backticks."},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"}
    }
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=15.0)
        if response.status_code == 200:
            res_data = response.json()
            generated_text = res_data['choices'][0]['message']['content'].strip()
            
            # Clean markdown code blocks if present
            if "```" in generated_text:
                parts = generated_text.split("```")
                for part in parts:
                    part_clean = part.strip()
                    if part_clean.startswith("json"):
                        part_clean = part_clean[4:].strip()
                    if part_clean.startswith("["):
                        generated_text = part_clean
                        break
            
            recommendations = json.loads(generated_text)
            if isinstance(recommendations, list):
                return recommendations
            elif isinstance(recommendations, dict) and 'recommendations' in recommendations:
                return recommendations['recommendations']
        print(f"OpenRouter API returned status code {response.status_code}: {response.text}")
        return None
    except Exception as e:
        print(f"Error calling OpenRouter API: {e}")
        return None

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
        all_titles = [m['title'] for m in metadata]
        close_matches = difflib.get_close_matches(query_title, all_titles, n=1, cutoff=0.5)
        
        if not close_matches:
            return jsonify({
                "error": "Not Found",
                "message": f"No matches found for '{query_title}'. Please try another search term."
            }), 404
            
        matched_title = close_matches[0]
        
        # ----------------------------------------------------
        # HYBRID DYNAMIC AI GENERATION INTEGRATION
        # ----------------------------------------------------
        recommendations = None
        source_used = "SentenceTransformer Local Lookup"
        
        # 1. Try Google Gemini API (if key is set)
        if GEMINI_API_KEY:
            print(f"Attempting to fetch Generative AI recommendations from Gemini for '{matched_title}'...")
            recommendations = get_gemini_recommendations(matched_title)
            if recommendations:
                source_used = "Google Gemini 2.5 Generative AI API"
                print("Successfully retrieved recommendations from Gemini API.")
                
        # 2. Try OpenRouter API (if key is set)
        if not recommendations and OPENROUTER_API_KEY:
            print(f"Attempting to fetch Generative AI recommendations from OpenRouter for '{matched_title}'...")
            recommendations = get_openrouter_recommendations(matched_title)
            if recommendations:
                source_used = "OpenRouter Llama-3-8B Generative AI API"
                print("Successfully retrieved recommendations from OpenRouter API.")

        # 3. Try Pollinations AI Whitelisted LLM (if Gemini/OpenRouter failed or were not configured)
        if not recommendations:
            print(f"Attempting to fetch Generative AI recommendations from Pollinations AI for '{matched_title}'...")
            recommendations = get_pollinations_recommendations(matched_title)
            if recommendations:
                source_used = "Pollinations AI (Llama-3/Qwen) Whitelisted LLM"
                print("Successfully retrieved recommendations from Pollinations AI.")
                
        # 3. Fallback to local lookup if all LLM options failed
        if not recommendations:
            print("Falling back to local SentenceTransformer lookup dictionary...")
            matched_row = next(m for m in metadata if m['title'] == matched_title)
            matched_id = int(matched_row['id'])
            recommended_indices = lookup.get(matched_id, [])
            
            recommendations = []
            for idx in recommended_indices:
                rec_row = metadata[idx]
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
            "recommendations_source": source_used,
            "recommendations_count": len(recommendations),
            "recommendations": recommendations
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": "Internal Server Error",
            "message": f"An error occurred while processing your request: {str(e)}"
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

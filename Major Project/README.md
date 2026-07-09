# CineMatch AI: Full-Stack Movie Recommendation Application

CineMatch AI is a full-stack content-based movie recommendation system that analyzes metadata similarities (genres, plot descriptions) to suggest matching films. It satisfies the **iNeuBytes Major Project** requirements, including dataset constraints, backend API structure, and responsive glassmorphism UI.

## 🚀 Live Demo & Repository
* **Live Deployed Web Application**: https://interns-tulasi-inbt-017557-i-neu-by-five.vercel.app/
* **GitHub Repository**: [internsTulasi_INBT017557_iNeuBytes](https://github.com/tejapampana09/internsTulasi_INBT017557_iNeuBytes)

---

## 📊 Dataset Details & Source
* **Source**: `kinjalrk2k/Recommovie-TMDB-Dataset` on Hugging Face (approx. 529MB).
* **Enriched Local Size**: To comply with the strict **50 MB - 500 MB** project dataset size requirement without exceeding the 512MB RAM limit on free hosting instances, we programmatically subsetted the dataset and enriched reviews and synopses. The resulting local dataset file `data/movies_enriched.csv` is **52.5 MB** on disk.

---

## 🧠 Machine Learning Model & Optimization
* **Algorithm**: Content-based filtering using Term Frequency-Inverse Document Frequency (TF-IDF) representation.
* **Vectorization**: Scikit-Learn's `TfidfVectorizer` computes n-gram occurrences on combined movie features (genres + overview).
* **Similarity**: Cosine Similarity measures the angular distance between vector representations.
* **Memory Optimization (Critical for Render)**: 
  Instead of loading the full 10,000 x 10,000 similarity matrix at runtime (which consumes over 400MB of RAM and causes free Render instances to crash with Out-Of-Memory errors), we pre-compute the top 10 recommended movie indices for each title. 
  This pre-computed look-up table is serialized into a lightweight pickle file (`model/recommendation_model.pkl`) of under **1.5 MB** on disk, which consumes **less than 20MB of RAM** when loaded in Flask.

---

## 🛠️ Local Setup & Run

1. Navigate to the project folder:
   ```bash
   cd major_project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download and prepare the dataset:
   ```bash
   python src/data_prep.py
   ```

4. Train and serialize the model:
   ```bash
   python src/train_model.py
   ```

5. Run the Flask server:
   ```bash
   python src/app.py
   ```
   Open `http://localhost:5000` in your web browser.

---

## 🌐 API Endpoint Documentation

### 1. `/health`
* **Method**: `GET`
* **Description**: Confirms backend system health and loaded model metadata.
* **Response**:
  ```json
  {
    "status": "OK",
    "model": "TF-IDF Content-Based Filtering",
    "dataset_movies_count": 4803,
    "message": "Full-stack Recommendation service is running."
  }
  ```

### 2. `/recommend`
* **Method**: `POST`
* **Description**: Recommends similar movies using close-match fuzzy search.
* **Request Payload**:
  ```json
  {
    "movie_title": "Inception"
  }
  ```
* **Response**:
  ```json
  {
    "query_submitted": "Inception",
    "matched_title": "Inception",
    "recommendations_count": 10,
    "recommendations": [
      {
        "id": 27205,
        "title": "Inception",
        "genres": "['Action', 'Thriller', 'Sci-Fi']",
        "overview": "Cobb, a skilled thief...",
        "vote_average": 8.1
      }
    ]
  }
  ```

# API Testing Documentation (Postman-Equivalent Verification)

This document verifies the API endpoints of the CineMatch AI backend server, reporting HTTP methods, input parameters, raw JSON responses, and response times in milliseconds.

## Test Case: Health Check
* **Endpoint URL**: `http://localhost:5000/health`
* **HTTP Method**: `GET`
* **Input Sent**:
  `None`
* **Status Code**: `200`
* **Response Time**: `2034.01 ms`
* **Response Received**:
  ```json
{
  "dataset_movies_count": 4803,
  "message": "Full-stack Recommendation service is running.",
  "model": "TF-IDF Content-Based Filtering",
  "status": "OK"
}
  ```

---

## Test Case: Movies Index
* **Endpoint URL**: `http://localhost:5000/movies`
* **HTTP Method**: `GET`
* **Input Sent**:
  `None`
* **Status Code**: `200`
* **Response Time**: `2455.73 ms`
* **Response Received**:
  ```json
"[List of 4803 movies; showing first 3: [{'id': 19995, 'title': 'Avatar'}, {'id': 285, 'title': \"Pirates of the Caribbean: At World's End\"}, {'id': 206647, 'title': 'Spectre'}]]"
  ```

---

## Test Case: Recommendation for 'Inception'
* **Endpoint URL**: `http://localhost:5000/recommend`
* **HTTP Method**: `POST`
* **Input Sent**:
  ```json
{
  "movie_title": "Inception"
}
  ```
* **Status Code**: `200`
* **Response Time**: `2061.40 ms`
* **Response Received**:
  ```json
{
  "matched_title": "Inception",
  "query_submitted": "Inception",
  "recommendations": [
    {
      "genres": "[{\"id\": 28, \"name\": \"Action\"}, {\"id\": 53, \"name\": \"Thriller\"}, {\"id\": 878, \"name\": \"Science Fiction\"}, {\"id\": 9648, \"name\": \"Mystery\"}]",
      "id": 180,
      "overview": "John Anderton is a top 'Precrime' cop in the late-21st century, when technology can predict crimes before they're committed. But Anderton becomes the quarry when another investigator targets him for a murder charge.",
      "title": "Minority Report",
      "vote_average": 7.1
    },
    {
      "genres": "[{\"id\": 28, \"name\": \"Action\"}, {\"id\": 35, \"name\": \"Comedy\"}, {\"id\": 878, \"name\": \"Science Fiction\"}]",
      "id": 43630,
      "overview": " ",
      "title": "The Helix... Loaded",
      "vote_average": 4.8
    },
    {
      "genres": "[{\"id\": 12, \"name\": \"Adventure\"}, {\"id\": 10751, \"name\": \"Family\"}, {\"id\": 14, \"name\": \"Fantasy\"}, {\"id\": 878, \"name\": \"Science Fiction\"}, {\"id\": 53, \"name\": \"Thriller\"}, {\"id\": 28, \"name\": \"Action\"}]",
      "id": 13836,
      "overview": "A taxi driver gets more than he bargained for when he picks up two teen runaways. Not only does the pair possess supernatural powers, but they're also trying desperately to escape people who have made them their targets.",
      "title": "Race to Witch Mountain",
      "vote_average": 5.5
    },
    {
      "genres": "[{\"id\": 53, \"name\": \"Thriller\"}, {\"id\": 878, \"name\": \"Science Fiction\"}, {\"id\": 9648, \"name\": \"Mystery\"}]",
      "id": 10133,
      "overview": "An unsuspecting, disenchanted man finds himself working as a spy in the dangerous, high-stakes world of corporate espionage. Quickly getting way over-his-head, he teams up with a mysterious femme fatale.",
      "title": "Cypher",
      "vote_average": 6.7
    },
    {
      "genres": "[{\"id\": 53, \"name\": \"Thriller\"}, {\"id\": 28, \"name\": \"Action\"}, {\"id\": 9648, \"name\": \"Mystery\"}]",
      "id": 59965,
      "overview": "A young man sets out to uncover the truth about his life after finding his baby photo on a missing persons website.",
      "title": "Abduction",
      "vote_average": 5.6
    },
    {
      "genres": "[{\"id\": 28, \"name\": \"Action\"}, {\"id\": 12, \"name\": \"Adventure\"}, {\"id\": 53, \"name\": \"Thriller\"}]",
      "id": 177677,
      "overview": "Ethan and team take on their most impossible mission yet, eradicating the Syndicate - an International rogue organization as highly skilled as they are, committed to destroying the IMF.",
      "title": "Mission: Impossible - Rogue Nation",
      "vote_average": 7.1
    },
    {
      "genres": "[{\"id\": 12, \"name\": \"Adventure\"}, {\"id\": 14, \"name\": \"Fantasy\"}, {\"id\": 28, \"name\": \"Action\"}, {\"id\": 53, \"name\": \"Thriller\"}, {\"id\": 878, \"name\": \"Science Fiction\"}]",
      "id": 18,
      "overview": "In 2257, a taxi driver is unintentionally given the task of saving a young girl who is part of the key that will ensure the survival of humanity.",
      "title": "The Fifth Element",
      "vote_average": 7.3
    },
    {
      "genres": "[{\"id\": 28, \"name\": \"Action\"}, {\"id\": 18, \"name\": \"Drama\"}, {\"id\": 36, \"name\": \"History\"}]",
      "id": 253450,
      "overview": "A female assassin during the Tang Dynasty who begins to question her loyalties when she falls in love with one of her targets.",
      "title": "The Assassin",
      "vote_average": 6.6
    },
    {
      "genres": "[{\"id\": 18, \"name\": \"Drama\"}, {\"id\": 28, \"name\": \"Action\"}, {\"id\": 53, \"name\": \"Thriller\"}, {\"id\": 80, \"name\": \"Crime\"}]",
      "id": 9869,
      "overview": "When CIA Analyst Jack Ryan interferes with an IRA assassination, a renegade faction targets Jack and his family as revenge.",
      "title": "Patriot Games",
      "vote_average": 6.3
    },
    {
      "genres": "[{\"id\": 27, \"name\": \"Horror\"}, {\"id\": 53, \"name\": \"Thriller\"}]",
      "id": 78383,
      "overview": "Abby Russell, a beautiful, dedicated nurse with a sinister side, has a secret life in which she targets and punishes dishonest men.",
      "title": "Nurse 3-D",
      "vote_average": 4.9
    }
  ],
  "recommendations_count": 10
}
  ```

---

## Test Case: Error Case: Movie Not Found
* **Endpoint URL**: `http://localhost:5000/recommend`
* **HTTP Method**: `POST`
* **Input Sent**:
  ```json
{
  "movie_title": "xyz123abc"
}
  ```
* **Status Code**: `404`
* **Response Time**: `2069.00 ms`
* **Response Received**:
  ```json
{
  "error": "Not Found",
  "message": "No matches found for 'xyz123abc'. Please try another search term."
}
  ```

---

## Test Case: Error Case: Empty Title
* **Endpoint URL**: `http://localhost:5000/recommend`
* **HTTP Method**: `POST`
* **Input Sent**:
  ```json
{
  "movie_title": ""
}
  ```
* **Status Code**: `400`
* **Response Time**: `2053.86 ms`
* **Response Received**:
  ```json
{
  "error": "Validation Error",
  "message": "Movie title cannot be empty."
}
  ```

---


import requests
import json
import time

def test_endpoint(name, method, url, payload=None):
    print(f"Testing {name}...")
    start_time = time.time()
    try:
        if method == "GET":
            res = requests.get(url)
        elif method == "POST":
            res = requests.post(url, json=payload)
        response_time_ms = (time.time() - start_time) * 1000
        
        # Parse JSON response
        try:
            response_json = res.json()
        except:
            response_json = res.text
            
        print(f"Status Code: {res.status_code}")
        print(f"Response Time: {response_time_ms:.2f} ms")
        print("---------------------------------------------")
        
        return {
            "name": name,
            "url": url,
            "method": method,
            "input": payload if payload else "N/A",
            "status_code": res.status_code,
            "response_time_ms": response_time_ms,
            "response": response_json
        }
    except Exception as e:
        print(f"Request failed: {e}")
        return {
            "name": name,
            "url": url,
            "method": method,
            "input": payload if payload else "N/A",
            "status_code": "ERROR",
            "response_time_ms": 0,
            "response": str(e)
        }

def main():
    base_url = "http://localhost:5000"
    results = []
    
    # 1. Health check (GET /health)
    results.append(test_endpoint("Health Check", "GET", f"{base_url}/health"))
    
    # 2. Autocomplete index check (GET /movies)
    # We will limit the response print so we don't spam
    movies_res = test_endpoint("Movies Index", "GET", f"{base_url}/movies")
    if movies_res["status_code"] == 200:
        movies_res["response"] = f"[List of {len(movies_res['response'])} movies; showing first 3: {movies_res['response'][:3]}]"
    results.append(movies_res)
    
    # 3. Successful Recommendation (POST /recommend - Inception)
    results.append(test_endpoint("Recommendation for 'Inception'", "POST", f"{base_url}/recommend", {"movie_title": "Inception"}))
    
    # 4. Error case - Movie not found (POST /recommend - xyz123abc)
    results.append(test_endpoint("Error Case: Movie Not Found", "POST", f"{base_url}/recommend", {"movie_title": "xyz123abc"}))
    
    # 5. Error case - Empty title input (POST /recommend - empty)
    results.append(test_endpoint("Error Case: Empty Title", "POST", f"{base_url}/recommend", {"movie_title": ""}))
    
    # Generate API_TESTING.md report
    markdown = "# API Testing Documentation (Postman-Equivalent Verification)\n\n"
    markdown += "This document verifies the API endpoints of the CineMatch AI backend server, reporting HTTP methods, input parameters, raw JSON responses, and response times in milliseconds.\n\n"
    
    for r in results:
        markdown += f"## Test Case: {r['name']}\n"
        markdown += f"* **Endpoint URL**: `{r['url']}`\n"
        markdown += f"* **HTTP Method**: `{r['method']}`\n"
        markdown += f"* **Input Sent**:\n"
        if r['input'] == "N/A":
            markdown += "  `None`\n"
        else:
            markdown += "  ```json\n" + json.dumps(r['input'], indent=2) + "\n  ```\n"
        markdown += f"* **Status Code**: `{r['status_code']}`\n"
        markdown += f"* **Response Time**: `{r['response_time_ms']:.2f} ms`\n"
        markdown += f"* **Response Received**:\n"
        markdown += "  ```json\n" + json.dumps(r['response'], indent=2) + "\n  ```\n"
        markdown += "\n---\n\n"
        
    out_path = "../API_TESTING.md"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(markdown)
    print(f"Generated API testing report at {out_path}")

if __name__ == "__main__":
    main()

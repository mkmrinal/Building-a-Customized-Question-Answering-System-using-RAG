import os
import json
import faiss
import numpy as np
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
import requests
from typing import List, Dict, Any
from pathlib import Path
from flask_cors import CORS

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)
CORS(app)  

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBGg1NhYaZ3uXk-b96cUq4WQW_LcLq5Hsk")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={GEMINI_API_KEY}"

def initialize_resources():
    """Initialize models and data with proper error handling"""
    try:
        # Construct absolute path to JSON file
        json_path = Path(r"C:\Users\mmukh\OneDrive\Desktop\Internship Project\App\major_cities.json")
        
        print(f"Looking for data file at: {json_path}")  # Debug output
        
        # Verify file exists
        if not json_path.exists():
            raise FileNotFoundError(f"City data file not found at: {json_path}")
        
        # Load city data
        with open(json_path, "r", encoding="utf-8") as f:
            city_data = json.load(f)
        
        # Initialize embedding model
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Prepare embeddings with proper type checking
        texts = []
        for city in city_data:
            # Safely handle all fields
            city_name = str(city.get('city', 'Unknown'))
            country = str(city.get('country', 'Unknown'))
            population = city.get('population', 0)
            altitude = city.get('altitude', 'N/A')
            landmarks = str(city.get('landmarks', ''))
            climate = str(city.get('climate', ''))
            
            text = f"""
            City: {city_name}
            Country: {country}
            Population: {population}
            Altitude: {altitude}
            Landmarks: {landmarks}
            Climate: {climate}
            """
            texts.append(text)
        
        embeddings = model.encode(texts).astype(np.float32)
        # Safely normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Prevent division by zero
        embeddings /= norms
        
        # Create FAISS index
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        
        return model, index, city_data
        
    except Exception as e:
        print(f"\nERROR DURING INITIALIZATION: {str(e)}\n")
        raise

try:
    model, index, city_data = initialize_resources()
    print("\nSuccessfully initialized all resources\n")
except Exception as e:
    print(f"\nFAILED TO INITIALIZE: {str(e)}\n")
    exit(1)

def generate_sub_questions(query: str) -> List[Dict[str, Any]]:
    """Generate sub-questions for complex queries"""
    prompt = f"""
    Analyze this question and generate 2-3 sub-questions needed to answer it.
    For each, specify:
    1. The sub-question text
    2. Required data attributes
    3. Retrieval method (vector/summary)
    4. Data source ('city_dataset' for our data)
    
    Use this exact JSON format:
    {{
        "question": "sub-question text",
        "attributes": ["list", "of", "attributes"],
        "retrieval": "vector|summary",
        "source": "city_dataset"
    }}

    Question: {query}

    Return only a valid JSON list of these objects.
    """
    
    data = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1}
    }
    
    try:
        response = requests.post(GEMINI_URL, json=data)
        if response.status_code == 200:
            return json.loads(response.json()["candidates"][0]["content"]["parts"][0]["text"])
    except:
        pass
    
    return [{
        "question": query,
        "attributes": ["population", "altitude", "landmarks", "climate"],
        "retrieval": "vector",
        "source": "city_dataset"
    }]

def vector_retrieval(query: str, attributes: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
    """Vector retrieval using FAISS index"""
    query_embedding = model.encode([query]).astype(np.float32)
    query_embedding /= np.linalg.norm(query_embedding)
    
    scores, indices = index.search(query_embedding, top_k * 3)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        city = city_data[idx]
        result = {"city": city["city"], "country": city["country"], "score": float(score)}
        for attr in attributes:
            if attr in city:
                result[attr] = city[attr]
        results.append(result)
    
    return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

def summary_retrieval(query: str) -> List[Dict[str, Any]]:
    """Summary retrieval returns full city records"""
    return city_data[:3]

def retrieve_city_data(sub_question: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Retrieve city data based on sub-question specification"""
    if sub_question["retrieval"] == "vector":
        return vector_retrieval(sub_question["question"], sub_question["attributes"])
    else:
        return summary_retrieval(sub_question["question"])

def format_sub_answer(city: Dict[str, Any], attribute: str) -> str:
    """Formats individual city data for sub-question responses"""
    if attribute == "population":
        if not isinstance(city.get('population'), (int, float)) or city['population'] <= 0:
            return None
        return f"{city['city']}: {round(city['population']/1000000, 1)}M"
    elif attribute == "altitude":
        if not city.get('altitude') or city['altitude'] == "N/A":
            return None
        return f"{city['city']}: {city['altitude']}m"
    elif attribute == "landmarks":
        if not city.get('landmarks'):
            return None
        return f"{city['city']}: {city['landmarks'].split('.')[0].strip()}"
    elif attribute == "climate":
        if not city.get('climate'):
            return None
        return f"{city['city']}: {city['climate'].split('.')[0].strip()}"
    return f"{city['city']}, {city['country']}"

def generate_final_response(query: str, sub_answers: List[Dict[str, Any]]) -> str:
    """
    Task 3: Response Aggregation
    Properly handles all question types and data fields from the city data
    """
    # Initialize default values
    final_answer = "Couldn't find specific information to answer this question."
    sub_responses = []
    
    # Extract country from query if specified
    query_lower = query.lower()
    target_country = None
    for country in ['india', 'usa', 'china', 'japan', 'uk', 'germany', 'france', 'australia', 'canada', 'brazil']:
        if country in query_lower:
            target_country = country
            break
    
    # Extract target city if specified
    target_city = None
    for city in [c['city'].lower() for c in city_data]:
        if city in query_lower:
            target_city = city
            break
    
    # Prepare data containers
    altitude_data = []
    population_data = []
    landmark_data = []
    climate_data = []
    
    # Collect all relevant city data
    for city in city_data:
        # Skip if country filter exists and doesn't match
        if target_country and city['country'].lower() != target_country:
            continue
        
        # Skip if city filter exists and doesn't match
        if target_city and city['city'].lower() != target_city:
            continue
        
        # Process altitude data
        if city.get('altitude') and city['altitude'] != 'N/A':
            try:
                altitude = float(city['altitude'].replace('m', '')) if 'm' in city['altitude'] else float(city['altitude'])
                altitude_data.append({
                    'city': city['city'],
                    'value': f"{altitude}m",
                    'numeric': altitude
                })
            except ValueError:
                pass
        
        # Process population data
        if city.get('population') and isinstance(city['population'], (int, float)):
            population_data.append({
                'city': city['city'],
                'value': f"{round(city['population']/1000000, 1)}M",
                'numeric': city['population']
            })
        
        # Process landmark data
        if city.get('landmarks') and city['landmarks'] not in ['N/A', '']:
            landmark_data.append({
                'city': city['city'],
                'value': city['landmarks'].split('.')[0].strip()
            })
        
        # Process climate data
        if city.get('climate') and city['climate'] not in ['N/A', '']:
            climate_data.append({
                'city': city['city'],
                'value': city['climate'].split('.')[0].strip()
            })

    # Determine question type and format response
    # Altitude questions
    if any(keyword in query_lower for keyword in ['altitude', 'elevation', 'height']):
        if altitude_data:
            altitude_data.sort(key=lambda x: x['numeric'], reverse='highest' in query_lower)
            top_entry = altitude_data[0]
            
            if 'highest' in query_lower:
                final_answer = f"{top_entry['city']} has the highest altitude in {target_country.title() if target_country else 'the world'} at {top_entry['value']}."
            elif 'lowest' in query_lower:
                final_answer = f"{top_entry['city']} has the lowest altitude in {target_country.title() if target_country else 'the world'} at {top_entry['value']}."
            else:
                final_answer = f"{top_entry['city']} has altitude {top_entry['value']}."
            
            sub_responses = [f"{x['city']}: {x['value']}" for x in altitude_data[:3]]
    
    # Population questions
    elif any(keyword in query_lower for keyword in ['population', 'populous']):
        if population_data:
            population_data.sort(key=lambda x: x['numeric'], reverse=True)
            top_entry = population_data[0]
            
            if 'highest' in query_lower:
                final_answer = f"{top_entry['city']} has the highest population in {target_country.title() if target_country else 'the world'} at {top_entry['value']}."
            elif 'lowest' in query_lower:
                final_answer = f"{top_entry['city']} has the lowest population in {target_country.title() if target_country else 'the world'} at {top_entry['value']}."
            else:
                final_answer = f"{top_entry['city']} has population {top_entry['value']}."
            
            sub_responses = [f"{x['city']}: {x['value']}" for x in population_data[:3]]
    
    # Landmark questions
    elif any(keyword in query_lower for keyword in ['landmark', 'place', 'attraction', 'tourist', 'visit', 'see']):
        if landmark_data:
            if target_city:
                landmarks = [x['value'] for x in landmark_data if x['city'].lower() == target_city]
                final_answer = f"Places to visit in {target_city.title()}: {', '.join(landmarks[:3])}."
            else:
                landmarks = [f"{x['city']}: {x['value']}" for x in landmark_data]
                final_answer = f"Top landmarks: {', '.join(landmarks[:3])}."
            
            sub_responses = [f"{x['city']}: {x['value']}" for x in landmark_data[:3]]
    
    # Climate questions
    elif any(keyword in query_lower for keyword in ['climate', 'weather', 'temperature']):
        if climate_data:
            if target_city:
                climates = [x['value'] for x in climate_data if x['city'].lower() == target_city]
                final_answer = f"Climate in {target_city.title()}: {', '.join(climates[:3])}."
            else:
                climates = [f"{x['city']}: {x['value']}" for x in climate_data]
                final_answer = f"Climate information: {', '.join(climates[:3])}."
            
            sub_responses = [f"{x['city']}: {x['value']}" for x in climate_data[:3]]
    
    # Format the output
    response_lines = ["○ Sub-question responses:"]
    for i, resp in enumerate(sub_responses[:3], 1):
        response_lines.append(f'■ "{resp}"')
    response_lines.append(f'○ Final Response: "{final_answer}"')
    
    return '\n'.join(response_lines)

def answer_city_question(query: str) -> str:
    """Complete RAG pipeline execution"""
    # Task 1: Generate sub-questions
    sub_questions = generate_sub_questions(query)
    
    # Task 2: Retrieve answers for each sub-question
    sub_answers = []
    for sub_q in sub_questions:
        cities = retrieve_city_data(sub_q)
        answers = []
        for city in cities:
            # Try all requested attributes until we get a valid answer
            for attr in sub_q["attributes"]:
                ans = format_sub_answer(city, attr)
                if ans:
                    answers.append(ans)
                    break
        
        sub_answers.append({
            "question": sub_q["question"],
            "answers": answers,
            "source": sub_q.get("source", "city_dataset")
        })
    
    # Task 3: Generate final response
    return generate_final_response(query, sub_answers)

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "Invalid request format"}), 400
            
        query = data['question'].strip()
        if not query:
            return jsonify({"error": "Question cannot be empty"}), 400
        
        print(f"Processing question: {query}")  
        response = answer_city_question(query)
        return jsonify({"response": response})
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
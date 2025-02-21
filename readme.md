### `README.md`

```markdown
# Doctor Matching API

This project is a FastAPI-based application that matches patients to doctors based on symptoms using a SNOMED-inspired model. It integrates a SQLite database to store doctor information and a symptom-to-specialty mapping, simulating SNOMED CT data. The API supports doctors with multiple specialties and provides a RESTful endpoint for symptom-based doctor recommendations.

## Features
- **Symptom-Based Matching**: Uses TF-IDF and cosine similarity to match patient symptoms to doctor specialties.
- **SQLite Integration**: Stores doctor data and SNOMED-like mappings in a lightweight database.
- **Multi-Specialty Support**: Doctors can have multiple specialties (stored as comma-separated strings).
- **FastAPI**: Modern, asynchronous API framework with automatic OpenAPI documentation.
- **Logging**: Includes basic logging for debugging and monitoring.

## Prerequisites
- Python 3.8+
- pip (Python package manager)

## Installation

1. **Clone the Repository** (if applicable):
   ```bash
   git clone https://github.com/balogun14/SymptomDocMatch.git
   cd SymptomDocMatch
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Save the Script**:
   Save the provided `app.py` script in your project directory.

## Setup

1. **Run the Application**:
   Execute the script to initialize the SQLite database (`doctor_matching.db`) and start the server:
   ```bash
   python app.py
   ```
   - This creates and seeds the database with:
     - A `doctors` table containing 95 doctor records.
     - A `snomed_mapping` table with simulated SNOMED data for 16 specialties.

2. **Verify the Server**:
   - The API will be available at `http://localhost:5000`.
   - Check the health endpoint: `http://localhost:5000/health` (returns `{"status": "healthy"}`).
   - Explore the interactive API docs: `http://localhost:5000/docs`.

## Usage

### API Endpoint
- **Endpoint**: `/match_doctors`
- **Method**: POST
- **Request Body**: JSON with a `symptoms` field (string).
- **Response**: JSON with a list of matched doctors, including their ID, name, specialties, matched specialty, and similarity score.

#### Example Request
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"symptoms": "stomach ache"}' \
     http://localhost:5000/match_doctors
```

#### Example Response
```json
{
    "status": "success",
    "matches": [
        {
            "id": "DOC67234",
            "name": "Dr. Robert Martinez",
            "specialties": ["Gastroenterology"],
            "matched_specialty": "Gastroenterology",
            "similarity_score": 0.707
        },
        {
            "id": "DOC12567",
            "name": "Dr. Elizabeth Scott",
            "specialties": ["Gastroenterology"],
            "matched_specialty": "Gastroenterology",
            "similarity_score": 0.707
        },
        {
            "id": "DOC78567",
            "name": "Dr. Lisa Martinez",
            "specialties": ["Gastroenterology"],
            "matched_specialty": "Gastroenterology",
            "similarity_score": 0.707
        }
    ]
}
```

### Error Responses
- **400 Bad Request**: If `symptoms` is missing or empty.
  ```json
  {"detail": "Symptoms must be a non-empty string"}
  ```
- **404 Not Found**: If no doctors match the symptoms.
  ```json
  {"detail": "No matching doctors found for the given symptoms"}
  ```
- **500 Internal Server Error**: For unexpected issues.
  ```json
  {"detail": "Internal server error: <error-message>"}
  ```

## Database Structure

### `doctors` Table
- `id` (TEXT PRIMARY KEY): Unique doctor identifier (e.g., "DOC78392").
- `name` (TEXT): Doctor’s full name (e.g., "Dr. Sarah Chen").
- `specialties` (TEXT): Comma-separated list of specialties (e.g., "Cardiology").

### `snomed_mapping` Table
- `specialty` (TEXT PRIMARY KEY): Medical specialty (e.g., "Gastroenterology").
- `snomed_symptoms` (TEXT): Space-separated symptoms with SNOMED-like annotations (e.g., "stomach ache abdominal pain nausea vomiting diarrhea bloating (SNOMED: 118434006, 422587007, 39621005)").

## Customization

### Adding Real SNOMED Data
The current `snomed_mapping` is a simulation. To integrate a real SNOMED CT API:
1. Replace `seed_snomed_mapping()` with an API call:
   ```python
   import requests
   def seed_snomed_mapping():
       response = requests.get("https://your-snomed-api.com/concepts", params={"term": "symptoms"})
       snomed_data = response.json()  # Parse and insert into SQLite
   ```
2. Update the table schema if needed (e.g., add `concept_id`).

### Updating Doctor Data
Modify `seed_doctors()` to load from your own dataset (e.g., CSV, JSON):
```python
import json
def seed_doctors():
    with open('doctors.json', 'r') as f:
        doctors = json.load(f)
    # Insert into SQLite as shown
```

### Multi-Specialty Support
The database supports multiple specialties (e.g., "Cardiology,Neurology"). Update the `doctors` list in `seed_doctors()` to include comma-separated specialties where applicable.

## Deployment
For production:
1. Use a WSGI server like Gunicorn with Uvicorn:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker doctor_matching:app
   ```
2. Containerize with Docker:
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY doctor_matching.py .
   CMD ["uvicorn", "doctor_matching:app", "--host", "0.0.0.0", "--port", "5000"]
   ```
3. Deploy to a cloud provider (e.g., AWS, Heroku).

## Limitations
- **SNOMED Simulation**: Uses a mock dataset instead of real SNOMED CT data.
- **Duplicate IDs**: Overwrites duplicate doctor IDs—modify `seed_doctors()` if merging specialties is preferred.
- **Scalability**: SQLite is lightweight; switch to PostgreSQL/MySQL for larger datasets.

## Contributing
Feel free to submit issues or pull requests to enhance functionality (e.g., real SNOMED integration, additional endpoints).

## License
This project is unlicensed—use at your own discretion.

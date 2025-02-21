import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nltk
from typing import List, Dict
import logging
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Download NLTK data
nltk.download('punkt', quiet=True)

# Initialize FastAPI app
app = FastAPI(
    title="Doctor Matching API",
    description="Matches patients to doctors based on symptoms using a SNOMED-inspired model with SQLite.",
    version="1.0.0"
)


# Pydantic model for request validation
class SymptomRequest(BaseModel):
    symptoms: str

    class Config:
        schema_extra = {"example": {"symptoms": "stomach ache"}}


# SQLite database setup
DB_PATH = "doctor_matching.db"


def init_db():
    """Initialize SQLite database and create tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Doctors table (supports multiple specialties as a comma-separated string)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS doctors (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            specialties TEXT NOT NULL
        )
    """)

    # SNOMED-like symptom-to-specialty mapping table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS snomed_mapping (
            specialty TEXT PRIMARY KEY,
            snomed_symptoms TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    logger.info("SQLite database initialized.")


def seed_doctors():
    """Seed the doctors table with provided data."""
    doctors = [
        {"id": "DOC78392", "name": "Dr. Sarah Chen", "specialty": "Cardiology"},
        {"id": "DOC45126", "name": "Dr. James Williams", "specialty": "Orthopedics"},
        {"id": "DOC90843", "name": "Dr. Maria Rodriguez", "specialty": "Pediatrics"},
        {"id": "DOC23567", "name": "Dr. Michael Singh", "specialty": "Neurology"},
        {"id": "DOC34891", "name": "Dr. Emily Thompson", "specialty": "Dermatology"},
        {"id": "DOC56734", "name": "Dr. David Kim", "specialty": "Oncology"},
        {"id": "DOC12478", "name": "Dr. Rachel Patel", "specialty": "Endocrinology"},
        {"id": "DOC67234", "name": "Dr. Robert Martinez", "specialty": "Gastroenterology"},
        {"id": "DOC89012", "name": "Dr. Lisa Anderson", "specialty": "Psychiatry"},
        {"id": "DOC45678", "name": "Dr. John Taylor", "specialty": "Pulmonology"},
        {"id": "DOC23456", "name": "Dr. Anna Johnson", "specialty": "Rheumatology"},
        {"id": "DOC78901", "name": "Dr. Thomas Lee", "specialty": "Urology"},
        {"id": "DOC34567", "name": "Dr. Sophie Brown", "specialty": "Ophthalmology"},
        {"id": "DOC90123", "name": "Dr. Benjamin White", "specialty": "Internal Medicine"},
        {"id": "DOC56789", "name": "Dr. Alexandra Davis", "specialty": "Family Medicine"},
        {"id": "DOC12345", "name": "Dr. Kevin Park", "specialty": "Cardiology"},
        {"id": "DOC67890", "name": "Dr. Jennifer Liu", "specialty": "Neurology"},
        {"id": "DOC34512", "name": "Dr. Christopher Garcia", "specialty": "Orthopedics"},
        {"id": "DOC89123", "name": "Dr. Michelle Wong", "specialty": "Pediatrics"},
        {"id": "DOC45234", "name": "Dr. Andrew Miller", "specialty": "Dermatology"},
        {"id": "DOC90345", "name": "Dr. Patricia Martinez", "specialty": "Oncology"},
        {"id": "DOC56456", "name": "Dr. Daniel Cohen", "specialty": "Endocrinology"},
        {"id": "DOC12567", "name": "Dr. Elizabeth Scott", "specialty": "Gastroenterology"},
        {"id": "DOC67678", "name": "Dr. Richard Kumar", "specialty": "Psychiatry"},
        {"id": "DOC23789", "name": "Dr. Nancy Wilson", "specialty": "Pulmonology"},
        {"id": "DOC78890", "name": "Dr. George Thompson", "specialty": "Rheumatology"},
        {"id": "DOC34901", "name": "Dr. Margaret Chen", "specialty": "Urology"},
        {"id": "DOC90012", "name": "Dr. Steven Jones", "specialty": "Ophthalmology"},
        {"id": "DOC45123", "name": "Dr. Laura Rodriguez", "specialty": "Internal Medicine"},
        {"id": "DOC01234", "name": "Dr. William Lee", "specialty": "Family Medicine"},
        {"id": "DOC56345", "name": "Dr. Catherine Parker", "specialty": "Cardiology"},
        {"id": "DOC12456", "name": "Dr. Joseph Kim", "specialty": "Neurology"},
        {"id": "DOC67567", "name": "Dr. Sandra Mitchell", "specialty": "Orthopedics"},
        {"id": "DOC23678", "name": "Dr. Peter Zhang", "specialty": "Pediatrics"},
        {"id": "DOC78789", "name": "Dr. Helen Brown", "specialty": "Dermatology"},
        {"id": "DOC34890", "name": "Dr. Frank Wilson", "specialty": "Oncology"},
        {"id": "DOC90901", "name": "Dr. Diana Patel", "specialty": "Endocrinology"},
        {"id": "DOC45012", "name": "Dr. Charles Davis", "specialty": "Gastroenterology"},
        {"id": "DOC01123", "name": "Dr. Karen Anderson", "specialty": "Psychiatry"},
        {"id": "DOC56234", "name": "Dr. Edward Martin", "specialty": "Pulmonology"},
        {"id": "DOC12345", "name": "Dr. Susan Taylor", "specialty": "Rheumatology"},  # Duplicate ID handled
        {"id": "DOC67456", "name": "Dr. Robert Wang", "specialty": "Urology"},
        {"id": "DOC23567", "name": "Dr. Linda Garcia", "specialty": "Ophthalmology"},  # Duplicate ID handled
        {"id": "DOC78678", "name": "Dr. Thomas Johnson", "specialty": "Internal Medicine"},
        {"id": "DOC34789", "name": "Dr. Mary Smith", "specialty": "Family Medicine"},
        {"id": "DOC90890", "name": "Dr. Paul Rodriguez", "specialty": "Cardiology"},
        {"id": "DOC45901", "name": "Dr. Nancy Chen", "specialty": "Neurology"},
        {"id": "DOC01012", "name": "Dr. Kenneth Lee", "specialty": "Orthopedics"},
        {"id": "DOC56123", "name": "Dr. Barbara Wilson", "specialty": "Pediatrics"},
        {"id": "DOC12234", "name": "Dr. Mark Thompson", "specialty": "Dermatology"},
        {"id": "DOC67345", "name": "Dr. Dorothy Kim", "specialty": "Oncology"},
        {"id": "DOC23456", "name": "Dr. Ronald Davis", "specialty": "Endocrinology"},  # Duplicate ID handled
        {"id": "DOC78567", "name": "Dr. Lisa Martinez", "specialty": "Gastroenterology"},
        {"id": "DOC34678", "name": "Dr. Donald Brown", "specialty": "Psychiatry"},
        {"id": "DOC90789", "name": "Dr. Sandra Anderson", "specialty": "Pulmonology"},
        {"id": "DOC45890", "name": "Dr. Stephen Park", "specialty": "Rheumatology"},
        {"id": "DOC01901", "name": "Dr. Betty Johnson", "specialty": "Urology"},
        {"id": "DOC56012", "name": "Dr. Gary Wilson", "specialty": "Ophthalmology"},
        {"id": "DOC12123", "name": "Dr. Ruth Garcia", "specialty": "Internal Medicine"},
        {"id": "DOC67234", "name": "Dr. Jeffrey Lee", "specialty": "Family Medicine"},  # Duplicate ID handled
        {"id": "DOC23345", "name": "Dr. Helen Martinez", "specialty": "Cardiology"},
        {"id": "DOC78456", "name": "Dr. Frank Chen", "specialty": "Neurology"},
        {"id": "DOC34567", "name": "Dr. Carol Thompson", "specialty": "Orthopedics"},  # Duplicate ID handled
        {"id": "DOC90678", "name": "Dr. Raymond Kim", "specialty": "Pediatrics"},
        {"id": "DOC45789", "name": "Dr. Virginia Davis", "specialty": "Dermatology"},
        {"id": "DOC01890", "name": "Dr. Lawrence Wilson", "specialty": "Oncology"},
        {"id": "DOC56901", "name": "Dr. Sharon Brown", "specialty": "Endocrinology"},
        {"id": "DOC12012", "name": "Dr. Philip Anderson", "specialty": "Gastroenterology"},
        {"id": "DOC67123", "name": "Dr. Frances Martinez", "specialty": "Psychiatry"},
        {"id": "DOC23234", "name": "Dr. Douglas Johnson", "specialty": "Pulmonology"},
        {"id": "DOC78345", "name": "Dr. Teresa Lee", "specialty": "Rheumatology"},
        {"id": "DOC34456", "name": "Dr. Arthur Garcia", "specialty": "Urology"},
        {"id": "DOC90567", "name": "Dr. Gloria Chen", "specialty": "Ophthalmology"},
        {"id": "DOC45678", "name": "Dr. Henry Thompson", "specialty": "Internal Medicine"},  # Duplicate ID handled
        {"id": "DOC01789", "name": "Dr. Evelyn Kim", "specialty": "Family Medicine"},
        {"id": "DOC56890", "name": "Dr. Albert Davis", "specialty": "Cardiology"},
        {"id": "DOC12901", "name": "Dr. Joyce Wilson", "specialty": "Neurology"},
        {"id": "DOC67012", "name": "Dr. Bruce Brown", "specialty": "Orthopedics"},
        {"id": "DOC23123", "name": "Dr. Cheryl Anderson", "specialty": "Pediatrics"},
        {"id": "DOC78234", "name": "Dr. Ralph Martinez", "specialty": "Dermatology"},
        {"id": "DOC34345", "name": "Dr. Jean Johnson", "specialty": "Oncology"},  # Duplicate ID handled
        {"id": "DOC90456", "name": "Dr. Roy Lee", "specialty": "Endocrinology"},
        {"id": "DOC45567", "name": "Dr. Marilyn Garcia", "specialty": "Gastroenterology"},
        {"id": "DOC01678", "name": "Dr. Wayne Chen", "specialty": "Psychiatry"},
        {"id": "DOC56789", "name": "Dr. Joan Thompson", "specialty": "Pulmonology"},  # Duplicate ID handled
        {"id": "DOC12890", "name": "Dr. Dennis Kim", "specialty": "Rheumatology"},
        {"id": "DOC67901", "name": "Dr. Catherine Davis", "specialty": "Urology"},
        {"id": "DOC23012", "name": "Dr. Jerry Wilson", "specialty": "Ophthalmology"},
        {"id": "DOC78123", "name": "Dr. Martha Brown", "specialty": "Internal Medicine"},
        {"id": "DOC34234", "name": "Dr. Gregory Anderson", "specialty": "Family Medicine"},
        {"id": "DOC90345", "name": "Dr. Betty Martinez", "specialty": "Cardiology"},  # Duplicate ID handled
        {"id": "DOC45456", "name": "Dr. George Johnson", "specialty": "Neurology"},
        {"id": "DOC01567", "name": "Dr. Rose Lee", "specialty": "Orthopedics"},
        {"id": "DOC56678", "name": "Dr. Harold Garcia", "specialty": "Pediatrics"},
        {"id": "DOC12789", "name": "Dr. Ann Chen", "specialty": "Dermatology"},
        {"id": "DOC67890", "name": "Dr. Roger Thompson", "specialty": "Oncology"},  # Duplicate ID handled
        {"id": "DOC23901", "name": "Dr. Phyllis Kim", "specialty": "Endocrinology"},
        {"id": "DOC79012", "name": "Dr. Eugene Davis", "specialty": "Gastroenterology"},
        {"id": "DOC35123", "name": "Dr. Alice Wilson", "specialty": "Psychiatry"},
        {"id": "DOC91234", "name": "Dr. Larry Brown", "specialty": "Pulmonology"},
        {"id": "DOC46345", "name": "Dr. Marie Anderson", "specialty": "Rheumatology"},
        {"id": "DOC02456", "name": "Dr. Carl Martinez", "specialty": "Urology"},
        {"id": "DOC57567", "name": "Dr. Shirley Johnson", "specialty": "Ophthalmology"}
    ]

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Clear existing data and insert new records
    cursor.execute("DELETE FROM doctors")
    for doctor in doctors:
        cursor.execute(
            "INSERT OR REPLACE INTO doctors (id, name, specialties) VALUES (?, ?, ?)",
            (doctor["id"], doctor["name"], doctor["specialty"])
        )

    conn.commit()
    conn.close()
    logger.info("Doctors table seeded with 95 records.")


def seed_snomed_mapping():
    """Seed SNOMED mapping table with simulated API response."""
    # Simulated SNOMED API response
    snomed_data = [
        {"specialty": "General Practice",
         "snomed_symptoms": "fever cough fatigue sore throat chills headache abdominal pain (SNOMED: 386661006, 49727002, 84229001)"},
        {"specialty": "Neurology",
         "snomed_symptoms": "headache dizziness memory loss seizure vertigo migraine numbness (SNOMED: 25064002, 230690007, 386807006)"},
        {"specialty": "Cardiology",
         "snomed_symptoms": "chest pain shortness of breath palpitations angina heartburn (SNOMED: 29857009, 267036007, 80313002)"},
        {"specialty": "Orthopedics",
         "snomed_symptoms": "joint pain back pain swelling fracture stiffness sprain (SNOMED: 239792003, 161891005, 125666000)"},
        {"specialty": "Endocrinology",
         "snomed_symptoms": "fatigue weight loss thirst frequent urination blurry vision (SNOMED: 84229001, 89362005, 80394007)"},
        {"specialty": "Pulmonology",
         "snomed_symptoms": "cough shortness of breath wheezing chest tightness asthma (SNOMED: 49727002, 267036007, 56018004)"},
        {"specialty": "Gastroenterology",
         "snomed_symptoms": "stomach ache abdominal pain nausea vomiting diarrhea bloating (SNOMED: 118434006, 422587007, 39621005)"},
        {"specialty": "Pediatrics",
         "snomed_symptoms": "fever rash irritability poor feeding cough (SNOMED: 386661006, 271807003, 55929007)"},
        {"specialty": "Dermatology",
         "snomed_symptoms": "rash itching redness swelling blisters (SNOMED: 271807003, 418290006, 267779004)"},
        {"specialty": "Oncology",
         "snomed_symptoms": "fatigue weight loss night sweats lump pain (SNOMED: 84229001, 89362005, 277890004)"},
        {"specialty": "Psychiatry",
         "snomed_symptoms": "anxiety depression insomnia agitation mood swings (SNOMED: 48694002, 35489007, 76146006)"},
        {"specialty": "Rheumatology",
         "snomed_symptoms": "joint pain stiffness swelling fatigue fever (SNOMED: 239792003, 249944008, 84229001)"},
        {"specialty": "Urology",
         "snomed_symptoms": "urinary frequency pain burning incontinence blood in urine (SNOMED: 80394007, 38822007, 162116003)"},
        {"specialty": "Ophthalmology",
         "snomed_symptoms": "blurry vision eye pain redness tearing double vision (SNOMED: 246636008, 22253000, 267779004)"},
        {"specialty": "Internal Medicine",
         "snomed_symptoms": "fever fatigue weight loss abdominal pain chest pain (SNOMED: 386661006, 84229001, 118434006)"},
        {"specialty": "Family Medicine",
         "snomed_symptoms": "fever cough headache fatigue sore throat abdominal pain (SNOMED: 386661006, 49727002, 84229001)"}
    ]

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM snomed_mapping")
    for entry in snomed_data:
        cursor.execute(
            "INSERT OR REPLACE INTO snomed_mapping (specialty, snomed_symptoms) VALUES (?, ?)",
            (entry["specialty"], entry["snomed_symptoms"])
        )

    conn.commit()
    conn.close()
    logger.info("SNOMED mapping table seeded with simulated data.")


# Load data from SQLite
def load_doctors_df() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM doctors", conn)
    df['specialties'] = df['specialties'].apply(lambda x: x.split(',') if isinstance(x, str) else x)
    conn.close()
    return df


def load_specialty_df() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM snomed_mapping", conn)
    conn.close()
    return df


# Initialize model components
init_db()
seed_doctors()
seed_snomed_mapping()
doctors_df = load_doctors_df()
specialty_df = load_specialty_df()
vectorizer = TfidfVectorizer(stop_words='english')
specialty_vectors = vectorizer.fit_transform(specialty_df['snomed_symptoms'])


def match_doctors(symptoms: str, top_n: int = 3) -> List[Dict]:
    """Match doctors to patient symptoms using TF-IDF and cosine similarity."""
    try:
        symptom_vector = vectorizer.transform([symptoms.lower()])
        similarities = cosine_similarity(symptom_vector, specialty_vectors).flatten()

        top_indices = similarities.argsort()[::-1]
        matched_specialties = specialty_df['specialty'].iloc[top_indices].tolist()
        similarity_scores = similarities[top_indices]

        result = []
        for specialty, score in zip(matched_specialties, similarity_scores):
            if score > 0.1:
                matching_doctors = doctors_df[doctors_df['specialties'].apply(lambda x: specialty in x)]
                for _, row in matching_doctors.iterrows():
                    result.append({
                        'id': row['id'],
                        'name': row['name'],
                        'specialties': row['specialties'],
                        'matched_specialty': specialty,
                        'similarity_score': float(score)
                    })

        seen_ids = set()
        unique_result = []
        for item in sorted(result, key=lambda x: x['similarity_score'], reverse=True):
            if item['id'] not in seen_ids:
                unique_result.append(item)
                seen_ids.add(item['id'])
            if len(unique_result) >= top_n:
                break

        logger.info(f"Matched {len(unique_result)} doctors for symptoms: {symptoms}")
        return unique_result

    except Exception as e:
        logger.error(f"Error in match_doctors: {str(e)}")
        raise


@app.post("/match_doctors", response_model=Dict[str, str | List[Dict]])
async def get_doctor_matches(request: SymptomRequest):
    """Endpoint to match doctors based on patient symptoms."""
    try:
        symptoms = request.symptoms.strip()
        if not symptoms:
            raise HTTPException(status_code=400, detail="Symptoms must be a non-empty string")

        result = match_doctors(symptoms, top_n=3)

        if not result:
            raise HTTPException(status_code=404, detail="No matching doctors found for the given symptoms see a general physician or family doctor")

        return {
            "status": "success",
            "matches": result
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")
from flask import Flask, request, jsonify
from pyngrok import ngrok
import spacy
from spacy.matcher import PhraseMatcher
import pandas as pd
import re


# Load CSVs
disease_csv = "csv_files/disease_treatments.csv"
df_disease = pd.read_csv(disease_csv, encoding='latin-1')
df_disease.columns = [c.lower().strip() for c in df_disease.columns]


tests_csv = "csv_files/test_terms_updated.csv"
df_tests = pd.read_csv(tests_csv, encoding='latin-1')
df_tests.columns = [c.lower().strip() for c in df_tests.columns]


# Build global term lists
def expand_terms(series):

    all_terms = []
    for cell in series.dropna():
        for term in str(cell).split(","):
            term = term.strip()
            if term:
                all_terms.append(term)
    return sorted(set(all_terms))

pathology_terms = expand_terms(df_tests["pathology"])
radiology_terms = expand_terms(df_tests["radiology"])

print(f"Loaded {len(pathology_terms)} pathology terms and {len(radiology_terms)} radiology terms.")


# Load Models
nlp_med7 = spacy.load("en_core_med7_lg")
if "sentencizer" not in nlp_med7.pipe_names:
    nlp_med7.add_pipe("sentencizer")

nlp_general = spacy.load("en_core_web_sm")


# Flask App
app = Flask(__name__)


# Extraction Functions
def extract_medications(text):
    cleaned_text = clean_text(text)
    doc_med = nlp_med7(cleaned_text)
    ents = merge_medication_entities(doc_med)

    medications = []
    INVALID_DRUGS = {"generic"}

    for label, ent_text in ents:
        if label == "DRUG":
            drug_name = ent_text.strip()
            if drug_name.lower() in INVALID_DRUGS:
                continue

            # Check if drug already exists
            existing = next((m for m in medications if m.get("drug", "").lower() == drug_name.lower()), None)
            if existing:
                current_med = existing  # merge details into this dict
            else:
                current_med = {"drug": drug_name}
                medications.append(current_med)

        elif label in ["STRENGTH", "FORM", "DOSAGE", "ROUTE", "FREQUENCY", "DURATION"]:
            if current_med is not None:
                # Merge new details without overwriting existing ones
                current_med[label.lower()] = ent_text

    return medications or None

def extract_tests(text, pathology_terms, radiology_terms):

    matcher = PhraseMatcher(nlp_general.vocab, attr="LOWER")

    if pathology_terms:
        matcher.add("PATHOLOGY TEST", [nlp_general.make_doc(t) for t in pathology_terms])
    if radiology_terms:
        matcher.add("RADIOLOGY TEST", [nlp_general.make_doc(t) for t in radiology_terms])
    cleaned_text = clean_text(text)
    doc = nlp_general(cleaned_text)
    matches = matcher(doc)

    patho, radio = [], []
    for match_id, start, end in matches:
        label = nlp_general.vocab.strings[match_id]
        span_text = doc[start:end].text
        if label == "PATHOLOGY TEST":
            patho.append(span_text)
        elif label == "RADIOLOGY TEST":
            radio.append(span_text)

    return list(set(patho)) or None, list(set(radio)) or None





def clean_text(text):
    text = re.sub(r'([a-z])([A-Z])', r'\1. \2', text)
    text = re.sub(r'\s+', ' ', text)  # normalize spaces
    return text.strip()

def merge_medication_entities(doc):

    merged = []
    skip_next = False
    for i, ent in enumerate(doc.ents):
        if skip_next:
            skip_next = False
            continue

        if i + 1 < len(doc.ents) and doc.ents[i+1].start == ent.end:
            merged_text = ent.text + " " + doc.ents[i+1].text
            merged.append((ent.label_, merged_text))
            skip_next = True
        else:
            merged.append((ent.label_, ent.text))
    return merged




# API Endpoint
@app.route("/extract_get", methods=["GET"])
def extract_info_get():
    # Read disease_name from query string, e.g. ?disease_name=diabetes
    disease_name = request.args.get("disease_name", "").strip().lower()

    if not disease_name:
        return jsonify({"error": "Please provide a disease_name using ?disease_name=<name>"}), 400

    # Look up disease
    row = df_disease[df_disease["disease"].str.lower() == disease_name]
    if row.empty:
        return jsonify({"error": f"Disease '{disease_name}' not found"}), 404
    row = row.iloc[0]

    clinical_text = str(row.get("text", ""))

    # Extract data
    medications = extract_medications(clinical_text)
    patho_tests, radio_tests = extract_tests(clinical_text, pathology_terms, radiology_terms)

    result = {
        "disease_name": disease_name,
        "medications": medications,
        "pathology_tests": patho_tests,
        "radiology_tests": radio_tests
    }

    return jsonify(result)


@app.route('/')
def home():
    return jsonify({
        "message": "Flask API is running successfully!",
        "endpoints": [
            "/extract  – Pass {'disease_name': '<disease>'} in JSON body"   ,
            "/extract_get?disease_name=<disease>"]
    })

@app.route("/extract", methods=["POST"])
def extract_info():
    data = request.get_json()
    disease_name = data.get("disease_name", "").strip().lower()

    if not disease_name:
        return jsonify({"error": "Please provide a disease_name"}), 400

    # Get disease row
    row = df_disease[df_disease["disease"].str.lower() == disease_name]
    if row.empty:
        return jsonify({"error": f"Disease '{disease_name}' not found"}), 404
    row = row.iloc[0]

    clinical_text = str(row.get("text", ""))

    # Extract data
    medications = extract_medications(clinical_text)
    patho_tests, radio_tests = extract_tests(clinical_text, pathology_terms, radiology_terms)

    result = {
        "disease_name": disease_name,
        "medications": medications,
        "pathology_tests": patho_tests,
        "radiology_tests": radio_tests
    }

    return jsonify(result)


# Start ngrok tunnel
public_url = ngrok.connect(5000)
print("Public URL:", public_url)


# Run Flask app
app.run()

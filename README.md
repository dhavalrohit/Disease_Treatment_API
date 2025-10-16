# Treatment Information Extraction API

This API extracts medications, pathology tests, and radiology tests for a given disease based on clinical text data stored in CSV files. It uses spaCy, Med7, and the spaCy PhraseMatcher API for natural language processing and text matching.

## Installation & Setup Instructions

1. Python Version: 3.10.0
2. Open the terminal and run the following commands:

pip install flask pyngrok pandas flask-ngrok 

pip install spacy==3.8.7 numpy==2.2.6

pip install "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl#sha256=1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85"

pip install "en-core-med7-lg @ https://huggingface.co/kormilitzin/en_core_med7_lg/resolve/main/en_core_med7_lg-any-py3-none-any.whl"

pip install --upgrade h5py

## Ngrok Authentication Key

Run the following command in the terminal to save your Ngrok AuthToken configuration:

ngrok authtoken 3406Re0LzMooLpJNKCl4fEUgXg5_4xtS28FCsAr4GB7zvkUn6

## Running the Application

Run the Flask application using:

python app.py

After running, open the generated Ngrok URL (for example):
https://robustiously-untraveled-anahi.ngrok-free.dev/

## API Endpoints

### POST /extract

URL:
https://robustiously-untraveled-anahi.ngrok-free.dev/extract

Body (raw JSON):
{
  "disease_name": "actual_disease_name"
}

Example:
{
  "disease_name": "Dengue"
}

Example Response:
{
  "disease_name": "dengue",
  "medications": [
    {
      "dosage": "one tablet",
      "drug": "Paracetamol",
      "duration": "for 5 days",
      "strength": "500 mg"
    }
  ],
  "pathology_tests": [
    "Platelet Count",
    "Dengue NS1 Antigen"
  ],
  "radiology_tests": [
    "Ultrasound Abdomen"
  ]
}

###  GET /extract_get

URL Format:
https://robustiously-untraveled-anahi.ngrok-free.dev/extract_get?disease_name=actual_disease_name

Example:
https://robustiously-untraveled-anahi.ngrok-free.dev/extract_get?disease_name=high fever

Example Response:
{
  "disease_name": "high fever",
  "medications": [
    {
      "dosage": "one tablet",
      "drug": "Paracetamol",
      "duration": "for 3 days",
      "strength": "500 mg"
    }
  ],
  "pathology_tests": [
    "Blood Culture"
  ],
  "radiology_tests": [
    "Chest X-Ray"
  ]
}

## Troubleshooting

If you encounter the following error:

import NumpyOps
File "thinc\\backends\\numpy_ops.pyx", line 1, in init thinc.backends.numpy_ops
ValueError: numpy.dtype size changed, may indicate binary incompatibility.
Expected 96 from C header, got 88 from PyObject

Run the following commands to fix it:

pip uninstall numpy thinc spacy -y

pip install numpy==2.2.6

pip install spacy==3.8.7


## Setup Instructions for Running in Virtual Environement/VS Code
Download Zip File From Github

Extract and open folder in VS Code

Select Python Version

Select Create Virtual Environement

Select .venv

Installed mentioned dependencies 

7.run app.py

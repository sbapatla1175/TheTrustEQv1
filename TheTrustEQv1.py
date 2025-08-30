import os
import sys
import json
import logging
import re
from dotenv import load_dotenv
import openai
import pdfplumber
from docx import Document

# --- Directory Setup ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
LOGS_DIR = os.path.join(BASE_DIR, 'Logs')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'Templates')
INPUT_DIR = os.path.join(BASE_DIR, 'input')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Logging Setup ---
logging.basicConfig(
    filename=os.path.join(LOGS_DIR, 'app.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info('--- TheTrustEQ script started ---')

# --- Load Environment Variables ---
load_dotenv('TheTrustEQ.env')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
OPENAI_API_TYPE = os.getenv('OPENAI_API_TYPE')
OPENAI_API_VERSION = os.getenv('OPENAI_API_VERSION')
OPENAI_DEPLOYMENT_COMPLETION = os.getenv('OPENAI_DEPLOYMENT_COMPLETION')

if not OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
    raise ValueError('Azure OpenAI API key or endpoint not set')

openai.api_key = OPENAI_API_KEY
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_type = OPENAI_API_TYPE
openai.api_version = OPENAI_API_VERSION

# -- Utility Functions --

def load_template(json_filename):
    with open(os.path.join(TEMPLATES_DIR, json_filename), 'r', encoding='utf-8') as f:
        return json.load(f)

def read_transcript_file(filename):
    filepath = os.path.join(INPUT_DIR, filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.txt':
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    elif ext == '.pdf':
        with pdfplumber.open(filepath) as pdf:
            return '\n\n'.join([page.extract_text() or '' for page in pdf.pages])
    elif ext in ('.docx', '.doc'):
        doc = Document(filepath)
        return '\n\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
    else:
        raise ValueError("Unsupported file type. Please provide a .txt, .pdf, or .docx file.")

def generate_completion(prompt):
    response = openai.chat.completions.create(
        model=OPENAI_DEPLOYMENT_COMPLETION,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2500,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content.strip()

def extract_json_block(text):
    start, curly = text.find('{'), 0
    if start == -1: return None
    for i in range(start, len(text)):
        if text[i] == '{': curly += 1
        elif text[i] == '}':
            curly -= 1
            if curly == 0: return text[start:i+1]
    return None

def safe_json_load(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        fixed = re.sub(r'(?<!\\)\n', '\\\\n', text)  # replace unescaped newlines in strings
        return json.loads(fixed)

def process_with_prompt(prompt_json_name, transcript_text):
    prompt = load_template(prompt_json_name)['instruction'] + "\n\n" + transcript_text
    result = generate_completion(prompt)
    json_block = extract_json_block(result)
    if not json_block:
        logging.error(f"No JSON block detected in LLM output:\n{result}")
        print("Error: No JSON detected in LLM output. See logs for details.")
        return None
    try:
        return safe_json_load(json_block)
    except Exception as e:
        logging.error(f"Failed parsing LLM JSON for {prompt_json_name}: {e}")
        print("=== RAW LLM OUTPUT ===")
        print(json_block)
        print("=== END RAW OUTPUT ===")
        print(f"Error parsing LLM JSON: {e}")
        return None

def process_transcript(transcript_filename):
    transcript_text = read_transcript_file(transcript_filename)
    base = os.path.splitext(transcript_filename)[0]

    # --- Credibility
    credibility = process_with_prompt('CredibilityBDDPrompt.json', transcript_text)
    if credibility:
        factors_file = os.path.join(OUTPUT_DIR, f"{base}_Credibility_Factors.json")
        with open(factors_file, 'w', encoding='utf-8') as f: json.dump(credibility, f, indent=2, ensure_ascii=False)
        summary_file = os.path.join(OUTPUT_DIR, f"{base}_Credibility_Summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f: f.write(credibility.get('Credibility', '').strip()+"\n")
        print(f"Credibility factors saved to: {factors_file}")
        print(f"Credibility summary saved to: {summary_file}")

    # --- First Impressions
    fi = process_with_prompt('FirstImpressions.json', transcript_text)
    if fi:
        summary_file = os.path.join(OUTPUT_DIR, f"{base}_FirstImpressions.txt")
        with open(summary_file, 'w', encoding='utf-8') as f: f.write(fi.get('First Impression', '').strip()+"\n")
        print(f"First Impression summary saved to: {summary_file}")

# --- Main Entry Point ---
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python TheTrustEQ.py <TranscriptFilename>')
        sys.exit(1)
    process_transcript(sys.argv[1])
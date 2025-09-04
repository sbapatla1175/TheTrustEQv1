import os
import sys
import json
import logging
import re
import time
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
load_dotenv('TheTrustEQRevised.env')

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
    attempts = 5
    backoff = 2
    last_err = None
    for i in range(1, attempts + 1):
        try:
            response = openai.chat.completions.create(
                model=OPENAI_DEPLOYMENT_COMPLETION,  # Azure DEPLOYMENT NAME
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
                temperature=0,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                timeout=30
            )

            # --- NEW: log rate-limit headers if present ---
            if hasattr(response, 'response_ms'):
                # SDK v1.x returns headers in .response
                headers = getattr(response, 'headers', None)
                if headers:
                    print("Rate limit headers:", headers)
                    logging.info(f"Rate limit headers: {headers}")

            return response.choices[0].message.content.strip()

        except openai.RateLimitError as e:
            # Try to pull headers from the error response if available
            if hasattr(e, 'response') and e.response is not None:
                hdrs = getattr(e.response, 'headers', {})
                print("429 headers:", hdrs)
                logging.warning(f"429 headers: {hdrs}")

            wait = min(60, backoff)
            logging.warning(f"429 rate-limited (attempt {i}/{attempts}). retrying in ~{wait}s. details={e}")
            time.sleep(wait + (0.5 * i))
            backoff *= 2
            last_err = e
        except openai.APIConnectionError as e:
            logging.error(f"Network/APIConnectionError: {e}")
            last_err = e
            time.sleep(3)
        except openai.APIError as e:
            # 4xx/5xx non-429; log & stop
            if hasattr(e, 'response') and e.response is not None:
                hdrs = getattr(e.response, 'headers', {})
                print("APIError headers:", hdrs)
                logging.error(f"APIError headers: {hdrs}")
            logging.error(f"APIError: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected exception: {e}")
            raise

    raise RuntimeError(f"Max attempts reached without success: {last_err}")



def extract_json_block(text):
    # Find first '{' or '['
    i_obj = text.find('{')
    i_arr = text.find('[')
    if i_obj == -1 and i_arr == -1:
        return None
    start = min([i for i in [i_obj, i_arr] if i != -1])

    # Choose matching bracket
    open_ch = text[start]
    close_ch = '}' if open_ch == '{' else ']'
    depth = 0

    for i in range(start, len(text)):
        ch = text[i]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start:i+1]
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
    credibility = process_with_prompt('CredibilityBDDPromptTest.json', transcript_text)
    if credibility:
        factors_file = os.path.join(OUTPUT_DIR, f"{base}_Credibility_Factors.json")
        with open(factors_file, 'w', encoding='utf-8') as f: json.dump(credibility, f, indent=2, ensure_ascii=False)
        summary_file = os.path.join(OUTPUT_DIR, f"{base}_Credibility_Summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f: f.write(credibility.get('Credibility', '').strip()+"\n")
        print(f"Credibility factors saved to: {factors_file}")
        print(f"Credibility summary saved to: {summary_file}")
    
    # # --- Reliability
    # reliability = process_with_prompt('ReliabilityBDDPrompt.json', transcript_text)
    # if reliability:
    #     factors_file = os.path.join(OUTPUT_DIR, f"{base}_Reliability_Factors.json")
    #     with open(factors_file, 'w', encoding='utf-8') as f: json.dump(reliability, f, indent=2, ensure_ascii=False)
    #     summary_file = os.path.join(OUTPUT_DIR, f"{base}_Reliability_Summary.txt")
    #     with open(summary_file, 'w', encoding='utf-8') as f: f.write(reliability.get('Reliability', '').strip()+"\n")
    #     print(f"Reliability factors saved to: {factors_file}")
    #     print(f"Reliability summary saved to: {summary_file}")

    # # --- Intimacy
    # intimacy = process_with_prompt('IntimacyBDDPrompt.json', transcript_text)
    # if intimacy:
    #     factors_file = os.path.join(OUTPUT_DIR, f"{base}_Intimacy_Factors.json")
    #     with open(factors_file, 'w', encoding='utf-8') as f: json.dump(intimacy, f, indent=2, ensure_ascii=False)
    #     summary_file = os.path.join(OUTPUT_DIR, f"{base}_Intimacy_Summary.txt")
    #     with open(summary_file, 'w', encoding='utf-8') as f: f.write(intimacy.get('Intimacy', '').strip()+"\n")
    #     print(f"Intimacy factors saved to: {factors_file}")
    #     print(f"Intimacy summary saved to: {summary_file}")

    # # --- Self Orientation
    # self_orientation = process_with_prompt('SelfOrientationBDDPrompt.json', transcript_text)
    # if self_orientation:
    #     factors_file = os.path.join(OUTPUT_DIR, f"{base}_SelfOrientation_Factors.json")
    #     with open(factors_file, 'w', encoding='utf-8') as f: json.dump(self_orientation, f, indent=2, ensure_ascii=False)
    #     summary_file = os.path.join(OUTPUT_DIR, f"{base}_SelfOrientation_Summary.txt")
    #     with open(summary_file, 'w', encoding='utf-8') as f: f.write(self_orientation.get('Self Orientation', '').strip()+"\n")
    #     print(f"Self Orientation factors saved to: {factors_file}")
    #     print(f"Self Orientation summary saved to: {summary_file}")

    # # --- First Impressions
    # fi = process_with_prompt('FirstImpressions.json', transcript_text)
    # if fi:
    #     summary_file = os.path.join(OUTPUT_DIR, f"{base}_FirstImpressions.txt")
    #     with open(summary_file, 'w', encoding='utf-8') as f: f.write(fi.get('First Impression', '').strip()+"\n")
    #     print(f"First Impression summary saved to: {summary_file}")

    # # --- Q&A
    # qa = process_with_prompt('QnABDDPrompt.json', transcript_text)
    # if qa:
    #     lines = []
    #     expl = qa.get('explicit_questions')
    #     impl = qa.get('implicit_questions')

    #     if isinstance(expl, list) and expl:
    #         lines.append("Explicit Questions:")
    #         for i, q in enumerate(expl, 1):
    #             if isinstance(q, str) and q.strip():
    #                 lines.append(f"{i}. {q.strip()}")

    #     if isinstance(impl, list) and impl:
    #         lines.append("Implicit Questions:")
    #         for i, q in enumerate(impl, 1):
    #             if isinstance(q, str) and q.strip():
    #                 lines.append(f"{i}. {q.strip()}")

    #     if lines:
    #         summary_file = os.path.join(OUTPUT_DIR, f"{base}_QnA_Summary.txt")
    #         with open(summary_file, 'w', encoding='utf-8') as f:
    #             f.write("\n".join(lines) + "\n")
    #         print(f"Q&A summary saved to: {summary_file}")
    #     else:
    #         logging.info("QnA returned no questions to write.")


    # # --- Summary
    # summary = process_with_prompt('SummaryBDDPrompt.json', transcript_text)
    # if summary:
    #     summary_file = os.path.join(OUTPUT_DIR, f"{base}_Summary_Summary.txt")
    #     with open(summary_file, 'w', encoding='utf-8') as f: f.write(summary.get('Summary', '').strip()+"\n")
    #     print(f"Summary summary saved to: {summary_file}")

    # # --- Progress
    # progress = process_with_prompt('ProgressBDDPrompt.json', transcript_text)
    # if progress:
    #     summary_file = os.path.join(OUTPUT_DIR, f"{base}_Progress_Summary.txt")
    #     with open(summary_file, 'w', encoding='utf-8') as f: f.write(progress.get('Progress', '').strip()+"\n")
    #     print(f"Progress summary saved to: {summary_file}")

# --- Main Entry Point ---
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python TheTrustEQ.py <TranscriptFilename>')
        sys.exit(1)
    process_transcript(sys.argv[1])
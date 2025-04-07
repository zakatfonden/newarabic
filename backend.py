# backend.py (Using Document AI Extraction, Translation, and Merging - Fix st NameError)

import io
import google.generativeai as genai
from docx import Document # For reading/writing .docx
from docxcompose.composer import Composer
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import logging
import os
import json # Needed again for credentials

# --- NEW: Import Document AI Client ---
from google.cloud import documentai
# ---

# --- FIX: Import Streamlit ---
import streamlit as st
# ---

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# --- RE-INTRODUCED: Runtime Credentials Setup (Needed for Document AI) ---
# Uses GOOGLE_CREDENTIALS_JSON secret from Streamlit
CREDENTIALS_FILENAME = "google_credentials.json"
_credentials_configured = False # Flag to track if setup was attempted

# Check for secret first (preferred method for Streamlit Cloud)
# This block now correctly uses 'st' after the import was added
if "GOOGLE_CREDENTIALS_JSON" in st.secrets:
    logging.info("Found GOOGLE_CREDENTIALS_JSON in Streamlit Secrets. Setting up credentials file.")
    try:
        credentials_json_content_from_secrets = st.secrets["GOOGLE_CREDENTIALS_JSON"]
        logging.info(f"Read {len(credentials_json_content_from_secrets)} characters from secret.")

        if not credentials_json_content_from_secrets.strip():
                logging.error("GOOGLE_CREDENTIALS_JSON secret is empty.")
                _credentials_configured = False
        else:
            file_written_successfully = False
            try:
                # Basic cleaning attempt for private key newlines
                cleaned_content = credentials_json_content_from_secrets
                try:
                    temp_data = json.loads(credentials_json_content_from_secrets)
                    if 'private_key' in temp_data and isinstance(temp_data['private_key'], str):
                        original_pk = temp_data['private_key']
                        cleaned_pk = original_pk.replace('\r\n', '\n').replace('\r', '\n').replace('\\n', '\n')
                        if cleaned_pk != original_pk:
                            logging.warning("Attempted to clean newline characters from private_key string.")
                            temp_data['private_key'] = cleaned_pk
                            cleaned_content = json.dumps(temp_data, indent=2)
                        else:
                            cleaned_content = json.dumps(temp_data, indent=2) # Ensure consistent formatting
                    else:
                         # If no private key or not string, just dump original parsed data if possible
                         cleaned_content = json.dumps(temp_data, indent=2)
                except json.JSONDecodeError:
                    logging.warning("Initial parse for cleaning failed. Writing raw secret string (may cause issues).")
                    cleaned_content = credentials_json_content_from_secrets # Fallback to raw

                with open(CREDENTIALS_FILENAME, "w", encoding='utf-8') as f:
                    f.write(cleaned_content)
                logging.info(f"Successfully wrote credentials to {CREDENTIALS_FILENAME}")
                file_written_successfully = True
            except Exception as write_err:
                logging.error(f"CRITICAL Error writing credentials file: {write_err}", exc_info=True)
                _credentials_configured = False

            if file_written_successfully:
                try:
                    # Verify file exists and set environment variable
                    if os.path.exists(CREDENTIALS_FILENAME):
                        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_FILENAME
                        logging.info(f"GOOGLE_APPLICATION_CREDENTIALS set to: {CREDENTIALS_FILENAME}")
                        # Final check: Try to initialize client briefly to catch auth errors early (optional)
                        try:
                             # Example: Initialize a client briefly to test credentials
                             # Replace 'us' with a valid location if needed for testing
                             test_opts = {"api_endpoint": "us-documentai.googleapis.com"}
                             test_client = documentai.DocumentProcessorServiceClient(client_options=test_opts)
                             logging.info("Credential file seems valid (client initialized).")
                             _credentials_configured = True # Mark as configured ONLY if successful
                        except Exception as client_init_err:
                             logging.error(f"Credential validation failed: Could not initialize Document AI client. Error: {client_init_err}", exc_info=True)
                             _credentials_configured = False
                    else:
                         logging.error(f"Credentials file {CREDENTIALS_FILENAME} not found after writing.")
                         _credentials_configured = False
                except Exception as env_err:
                    logging.error(f"Error setting or checking credentials environment: {env_err}", exc_info=True)
                    _credentials_configured = False

    except Exception as e:
        logging.error(f"CRITICAL Error reading/processing GOOGLE_CREDENTIALS_JSON secret: {e}", exc_info=True)
        _credentials_configured = False

# Fallback check for externally set environment variable
elif "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
    logging.info("Using GOOGLE_APPLICATION_CREDENTIALS environment variable set externally.")
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") and os.path.exists(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]):
        logging.info(f"External credentials file found at: {os.environ['GOOGLE_APPLICATION_CREDENTIALS']}")
        _credentials_configured = True
    else:
        logging.error(f"External GOOGLE_APPLICATION_CREDENTIALS path not found or not set: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}")
        _credentials_configured = False
else:
    logging.warning("Google Cloud Credentials NOT found: Set GOOGLE_CREDENTIALS_JSON secret or GOOGLE_APPLICATION_CREDENTIALS env var.")
    _credentials_configured = False
# --- END: Runtime Credentials Setup ---


# --- Function to extract text using Document AI (Unchanged) ---
def extract_text_with_docai(file_obj, project_id: str, location: str, processor_id: str):
    """
    Extracts text from a PDF/image file object using Google Cloud Document AI.
    """
    global _credentials_configured
    if not _credentials_configured:
        return "Error: Document AI authentication failed (Credentials not configured)."
    if not all([project_id, location, processor_id]):
        return "Error: Missing Document AI configuration (Project ID, Location, or Processor ID)."
    try:
        opts = {"api_endpoint": f"{location}-documentai.googleapis.com"}
        client = documentai.DocumentProcessorServiceClient(client_options=opts)
        name = client.processor_path(project_id, location, processor_id)
        file_obj.seek(0)
        image_content = file_obj.read()
        mime_type = file_obj.type
        if not mime_type:
            ext = os.path.splitext(file_obj.name)[1].lower()
            if ext == ".pdf": mime_type = "application/pdf"
            elif ext == ".png": mime_type = "image/png"
            elif ext in [".jpg", ".jpeg"]: mime_type = "image/jpeg"
            elif ext in [".tif", ".tiff"]: mime_type = "image/tiff"
            else: return f"Error: Cannot determine MIME type for file '{file_obj.name}'."
            logging.warning(f"Guessed MIME type '{mime_type}' from extension for {file_obj.name}")

        raw_document = documentai.RawDocument(content=image_content, mime_type=mime_type)
        request = documentai.ProcessRequest(name=name, raw_document=raw_document)
        logging.info(f"Sending request to Document AI processor: {name} for file: {file_obj.name} ({mime_type})")
        result = client.process_document(request=request)
        document = result.document
        logging.info(f"Received response from Document AI. Extracted text length: {len(document.text)}")
        return document.text
    except Exception as e:
        logging.error(f"Error calling Document AI for file '{file_obj.name}': {e}", exc_info=True)
        return f"Error: Failed to process document with Document AI. Details: {e}"


# --- Gemini Processing (Unchanged) ---
def process_text_with_gemini(api_key: str, raw_text: str, rules_prompt: str, model_name: str):
    """
    Sends text (extracted via Document AI) to Gemini for translation based on rules.
    """
    if not api_key: return "Error: Gemini API key is missing."
    if not raw_text or not raw_text.strip():
        logging.warning("Skipping Gemini call: No text extracted from Document AI.")
        return ""
    if not model_name: return "Error: Gemini model name not specified."
    try:
        genai.configure(api_key=api_key)
        logging.info(f"Initializing Gemini model: {model_name}")
        model = genai.GenerativeModel(model_name)
        full_prompt = f"**Instructions:**\n{rules_prompt}\n\n**Text to Process:**\n---\n{raw_text}\n---\n\n**Output:**\nReturn ONLY the processed text (Arabic translation) according to the instructions."
        logging.info(f"Sending request to Gemini model: {model_name} for translation. Text length: {len(raw_text)}")
        response = model.generate_content(full_prompt)
        if not response.parts:
            block_reason = getattr(getattr(response, 'prompt_feedback', None), 'block_reason', None)
            if block_reason:
                logging.error(f"Gemini request ({model_name}) blocked. Reason: {block_reason}.")
                return f"Error: Content blocked by Gemini safety filters. Reason: {block_reason}"
            else:
                 logging.warning(f"Gemini ({model_name}) returned no parts (empty response).")
                 return ""
        processed_text = response.text
        logging.info(f"Successfully received translation from Gemini ({model_name}). Length: {len(processed_text)}")
        return processed_text
    except Exception as e:
        logging.error(f"Error interacting with Gemini API ({model_name}): {e}", exc_info=True)
        return f"Error: Failed to process text with Gemini ({model_name}). Details: {e}"


# --- Create SINGLE Word Document (Unchanged) ---
def create_arabic_word_doc_from_text(arabic_text: str, filename: str):
    """
    Creates a single Word document (.docx) in memory containing the translated Arabic text.
    """
    try:
        document = Document()
        style = document.styles['Normal']
        font = style.font; font.name = 'Arial'; font.rtl = True
        style_element = style.element
        rpr = style_element.xpath('.//w:rPr')[0] if style_element.xpath('.//w:rPr') else OxmlElement('w:rPr')
        if not style_element.xpath('.//w:rPr'): style_element.append(rpr)
        font_name_el = rpr.find(qn('w:rFonts'))
        if font_name_el is None: font_name_el = OxmlElement('w:rFonts'); rpr.append(font_name_el)
        font_name_el.set(qn('w:cs'), 'Arial')
        p_fmt = style.paragraph_format; p_fmt.alignment = WD_ALIGN_PARAGRAPH.RIGHT; p_fmt.right_to_left = True
        if arabic_text and arabic_text.strip():
            lines = arabic_text.strip().split('\n')
            for line in lines:
                if line.strip():
                    p = document.add_paragraph(line.strip())
                    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.RIGHT; p.paragraph_format.right_to_left = True
                    for run in p.runs: run.font.name = 'Arial'; run.font.rtl = True; run.font.complex_script = True
        else:
            p = document.add_paragraph(f"[No translation generated for '{filename}']")
            p.italic = True; p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.RIGHT; p.paragraph_format.right_to_left = True
            for run in p.runs: run.font.name = 'Arial'; run.font.rtl = True; run.font.complex_script = True
        doc_stream = io.BytesIO(); document.save(doc_stream); doc_stream.seek(0)
        logging.info(f"Successfully created intermediate Word doc stream for '{filename}'.")
        return doc_stream
    except Exception as e:
        logging.error(f"Error creating intermediate Word document for '{filename}': {e}", exc_info=True)
        return None


# --- Merge Word Documents (Unchanged) ---
def merge_word_documents(doc_streams_data: list[tuple[str, io.BytesIO]]):
    """
    Merges multiple Word documents (provided as BytesIO streams) into one single document.
    """
    if not doc_streams_data: return None
    try:
        first_filename, first_stream = doc_streams_data[0]; first_stream.seek(0)
        master_doc = Document(first_stream); composer = Composer(master_doc)
        logging.info(f"Initialized merger with base document from '{first_filename}'.")
        if len(doc_streams_data) > 1:
            for i in range(1, len(doc_streams_data)):
                filename, stream = doc_streams_data[i]; stream.seek(0)
                logging.info(f"Appending content from '{filename}'...")
                sub_doc = Document(stream); composer.master.add_page_break()
                composer.append(sub_doc)
                logging.info(f"Successfully appended content from '{filename}'.")
        merged_stream = io.BytesIO(); composer.save(merged_stream); merged_stream.seek(0)
        logging.info(f"Successfully merged {len(doc_streams_data)} documents.")
        return merged_stream
    except Exception as e:
        logging.error(f"Error merging Word documents using docxcompose: {e}", exc_info=True)
        return None

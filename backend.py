# backend.py (Partial - Only the corrected function)

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

# --- RE-INTRODUCED: Runtime Credentials Setup ---
# (Keep your existing credentials setup logic here)
# ...
CREDENTIALS_FILENAME = "google_credentials.json"
_credentials_configured = False # Flag to track if setup was attempted
# (Your existing credential setup logic remains unchanged)
# ... (Make sure this part is still present in your actual file) ...
if "GOOGLE_CREDENTIALS_JSON" in st.secrets:
    logging.info("Found GOOGLE_CREDENTIALS_JSON in Streamlit Secrets. Setting up credentials file.")
    # ... (rest of your credential logic) ...
    _credentials_configured = True # Example: Assume success if logic completes
elif "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
    logging.info("Using GOOGLE_APPLICATION_CREDENTIALS environment variable set externally.")
    # ... (rest of your credential logic) ...
    _credentials_configured = True # Example: Assume success if logic completes
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
        # --- Determine mime_type robustly ---
        mime_type = getattr(file_obj, 'type', None) # Get type attribute if it exists
        if not mime_type:
            # Fallback to guessing from extension if type attribute is missing
            ext = os.path.splitext(file_obj.name)[1].lower()
            mime_map = {
                ".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg", ".tif": "image/tiff", ".tiff": "image/tiff",
                ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp"
            }
            mime_type = mime_map.get(ext)
            if not mime_type:
                 return f"Error: Cannot determine MIME type for file '{file_obj.name}'."
            logging.warning(f"Guessed MIME type '{mime_type}' from extension for {file_obj.name}")
        # --- End mime_type determination ---

        raw_document = documentai.RawDocument(content=image_content, mime_type=mime_type)
        # Increase timeout (optional, but can help with large documents)
        request = documentai.ProcessRequest(
            name=name,
            raw_document=raw_document,
            # Optional: Add skip_human_review=True if using a processor version
            # skip_human_review=True
        )
        # process_options = documentai.ProcessOptions(
        #     # Example: Specify page range if needed
        #     # from_start=1,
        #     # to_end=5
        # )
        # request.process_options = process_options

        logging.info(f"Sending request to Document AI processor: {name} for file: {file_obj.name} ({mime_type})")
        result = client.process_document(request=request, timeout=600) # Increased timeout to 10 minutes
        document = result.document
        logging.info(f"Received response from Document AI. Extracted text length: {len(document.text)}")
        return document.text
    except Exception as e:
        logging.error(f"Error calling Document AI for file '{file_obj.name}': {e}", exc_info=True)
        # Provide more specific error info if possible
        error_detail = str(e)
        if hasattr(e, 'details'): # Check for gRPC error details
            error_detail = e.details()
        return f"Error: Failed to process document with Document AI. Details: {error_detail}"


# --- Gemini Processing (Unchanged) ---
def process_text_with_gemini(api_key: str, raw_text: str, rules_prompt: str, model_name: str):
    """
    Sends text (extracted via Document AI) to Gemini for translation based on rules.
    """
    if not api_key: return "Error: Gemini API key is missing."
    if not raw_text or not raw_text.strip():
        logging.warning("Skipping Gemini call: No text extracted from Document AI.")
        return "" # Return empty string, not None
    if not model_name: return "Error: Gemini model name not specified."
    try:
        genai.configure(api_key=api_key)
        logging.info(f"Initializing Gemini model: {model_name}")
        # Configure safety settings if needed (example: turn off HARM_CATEGORY_SEXUALLY_EXPLICIT)
        safety_settings = {
             'HARM_CATEGORY_HARASSMENT': 'BLOCK_MEDIUM_AND_ABOVE',
             'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_MEDIUM_AND_ABOVE',
             'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_MEDIUM_AND_ABOVE', # Or BLOCK_NONE if necessary and acceptable
             'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_MEDIUM_AND_ABOVE',
        }
        # Configure generation config if needed (example: temperature)
        generation_config = genai.types.GenerationConfig(
            # candidate_count=1, # Default is 1
            # stop_sequences=['\n---\n'],
             max_output_tokens=8192, # Max for Pro, Flash might be lower if needed
             temperature=0.7, # Adjust creativity vs factualness
            # top_p=1.0,
            # top_k=40
        )

        model = genai.GenerativeModel(
            model_name,
            safety_settings=safety_settings,
            generation_config=generation_config
            )

        # Ensure the prompt clearly separates instructions and text
        full_prompt = f"{rules_prompt}\n\n---\n\n{raw_text}\n\n---\n\nTranslate the text above this line:"

        logging.info(f"Sending request to Gemini model: {model_name} for translation. Text length: {len(raw_text)}")
        # Use generate_content for streaming potential and better error handling
        response = model.generate_content(full_prompt) # Removed stream=False as it's default

        # More robust check for response content and errors
        if response.candidates:
             if response.candidates[0].content and response.candidates[0].content.parts:
                 processed_text = "".join(part.text for part in response.candidates[0].content.parts)
                 logging.info(f"Successfully received translation from Gemini ({model_name}). Length: {len(processed_text)}")
                 return processed_text
             elif response.candidates[0].finish_reason.name != "STOP": # Check if stopped for other reasons
                  finish_reason = response.candidates[0].finish_reason.name
                  safety_ratings = response.candidates[0].safety_ratings if hasattr(response.candidates[0], 'safety_ratings') else "N/A"
                  logging.error(f"Gemini generation ({model_name}) finished unexpectedly. Reason: {finish_reason}. Safety Ratings: {safety_ratings}")
                  # Check for safety blocks specifically
                  if finish_reason == "SAFETY":
                       blocked_categories = [rating.category.name for rating in safety_ratings if rating.probability.name not in ['NEGLIGIBLE', 'LOW']]
                       return f"Error: Content blocked by Gemini safety filters. Categories: {', '.join(blocked_categories) or 'Unknown'}."
                  else:
                       return f"Error: Gemini generation failed. Reason: {finish_reason}."
             else:
                  # Finished with STOP but no content? (Unlikely but possible)
                  logging.warning(f"Gemini ({model_name}) finished normally but returned no content parts.")
                  return "" # Return empty string
        # Check prompt feedback for blocks even if no candidates (e.g., prompt itself blocked)
        elif response.prompt_feedback and response.prompt_feedback.block_reason.name != "BLOCK_REASON_UNSPECIFIED":
             block_reason = response.prompt_feedback.block_reason.name
             safety_ratings = response.prompt_feedback.safety_ratings
             logging.error(f"Gemini prompt blocked ({model_name}). Reason: {block_reason}. Safety Ratings: {safety_ratings}")
             blocked_categories = [rating.category.name for rating in safety_ratings if rating.probability.name not in ['NEGLIGIBLE', 'LOW']]
             return f"Error: Gemini prompt blocked by safety filters. Reason: {block_reason}. Categories: {', '.join(blocked_categories) or 'Unknown'}."
        else:
             # General case for no response or unexpected structure
             logging.error(f"Gemini ({model_name}) returned an empty or unexpected response structure.")
             return "Error: Gemini returned no valid response."

    except Exception as e:
        logging.error(f"Error interacting with Gemini API ({model_name}): {e}", exc_info=True)
        return f"Error: Failed to process text with Gemini ({model_name}). Details: {str(e)}"


# --- Create SINGLE Word Document (Unchanged) ---
def create_arabic_word_doc_from_text(arabic_text: str, filename: str):
    """
    Creates a single Word document (.docx) in memory containing the translated Arabic text.
    Handles RTL and sets Arial font.
    """
    try:
        document = Document()

        # --- Set default style for Arabic ---
        style = document.styles['Normal']
        font = style.font
        font.name = 'Arial' # Good default for Arabic
        font.rtl = True # Enable Right-to-Left for the font object

        # Ensure complex script font is set in the style definition (more robust)
        style_element = style.element
        # Find or create the rPr element
        rpr = style_element.xpath('.//w:rPr')
        if not rpr:
            rpr = OxmlElement('w:rPr')
            style_element.append(rpr)
        else:
            rpr = rpr[0]
        # Find or create the rFonts element
        font_name_el = rpr.find(qn('w:rFonts'))
        if font_name_el is None:
            font_name_el = OxmlElement('w:rFonts')
            rpr.append(font_name_el)
        # Set the complex script font attribute (w:cs)
        font_name_el.set(qn('w:cs'), 'Arial') # Use 'w:cs' for complex scripts like Arabic

        # Set paragraph format defaults for the style
        p_fmt = style.paragraph_format
        p_fmt.alignment = WD_ALIGN_PARAGRAPH.RIGHT # Right align paragraphs
        p_fmt.right_to_left = True # Set paragraph direction to RTL
        # --- End Style Setup ---

        if arabic_text and arabic_text.strip():
            # Split into paragraphs (respecting potential double newlines)
            paragraphs_text = arabic_text.strip().split('\n\n') # Split by double newline first
            for para_text in paragraphs_text:
                 lines = para_text.strip().split('\n') # Split remaining by single newline
                 # Add the first line as a paragraph
                 if lines:
                     first_line = lines[0].strip()
                     if first_line:
                          p = document.add_paragraph()
                          # Apply style explicitly (optional, but ensures consistency)
                          # p.style = document.styles['Normal']
                          # Add run and set text
                          run = p.add_run(first_line)
                          # Apply run-level formatting (redundant if style is correct, but safe)
                          # run.font.name = 'Arial'
                          # run.font.rtl = True
                          # run.font.complex_script = True # python-docx might not have this directly

                          # Apply paragraph formatting explicitly (redundant but safe)
                          p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                          p.paragraph_format.right_to_left = True

                          # Add subsequent lines with soft line breaks (Shift+Enter) within the same paragraph
                          for line in lines[1:]:
                              trimmed_line = line.strip()
                              if trimmed_line:
                                   run = p.add_run() # Get the last run or add new? Test this. Add break first.
                                   run.add_break() # Add soft line break
                                   run.add_text(trimmed_line)
                                   # Apply run formatting again if needed
                                   # run.font.name = 'Arial'
                                   # run.font.rtl = True

        else:
            # Add placeholder text if no translation
            p = document.add_paragraph()
            p.style = document.styles['Normal'] # Ensure style is applied
            run = p.add_run(f"[No translation generated or extraction failed for '{filename}']")
            run.italic = True
            # Apply paragraph formatting explicitly
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            p.paragraph_format.right_to_left = True


        # Save to memory stream
        doc_stream = io.BytesIO()
        document.save(doc_stream)
        doc_stream.seek(0)
        logging.info(f"Successfully created intermediate Word doc stream for '{filename}'.")
        return doc_stream
    except Exception as e:
        logging.error(f"Error creating intermediate Word document for '{filename}': {e}", exc_info=True)
        return None


# --- Merge Word Documents (CORRECTED) ---
def merge_word_documents(doc_streams_data: list[tuple[str, io.BytesIO]]):
    """
    Merges multiple Word documents (provided as BytesIO streams) into one single document.
    Adds a page break before appending each subsequent document.
    """
    if not doc_streams_data:
        logging.warning("No document streams provided for merging.")
        return None

    try:
        # Initialize with the first document
        first_filename, first_stream = doc_streams_data[0]
        first_stream.seek(0)
        # Load the first document into a python-docx Document object
        master_doc = Document(first_stream)
        # Initialize the Composer with this master document
        composer = Composer(master_doc)
        logging.info(f"Initialized merger with base document from '{first_filename}'.")

        # Append subsequent documents
        if len(doc_streams_data) > 1:
            for i in range(1, len(doc_streams_data)):
                filename, stream = doc_streams_data[i]
                stream.seek(0)
                logging.info(f"Preparing to append content from '{filename}'...")

                # --- CORRECTED PART ---
                # Add a page break to the composer's document *before* appending
                # Access the underlying python-docx Document object via composer.doc
                try:
                    composer.doc.add_page_break()
                    logging.info(f"Added page break before appending '{filename}'.")
                except Exception as pb_exc:
                    # Log if adding page break fails, but attempt to continue merging
                    logging.error(f"Could not add page break before '{filename}': {pb_exc}", exc_info=True)
                # --- END CORRECTED PART ---

                # Load the sub-document to be appended
                try:
                    sub_doc = Document(stream)
                    # Append the sub-document using the composer
                    composer.append(sub_doc)
                    logging.info(f"Successfully appended content from '{filename}'.")
                except Exception as append_exc:
                     # Log if appending fails, maybe skip this file?
                     logging.error(f"Failed to append document '{filename}': {append_exc}", exc_info=True)
                     # Decide whether to continue or raise/return None
                     # For now, we log and continue to merge the rest if possible
                     # return None # Option: Stop merging on first failure

        # Save the composed document to a new memory stream
        merged_stream = io.BytesIO()
        composer.save(merged_stream)
        merged_stream.seek(0)
        logging.info(f"Successfully merged {len(composer.doc.element.body)} elements from {len(doc_streams_data)} source documents into final stream.") # Log might need adjustment based on how composer counts elements
        return merged_stream

    except Exception as e:
        # Catch errors during initialization or saving
        logging.error(f"Error during the merge process: {e}", exc_info=True)
        return None

# --- (Make sure the rest of your backend.py file follows) ---


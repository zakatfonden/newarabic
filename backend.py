# backend.py (Complete and Updated)

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
# Configures basic logging to provide feedback on the backend operations.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# --- RE-INTRODUCED: Runtime Credentials Setup ---
# This section handles Google Cloud credentials, needed for Document AI.
# It prioritizes Streamlit secrets, then environment variables.
CREDENTIALS_FILENAME = "google_credentials.json" # Temp file to store credentials if using secrets
_credentials_configured = False # Flag to track if credential setup was successful

# Try getting credentials from Streamlit secrets first
if "GOOGLE_CREDENTIALS_JSON" in st.secrets:
    logging.info("Found GOOGLE_CREDENTIALS_JSON in Streamlit Secrets. Setting up credentials file.")
    try:
        # Read the JSON content from secrets
        credentials_json_content_from_secrets = st.secrets["GOOGLE_CREDENTIALS_JSON"]
        logging.info(f"Read {len(credentials_json_content_from_secrets)} characters from secret.")

        # Basic check if the secret is empty
        if not credentials_json_content_from_secrets.strip():
            logging.error("GOOGLE_CREDENTIALS_JSON secret is empty.")
            _credentials_configured = False
        else:
            file_written_successfully = False
            try:
                # Attempt to parse and clean potential newline issues in the private key
                cleaned_content = credentials_json_content_from_secrets
                try:
                    temp_data = json.loads(credentials_json_content_from_secrets)
                    if 'private_key' in temp_data and isinstance(temp_data['private_key'], str):
                        original_pk = temp_data['private_key']
                        # Replace common problematic newline representations
                        cleaned_pk = original_pk.replace('\\r\\n', '\n').replace('\\r', '\n').replace('\\n', '\n')
                        if cleaned_pk != original_pk:
                            logging.warning("Attempted to clean newline characters from private_key string in credentials.")
                            temp_data['private_key'] = cleaned_pk
                        # Re-serialize with cleaned key or just for consistent formatting
                        cleaned_content = json.dumps(temp_data, indent=2)
                    else:
                        # If no private key or not a string, just format the parsed data
                         cleaned_content = json.dumps(temp_data, indent=2)
                except json.JSONDecodeError:
                    # If parsing fails, use the raw content (might still cause issues later)
                    logging.warning("Initial JSON parse for cleaning credentials failed. Writing raw secret string (may cause auth issues).")
                    cleaned_content = credentials_json_content_from_secrets # Fallback

                # Write the (potentially cleaned) credentials to a temporary file
                with open(CREDENTIALS_FILENAME, "w", encoding='utf-8') as f:
                    f.write(cleaned_content)
                logging.info(f"Successfully wrote credentials to {CREDENTIALS_FILENAME}")
                file_written_successfully = True
            except Exception as write_err:
                logging.error(f"CRITICAL Error writing credentials file '{CREDENTIALS_FILENAME}': {write_err}", exc_info=True)
                _credentials_configured = False

            # If the file was written, set the environment variable and verify
            if file_written_successfully:
                try:
                    if os.path.exists(CREDENTIALS_FILENAME):
                        # Point Google Cloud libraries to the credentials file
                        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_FILENAME
                        logging.info(f"GOOGLE_APPLICATION_CREDENTIALS set to: {CREDENTIALS_FILENAME}")

                        # Optional: Try a quick client initialization to validate credentials early
                        try:
                            # Use a common region like 'us' for the test endpoint
                            test_opts = {"api_endpoint": "us-documentai.googleapis.com"}
                            test_client = documentai.DocumentProcessorServiceClient(client_options=test_opts)
                            logging.info("Credential file seems valid (Document AI client initialized successfully).")
                            _credentials_configured = True # Mark as configured ONLY if successful
                        except Exception as client_init_err:
                            logging.error(f"Credential validation failed: Could not initialize Document AI client. Error: {client_init_err}", exc_info=True)
                            _credentials_configured = False
                    else:
                        logging.error(f"Credentials file {CREDENTIALS_FILENAME} not found after attempting to write it.")
                        _credentials_configured = False
                except Exception as env_err:
                    logging.error(f"Error setting or checking credentials environment variable: {env_err}", exc_info=True)
                    _credentials_configured = False

    except Exception as e:
        # Catch errors reading the secret itself
        logging.error(f"CRITICAL Error reading/processing GOOGLE_CREDENTIALS_JSON secret: {e}", exc_info=True)
        _credentials_configured = False

# Fallback: Check if the environment variable is set externally (e.g., in local dev)
elif "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
    logging.info("Using GOOGLE_APPLICATION_CREDENTIALS environment variable set externally.")
    ext_cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if ext_cred_path and os.path.exists(ext_cred_path):
        logging.info(f"External credentials file found at: {ext_cred_path}")
        _credentials_configured = True # Assume valid if path exists
    else:
        logging.error(f"External GOOGLE_APPLICATION_CREDENTIALS path not found or invalid: '{ext_cred_path}'")
        _credentials_configured = False
# If neither secret nor env var is found
else:
    logging.warning("Google Cloud Credentials NOT found: Set 'GOOGLE_CREDENTIALS_JSON' Streamlit secret or 'GOOGLE_APPLICATION_CREDENTIALS' environment variable.")
    _credentials_configured = False
# --- END: Runtime Credentials Setup ---


# --- Function to extract text using Document AI ---
def extract_text_with_docai(file_obj, project_id: str, location: str, processor_id: str):
    """
    Extracts text from a PDF or image file object using Google Cloud Document AI.

    Args:
        file_obj: A file-like object (e.g., from st.file_uploader) containing the document.
        project_id: Google Cloud Project ID.
        location: Region of the Document AI processor (e.g., 'us', 'eu').
        processor_id: The ID of the Document AI processor to use.

    Returns:
        A string containing the extracted text, or an error message string.
    """
    global _credentials_configured
    # Check if credentials were set up successfully earlier
    if not _credentials_configured:
        logging.error("Document AI call skipped: Credentials not configured.")
        return "Error: Document AI authentication failed (Credentials not configured)."
    # Check if required configuration parameters are provided
    if not all([project_id, location, processor_id]):
        logging.error("Document AI call skipped: Missing configuration.")
        return "Error: Missing Document AI configuration (Project ID, Location, or Processor ID)."

    try:
        # Set the regional API endpoint for the Document AI client
        opts = {"api_endpoint": f"{location}-documentai.googleapis.com"}
        client = documentai.DocumentProcessorServiceClient(client_options=opts)

        # Construct the full processor resource name
        name = client.processor_path(project_id, location, processor_id)

        # Read the file content
        file_obj.seek(0) # Ensure reading from the start
        image_content = file_obj.read()

        # --- Determine mime_type robustly ---
        # Prefer the 'type' attribute if available (set by Streamlit uploader)
        mime_type = getattr(file_obj, 'type', None)
        if not mime_type:
            # Fallback to guessing from the filename extension
            ext = os.path.splitext(file_obj.name)[1].lower()
            mime_map = {
                ".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg", ".tif": "image/tiff", ".tiff": "image/tiff",
                ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp"
            }
            mime_type = mime_map.get(ext)
            # If type cannot be determined, return an error
            if not mime_type:
                 logging.error(f"Cannot determine MIME type for file '{file_obj.name}' from extension '{ext}'.")
                 return f"Error: Cannot determine MIME type for file '{file_obj.name}'."
            logging.warning(f"Guessed MIME type '{mime_type}' from extension for {file_obj.name}")
        # --- End mime_type determination ---

        # Prepare the raw document content for the API request
        raw_document = documentai.RawDocument(content=image_content, mime_type=mime_type)

        # Prepare the process request
        request = documentai.ProcessRequest(
            name=name,
            raw_document=raw_document,
            # Optional: Add skip_human_review=True if using a processor version that supports it
            # skip_human_review=True
        )
        # Optional: Add process options like page range if needed
        # process_options = documentai.ProcessOptions(from_start=1, to_end=5)
        # request.process_options = process_options

        logging.info(f"Sending request to Document AI processor: {name} for file: {file_obj.name} ({mime_type})")
        # Make the API call with an increased timeout (e.g., 10 minutes for large docs)
        result = client.process_document(request=request, timeout=600)
        document = result.document

        logging.info(f"Received response from Document AI for '{file_obj.name}'. Extracted text length: {len(document.text)}")
        # Return the extracted text
        return document.text

    except Exception as e:
        # Log the error and return a user-friendly error message
        logging.error(f"Error calling Document AI for file '{file_obj.name}': {e}", exc_info=True)
        # Try to get more specific error details if available (e.g., from gRPC errors)
        error_detail = str(e)
        if hasattr(e, 'details') and callable(e.details):
            error_detail = e.details()
        elif hasattr(e, 'message'): # Handle other potential error attributes
             error_detail = e.message
        return f"Error: Failed to process document with Document AI. Details: {error_detail}"


# --- Gemini Processing ---
def process_text_with_gemini(api_key: str, raw_text: str, rules_prompt: str, model_name: str):
    """
    Sends text (e.g., extracted via Document AI) to the Gemini API for processing
    (e.g., translation) based on provided rules/prompt.

    Args:
        api_key: The Google Gemini API key.
        raw_text: The input text to be processed.
        rules_prompt: Instructions for the Gemini model (e.g., "Translate to Arabic...").
        model_name: The specific Gemini model ID to use (e.g., "gemini-1.5-flash-latest").

    Returns:
        A string containing the processed text from Gemini, an empty string if input
        text was empty, or an error message string.
    """
    # Check for API key
    if not api_key:
        logging.error("Gemini call skipped: API key missing.")
        return "Error: Gemini API key is missing."
    # If input text is empty or whitespace only, skip the API call
    if not raw_text or not raw_text.strip():
        logging.warning("Skipping Gemini call: Input text is empty.")
        return "" # Return empty string, consistent with success but no output
    # Check if model name is provided
    if not model_name:
        logging.error("Gemini call skipped: Model name not specified.")
        return "Error: Gemini model name not specified."

    try:
        # Configure the Gemini client library with the API key
        genai.configure(api_key=api_key)
        logging.info(f"Initializing Gemini model: {model_name}")

        # --- Configure Safety Settings (Optional but Recommended) ---
        # Adjust blocking thresholds as needed. BLOCK_NONE should be used cautiously.
        safety_settings = {
             'HARM_CATEGORY_HARASSMENT': 'BLOCK_MEDIUM_AND_ABOVE',
             'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_MEDIUM_AND_ABOVE',
             'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_MEDIUM_AND_ABOVE',
             'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_MEDIUM_AND_ABOVE',
        }

        # --- Configure Generation Parameters (Optional) ---
        # Control aspects like creativity (temperature), output length, etc.
        generation_config = genai.types.GenerationConfig(
            # candidate_count=1, # How many responses to generate (default 1)
            # stop_sequences=['\n---\n'], # Custom sequences to stop generation
             max_output_tokens=8192, # Max tokens for the response (check model limits)
             temperature=0.7, # Lower values = more deterministic, Higher = more creative
            # top_p=1.0, # Nucleus sampling probability threshold
            # top_k=40 # Consider only top_k tokens at each step
        )

        # Initialize the generative model with configuration
        model = genai.GenerativeModel(
            model_name,
            safety_settings=safety_settings,
            generation_config=generation_config
        )

        # --- Construct the Prompt ---
        # Clearly separate instructions, input text, and expected output format.
        # Using separators like '---' can help the model distinguish sections.
        full_prompt = f"{rules_prompt}\n\n---\n\n{raw_text}\n\n---\n\nTranslate the text above this line:"

        logging.info(f"Sending request to Gemini model: {model_name}. Input text length: {len(raw_text)}")

        # --- Make the API Call ---
        # Use generate_content for robust handling of responses and errors
        response = model.generate_content(full_prompt)

        # --- Process the Response ---
        # Check if candidates were generated
        if response.candidates:
            candidate = response.candidates[0] # Usually only one candidate unless configured otherwise
            # Check if the candidate has content parts
            if candidate.content and candidate.content.parts:
                # Concatenate text from all parts
                processed_text = "".join(part.text for part in candidate.content.parts)
                logging.info(f"Successfully received response from Gemini ({model_name}). Output length: {len(processed_text)}")
                return processed_text
            # Check if generation stopped for reasons other than normal completion
            elif candidate.finish_reason.name != "STOP":
                finish_reason = candidate.finish_reason.name
                safety_ratings = getattr(candidate, 'safety_ratings', "N/A") # Get safety ratings if available
                logging.error(f"Gemini generation ({model_name}) finished unexpectedly. Reason: {finish_reason}. Safety Ratings: {safety_ratings}")
                # Specifically handle safety blocks
                if finish_reason == "SAFETY":
                    # Extract categories that caused the block
                    blocked_categories = [rating.category.name for rating in safety_ratings if rating.probability.name not in ['NEGLIGIBLE', 'LOW']]
                    return f"Error: Content blocked by Gemini safety filters. Categories: {', '.join(blocked_categories) or 'Unknown'}."
                else:
                    # Handle other non-STOP finish reasons (e.g., MAX_TOKENS, RECITATION)
                    return f"Error: Gemini generation failed. Reason: {finish_reason}."
            else:
                # Finished normally (STOP) but no content parts? (Should be rare)
                logging.warning(f"Gemini ({model_name}) finished normally but returned no content parts.")
                return "" # Return empty string
        # Check if the prompt itself was blocked (even if no candidates generated)
        elif response.prompt_feedback and response.prompt_feedback.block_reason.name != "BLOCK_REASON_UNSPECIFIED":
            block_reason = response.prompt_feedback.block_reason.name
            safety_ratings = response.prompt_feedback.safety_ratings
            logging.error(f"Gemini prompt blocked ({model_name}). Reason: {block_reason}. Safety Ratings: {safety_ratings}")
            blocked_categories = [rating.category.name for rating in safety_ratings if rating.probability.name not in ['NEGLIGIBLE', 'LOW']]
            return f"Error: Gemini prompt blocked by safety filters. Reason: {block_reason}. Categories: {', '.join(blocked_categories) or 'Unknown'}."
        else:
            # General case for empty or unexpected response structure
            logging.error(f"Gemini ({model_name}) returned an empty or unexpected response structure: {response}")
            return "Error: Gemini returned no valid response."

    except Exception as e:
        # Catch general exceptions during API interaction
        logging.error(f"Error interacting with Gemini API ({model_name}): {e}", exc_info=True)
        return f"Error: Failed to process text with Gemini ({model_name}). Details: {str(e)}"


# --- Create SINGLE Word Document ---
def create_arabic_word_doc_from_text(arabic_text: str, filename: str):
    """
    Creates a single Word document (.docx) in memory containing the provided text,
    formatted for Right-to-Left (RTL) Arabic script using Arial font.

    Args:
        arabic_text: The string containing the Arabic text (or other text to format RTL).
        filename: The original filename, used for placeholder text if input is empty.

    Returns:
        A BytesIO stream containing the Word document, or None if an error occurs.
    """
    try:
        # Create a new Word document object
        document = Document()

        # --- Set Default Style for Arabic/RTL ---
        # Get the default 'Normal' style
        style = document.styles['Normal']
        # Set font properties for the style
        font = style.font
        font.name = 'Arial' # A common font supporting Arabic
        font.rtl = True     # Enable Right-to-Left rendering for the font

        # Ensure complex script font settings are applied via underlying XML (more reliable)
        style_element = style.element # Get the OxmlElement for the style
        # Find or create the run properties element (<w:rPr>)
        rpr = style_element.xpath('.//w:rPr')
        if not rpr:
            rpr = OxmlElement('w:rPr')
            style_element.append(rpr)
        else:
            rpr = rpr[0]
        # Find or create the font definition element (<w:rFonts>)
        font_name_el = rpr.find(qn('w:rFonts')) # qn resolves the namespace prefix
        if font_name_el is None:
            font_name_el = OxmlElement('w:rFonts')
            rpr.append(font_name_el)
        # Set the complex script font attribute (w:cs) to Arial
        font_name_el.set(qn('w:cs'), 'Arial') # 'cs' = Complex Script

        # Set paragraph format defaults for the style
        p_fmt = style.paragraph_format
        p_fmt.alignment = WD_ALIGN_PARAGRAPH.RIGHT # Right-align paragraphs
        p_fmt.right_to_left = True                # Set paragraph direction to RTL
        # --- End Style Setup ---

        # Check if there is text to add
        if arabic_text and arabic_text.strip():
            # Split text into paragraphs based on double newlines (common practice)
            paragraphs_text = arabic_text.strip().split('\n\n')
            for para_text in paragraphs_text:
                # Split paragraphs into lines based on single newlines
                lines = para_text.strip().split('\n')
                if lines:
                    # Add the first line as a new paragraph
                    first_line = lines[0].strip()
                    if first_line:
                        p = document.add_paragraph()
                        # Explicitly apply the 'Normal' style (optional, should inherit)
                        # p.style = document.styles['Normal']

                        # Add the text to the paragraph using a run
                        run = p.add_run(first_line)
                        # Run-level formatting (optional if style is set correctly)
                        # run.font.name = 'Arial'
                        # run.font.rtl = True

                        # Explicitly set paragraph alignment/direction (optional if style is set)
                        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                        p.paragraph_format.right_to_left = True

                        # Add subsequent lines within the same paragraph using soft line breaks
                        for line in lines[1:]:
                            trimmed_line = line.strip()
                            if trimmed_line:
                                # Add a soft line break (like Shift+Enter)
                                p.add_run().add_break()
                                # Add the text for the new line
                                p.add_run(trimmed_line)
                                # Ensure formatting is applied to these runs too if needed

        else:
            # If no text was provided (e.g., extraction/translation failed), add placeholder
            p = document.add_paragraph()
            p.style = document.styles['Normal'] # Ensure style is applied
            run = p.add_run(f"[No translation generated or extraction failed for '{filename}']")
            run.italic = True
            # Apply paragraph formatting explicitly for the placeholder
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            p.paragraph_format.right_to_left = True

        # Save the document content to a BytesIO stream (in-memory file)
        doc_stream = io.BytesIO()
        document.save(doc_stream)
        doc_stream.seek(0) # Reset stream position to the beginning for reading
        logging.info(f"Successfully created intermediate Word doc stream for '{filename}'.")
        return doc_stream # Return the stream

    except Exception as e:
        # Log errors during document creation
        logging.error(f"Error creating intermediate Word document for '{filename}': {e}", exc_info=True)
        return None # Return None to indicate failure


# --- Merge Word Documents (CORRECTED) ---
def merge_word_documents(doc_streams_data: list[tuple[str, io.BytesIO]]):
    """
    Merges multiple Word documents (provided as BytesIO streams) into one single document.
    Adds a page break before appending each subsequent document (after the first).

    Args:
        doc_streams_data: A list of tuples, where each tuple contains:
                          (original_filename: str, doc_stream: io.BytesIO).

    Returns:
        A BytesIO stream containing the merged Word document, or None if an error occurs
        or if the input list is empty.
    """
    # Check if there are any documents to merge
    if not doc_streams_data:
        logging.warning("No document streams provided for merging.")
        return None

    try:
        # Initialize the process with the first document in the list
        first_filename, first_stream = doc_streams_data[0]
        first_stream.seek(0) # Ensure stream is at the beginning

        # Load the first document using python-docx
        master_doc = Document(first_stream)
        # Initialize the Composer object with this first document as the base
        composer = Composer(master_doc)
        logging.info(f"Initialized merger with base document from '{first_filename}'.")

        # Append the rest of the documents (if any)
        if len(doc_streams_data) > 1:
            # Loop through the remaining documents starting from the second one (index 1)
            for i in range(1, len(doc_streams_data)):
                filename, stream = doc_streams_data[i]
                stream.seek(0) # Ensure stream is at the beginning
                logging.info(f"Preparing to append content from '{filename}' (index {i})...")

                # --- Add Page Break Before Appending ---
                # Access the underlying python-docx Document object via composer.doc
                try:
                    composer.doc.add_page_break()
                    logging.info(f"Added page break before appending '{filename}'.")
                except Exception as pb_exc:
                    # Log if adding page break fails, but attempt to continue merging
                    logging.error(f"Could not add page break before '{filename}': {pb_exc}", exc_info=True)
                # --- End Page Break Addition ---

                # Load the sub-document to be appended
                try:
                    sub_doc = Document(stream)
                    # Append the content of the sub-document to the composer's document
                    composer.append(sub_doc)
                    logging.info(f"Successfully appended content from '{filename}'.")
                except Exception as append_exc:
                     # Log if appending a specific document fails
                     logging.error(f"Failed to append document '{filename}': {append_exc}", exc_info=True)
                     # Option: Stop merging entirely on failure: return None
                     # Option: Skip this file and continue (current behavior)
                     pass # Continue with the next file

        # Save the final composed document to a new memory stream
        merged_stream = io.BytesIO()
        composer.save(merged_stream)
        merged_stream.seek(0) # Reset stream position for reading
        # Log success, maybe include number of source docs merged
        logging.info(f"Successfully merged content from {len(doc_streams_data)} source documents into final stream.")
        return merged_stream # Return the final merged document stream

    except Exception as e:
        # Catch errors during initialization, saving, or unexpected issues in the loop
        logging.error(f"Error during the overall merge process: {e}", exc_info=True)
        return None # Return None to indicate failure

# --- (End of backend.py) ---

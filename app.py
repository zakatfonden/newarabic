# app.py (Using Document AI Extraction and Gemini Processing - RTL Focus)

import streamlit as st
import backend # Assumes backend.py is in the same directory
import os
from io import BytesIO
import logging

# Configure basic logging if needed
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="DocAI Extractor + Gemini Processor", # Keeping title general
    page_icon="📄✨",
    layout="wide"
)

# --- Initialize Session State ---
default_state = {
    'merged_doc_buffer': None,
    'files_processed_count': 0,
    'processing_complete': False,
    'processing_started': False,
    'ordered_files': [], # List for PDF/Image UploadedFile objects
    # --- DocAI config ---
    'docai_project_id': '',
    'docai_location': '',
    'docai_processor_id': '',
    # --- Gemini API Key (kept in session state implicitly via widget) ---
}
for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Helper Functions (Unchanged) ---
def reset_processing_state():
    st.session_state.merged_doc_buffer = None
    st.session_state.files_processed_count = 0
    st.session_state.processing_complete = False
    st.session_state.processing_started = False

def move_file(index, direction):
    files = st.session_state.ordered_files
    if not (0 <= index < len(files)): return
    new_index = index + direction
    if not (0 <= new_index < len(files)): return
    files[index], files[new_index] = files[new_index], files[index]
    st.session_state.ordered_files = files
    reset_processing_state()

def remove_file(index):
    files = st.session_state.ordered_files
    if 0 <= index < len(files):
        removed_file = files.pop(index)
        st.toast(f"Removed '{removed_file.name}'.")
        st.session_state.ordered_files = files
        reset_processing_state()
    else:
        st.warning(f"Could not remove file at index {index} (already removed or invalid?).")

def handle_uploads():
    """Adds newly uploaded PDF/Image files to the ordered list."""
    uploader_key = "docai_uploader" # Use a distinct key
    if uploader_key in st.session_state and st.session_state[uploader_key]:
        current_filenames = {f.name for f in st.session_state.ordered_files}
        new_files_added_count = 0
        skipped_count = 0
        # --- Define allowed types for Document AI ---
        allowed_types = ["application/pdf", "image/jpeg", "image/png", "image/tiff", "image/gif", "image/bmp", "image/webp"]
        allowed_ext = [".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".gif", ".bmp", ".webp"]

        for uploaded_file in st.session_state[uploader_key]:
            # Check both MIME type and extension for robustness
            file_allowed = False
            file_type = getattr(uploaded_file, 'type', 'unknown') # Get type safely
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()

            if file_type in allowed_types:
                file_allowed = True
            elif file_ext in allowed_ext:
                 file_allowed = True
                 logging.warning(f"File '{uploaded_file.name}' type '{file_type}' not in explicit list, but extension '{file_ext}' is allowed.")
            else:
                 logging.warning(f"Skipping '{uploaded_file.name}': Type '{file_type}' and extension '{file_ext}' not supported.")


            if file_allowed:
                if uploaded_file.name not in current_filenames:
                    st.session_state.ordered_files.append(uploaded_file)
                    current_filenames.add(uploaded_file.name)
                    new_files_added_count += 1
            else:
                skipped_count += 1

        if new_files_added_count > 0:
            st.toast(f"Added {new_files_added_count} new file(s) to the list.")
            reset_processing_state()
        if skipped_count > 0:
             st.warning(f"Skipped {skipped_count} file(s) due to unsupported type. Allowed types: PDF, JPG, PNG, TIFF, GIF, BMP, WEBP.", icon="⚠️")
        # Clear uploader widget state after processing uploads
        st.session_state[uploader_key] = [] # Important to prevent re-adding on script rerun

def clear_all_files_callback():
    st.session_state.ordered_files = []
    uploader_key = "docai_uploader"
    if uploader_key in st.session_state:
        st.session_state[uploader_key] = []
    reset_processing_state()
    st.toast("Removed all files from the list.")


# --- Page Title ---
st.title("📄✨ Document AI Extractor + Gemini Processor")
st.markdown("Upload PDF/Image files (Arabic/Urdu focus), extract text using Document AI, process the text using Gemini with custom rules (RTL formatting applied), and download the merged Word document.") # Updated description slightly

# --- Sidebar ---
st.sidebar.header("⚙️ Configuration")

# --- Document AI Configuration (Unchanged) ---
st.sidebar.subheader("Document AI Settings")
st.session_state.docai_project_id = st.sidebar.text_input(
    "Google Cloud Project ID",
    value=st.session_state.docai_project_id or st.secrets.get("DOCAI_PROJECT_ID", ""),
    help="Your Google Cloud Project ID."
)
st.session_state.docai_location = st.sidebar.text_input(
    "Processor Location",
    value=st.session_state.docai_location or st.secrets.get("DOCAI_LOCATION", "us"), # Default to 'us'
    help="The region of your Document AI processor (e.g., 'us', 'eu')."
)
st.session_state.docai_processor_id = st.sidebar.text_input(
    "Processor ID",
    value=st.session_state.docai_processor_id or st.secrets.get("DOCAI_PROCESSOR_ID", ""),
    help="The ID of your Document AI processor (e.g., the Document OCR processor ID)."
)
docai_configured = all([st.session_state.docai_project_id, st.session_state.docai_location, st.session_state.docai_processor_id])
if docai_configured:
    st.sidebar.success("Document AI configured.", icon="✅")
else:
    st.sidebar.warning("Document AI configuration missing.", icon="⚠️")
# ---

st.sidebar.markdown("---")

# --- Gemini Configuration (Unchanged except default rules) ---
st.sidebar.subheader("Gemini Settings")
# Gemini API Key Input
api_key_from_secrets = st.secrets.get("GEMINI_API_KEY", "")
api_key = st.sidebar.text_input(
    "Enter your Google Gemini API Key", type="password",
    help="Required for text processing via Gemini. Get your key from Google AI Studio.",
    value=api_key_from_secrets or "",
    key="gemini_api_key_input" # Unique key
)
gemini_configured = False
if api_key_from_secrets and api_key == api_key_from_secrets:
    st.sidebar.success("Gemini Key loaded from Secrets.", icon="🔑")
    gemini_configured = True
elif not api_key_from_secrets and not api_key:
    st.sidebar.warning("Gemini Key not found or entered.", icon="❓")
elif api_key and not api_key_from_secrets:
    st.sidebar.info("Using manually entered Gemini Key.", icon="⌨️")
    gemini_configured = True
elif api_key and api_key_from_secrets and api_key != api_key_from_secrets:
    st.sidebar.info("Using manually entered Gemini Key (overrides secret).", icon="⌨️")
    gemini_configured = True

# Model Selection
model_options = {
    "Gemini 1.5 Flash (Fastest, Cost-Effective)": "gemini-1.5-flash-latest",
    "Gemini 1.5 Pro (Advanced, Slower, Higher Cost)": "gemini-1.5-pro-latest",
}
selected_model_display_name = st.sidebar.selectbox(
    "Choose the Gemini model for processing:",
    options=list(model_options.keys()), index=0, key="gemini_model_select",
    help="Select the AI model for processing the extracted text."
)
selected_model_id = model_options[selected_model_display_name]
st.sidebar.caption(f"Selected model ID: `{selected_model_id}`")

# Processing Rules (CHANGED Default Rules)
st.sidebar.markdown("---")
st.sidebar.header("📜 Gemini Processing Rules")
# --- NEW DEFAULT RULES ---
default_rules = """Structure the text into paragraphs.
Delete headers (typically signifies the name of a chapter).
Delete footnotes.
Inspect the pdf. There are two lines.: a top line and a bottom line.
The top line is at the top of the page. Within the first 5cm of the page. Remember all the text above the top line.
Remember all the text below the bottom line.
Compare with the extracted text.
Now delete all text that can be identified as above the top line.
Then delete all the text that can be identified as below the bottom line. Everything below the bottom line is footnotes, so it must be deleted.
Return ONLY the processed text, without any introductory phrases or explanations."""
# --- END NEW DEFAULT RULES ---
rules_prompt = st.sidebar.text_area(
    "Enter the processing instructions for Gemini:",
    value=default_rules, height=250, # Increased height slightly
    help="Instructions for how Gemini should process the text extracted by Document AI. (Note: Spatial rules like '5cm' might be difficult for the AI to follow precisely based only on text)."
)
# --- End Gemini Configuration ---


# --- Main Area ---

st.header("📁 Manage Files for Extraction & Processing")

# --- File Uploader (Unchanged) ---
st.file_uploader(
    "Choose PDF or Image files:",
    type=["pdf", "png", "jpg", "jpeg", "tiff", "tif", "gif", "bmp", "webp"],
    accept_multiple_files=True,
    key="docai_uploader",
    on_change=handle_uploads,
    label_visibility="visible"
)
# ---

st.markdown("---")

# --- TOP: Buttons Area & Progress Indicators (Unchanged) ---
st.subheader("🚀 Actions & Progress (Top)")
col_b1_top, col_b2_top = st.columns([3, 2])

with col_b1_top:
    process_button_enabled = (
        not st.session_state.processing_started and
        st.session_state.ordered_files and
        docai_configured and
        gemini_configured
    )
    process_button_top_clicked = st.button(
        "✨ Process Files & Merge (Top)",
        key="process_button_top_combined",
        use_container_width=True, type="primary",
        disabled=not process_button_enabled
    )

with col_b2_top:
    if st.session_state.merged_doc_buffer and not st.session_state.processing_started:
        st.download_button(
            label=f"📥 Download Processed Text ({st.session_state.files_processed_count}) (.docx)",
            data=st.session_state.merged_doc_buffer,
            file_name="merged_processed_text.docx", # Keeping filename generic
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            key="download_merged_button_top_combined",
            use_container_width=True
        )
    elif st.session_state.processing_started:
        st.info("Processing in progress...", icon="⏳")
    elif not docai_configured and st.session_state.ordered_files:
         st.warning("Configure Document AI in sidebar.", icon="⚙️")
    elif not gemini_configured and st.session_state.ordered_files:
         st.warning("Configure Gemini API Key in sidebar.", icon="🔑")
    elif not st.session_state.ordered_files:
         st.markdown("*(Upload files to enable processing)*")
    else: # Ready but not processed yet
        st.markdown("*(Download button appears here after processing)*")


# Placeholders for top progress indicators
progress_bar_placeholder_top = st.empty()
status_text_placeholder_top = st.empty()

st.markdown("---") # Separator before file list

# --- Interactive File List (Unchanged) ---
st.subheader(f"Files in Processing Order ({len(st.session_state.ordered_files)}):")

if not st.session_state.ordered_files:
    st.info("Use the uploader above to add PDF or Image files.")
else:
    col_h1, col_h2, col_h3, col_h4, col_h5 = st.columns([0.5, 5, 1, 1, 1])
    with col_h1: st.markdown("**#**")
    with col_h2: st.markdown("**Filename**")
    with col_h3: st.markdown("**Up**")
    with col_h4: st.markdown("**Down**")
    with col_h5: st.markdown("**Remove**")

    for i, file in enumerate(st.session_state.ordered_files):
        col1, col2, col3, col4, col5 = st.columns([0.5, 5, 1, 1, 1])
        with col1: st.write(f"{i+1}")
        with col2: st.write(file.name)
        with col3: st.button("⬆️", key=f"up_combined_{i}", on_click=move_file, args=(i, -1), disabled=(i == 0), help="Move Up")
        with col4: st.button("⬇️", key=f"down_combined_{i}", on_click=move_file, args=(i, 1), disabled=(i == len(st.session_state.ordered_files) - 1), help="Move Down")
        with col5: st.button("❌", key=f"del_combined_{i}", on_click=remove_file, args=(i,), help="Remove")

    st.button("🗑️ Remove All Files",
              key="remove_all_button_combined",
              on_click=clear_all_files_callback,
              help="Click to remove all files from the list.",
              type="secondary")


st.markdown("---") # Separator after file list

# --- BOTTOM: Buttons Area & Progress Indicators (Unchanged) ---
st.subheader("🚀 Actions & Progress (Bottom)")
col_b1_bottom, col_b2_bottom = st.columns([3, 2])

with col_b1_bottom:
    process_button_bottom_clicked = st.button(
        "✨ Process Files & Merge (Bottom)",
        key="process_button_bottom_combined",
        use_container_width=True, type="primary",
        disabled=not process_button_enabled
    )

with col_b2_bottom:
    if st.session_state.merged_doc_buffer and not st.session_state.processing_started:
        st.download_button(
            label=f"📥 Download Processed Text ({st.session_state.files_processed_count}) (.docx)",
            data=st.session_state.merged_doc_buffer,
            file_name="merged_processed_text.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            key="download_merged_button_bottom_combined",
            use_container_width=True
        )
    elif st.session_state.processing_started:
        st.info("Processing in progress...", icon="⏳")
    elif not docai_configured and st.session_state.ordered_files:
         st.warning("Configure Document AI in sidebar.", icon="⚙️")
    elif not gemini_configured and st.session_state.ordered_files:
         st.warning("Configure Gemini API Key in sidebar.", icon="🔑")
    elif not st.session_state.ordered_files:
         st.markdown("*(Upload files to enable processing)*")
    else: # Ready but not processed yet
        st.markdown("*(Download button appears here after processing)*")

# Placeholders for bottom progress indicators
progress_bar_placeholder_bottom = st.empty()
status_text_placeholder_bottom = st.empty()

# --- Container for Individual File Results (Unchanged) ---
results_container = st.container()


# --- == Processing Logic (Unchanged structure, uses updated backend function) == ---
if process_button_top_clicked or process_button_bottom_clicked:
    reset_processing_state()
    st.session_state.processing_started = True

    # --- Get Config ---
    project_id = st.session_state.docai_project_id
    location = st.session_state.docai_location
    processor_id = st.session_state.docai_processor_id
    docai_configured = all([project_id, location, processor_id])
    gemini_api_key = api_key
    gemini_configured = bool(gemini_api_key)
    current_rules = rules_prompt

    # --- Pre-flight Checks ---
    if not st.session_state.ordered_files:
        st.warning("⚠️ No files in the list to process.")
        st.session_state.processing_started = False
    elif not docai_configured:
        st.error("❌ Document AI is not configured in the sidebar.")
        st.session_state.processing_started = False
    elif not gemini_configured:
        st.error("❌ Gemini API Key is missing or not configured in the sidebar.")
        st.session_state.processing_started = False
    elif not current_rules.strip():
        st.warning("⚠️ The 'Gemini Processing Rules' field is empty. Using default rules.")
        current_rules = default_rules # Use updated default rules
    elif not selected_model_id:
        st.error("❌ No Gemini model selected in the sidebar.")
        st.session_state.processing_started = False
    # --- End Pre-flight Checks ---

    if st.session_state.processing_started:
        processed_doc_streams_for_merge = []
        files_successfully_processed = 0
        total_files = len(st.session_state.ordered_files)

        progress_bar_top = progress_bar_placeholder_top.progress(0, text="Starting processing...")
        progress_bar_bottom = progress_bar_placeholder_bottom.progress(0, text="Starting processing...")

        for i, file_to_process in enumerate(st.session_state.ordered_files):
            original_filename = file_to_process.name
            current_file_status = f"'{original_filename}' ({i + 1}/{total_files})"
            progress_text = f"Processing {current_file_status}..."

            progress_value = i / total_files
            progress_bar_top.progress(progress_value, text=progress_text)
            progress_bar_bottom.progress(progress_value, text=progress_text)
            status_text_placeholder_top.info(f"🔄 Starting {current_file_status}")
            status_text_placeholder_bottom.info(f"🔄 Starting {current_file_status}")

            with results_container:
                st.markdown(f"--- \n**Processing: {original_filename}**")

            raw_text = None
            processed_text = ""
            extraction_error = False
            gemini_error_occurred = False
            word_creation_error_occurred = False

            # 1. Extract Text with Document AI
            status_text_placeholder_top.info(f"📄 Extracting text via Document AI for {current_file_status}...")
            status_text_placeholder_bottom.info(f"📄 Extracting text via Document AI for {current_file_status}...")
            try:
                raw_text_result = backend.extract_text_with_docai(
                    file_to_process, project_id, location, processor_id
                )
                if isinstance(raw_text_result, str) and raw_text_result.startswith("Error:"):
                    with results_container: st.error(f"❌ Document AI Error for '{original_filename}': {raw_text_result}")
                    extraction_error = True
                    raw_text = None
                elif not raw_text_result or not raw_text_result.strip():
                    with results_container: st.warning(f"⚠️ No text extracted by Document AI from '{original_filename}'.")
                    raw_text = ""
                else:
                    raw_text = raw_text_result
                    with results_container: st.info(f"📄 Text extracted successfully for '{original_filename}'.")
            except Exception as ext_exc:
                with results_container: st.error(f"❌ Unexpected error during Document AI extraction for '{original_filename}': {ext_exc}")
                extraction_error = True
                raw_text = None

            # 2. Process with Gemini
            if not extraction_error and raw_text is not None:
                if raw_text.strip():
                    status_text_placeholder_top.info(f"✨ Processing text from {current_file_status} via Gemini ({selected_model_display_name})...")
                    status_text_placeholder_bottom.info(f"✨ Processing text from {current_file_status} via Gemini ({selected_model_display_name})...")
                    try:
                        processed_text_result = backend.process_text_with_gemini(
                            gemini_api_key, raw_text, current_rules, selected_model_id
                        )
                        if isinstance(processed_text_result, str) and processed_text_result.startswith("Error:"):
                            with results_container: st.error(f"❌ Gemini processing error for '{original_filename}': {processed_text_result}")
                            gemini_error_occurred = True
                            processed_text = ""
                        elif processed_text_result is None:
                             with results_container: st.error(f"❌ Gemini processing error for '{original_filename}': Received None response.")
                             gemini_error_occurred = True
                             processed_text = ""
                        else:
                            processed_text = processed_text_result
                            with results_container: st.success(f"✨ Text processed successfully by Gemini for '{original_filename}'.")
                    except Exception as gem_exc:
                        with results_container: st.error(f"❌ Unexpected error during Gemini processing for '{original_filename}': {gem_exc}")
                        gemini_error_occurred = True
                        processed_text = ""
                else:
                    logging.info(f"Skipping Gemini processing for '{original_filename}' as extracted text was empty.")
                    processed_text = ""
                    with results_container: st.info(f"✨ Skipping Gemini processing for '{original_filename}' (no extracted text).")

            # 3. Create Individual Word Document (using updated backend function for RTL)
            status_text_placeholder_top.info(f"📝 Creating intermediate Word document for {current_file_status}...")
            status_text_placeholder_bottom.info(f"📝 Creating intermediate Word document for {current_file_status}...")
            try:
                # Backend function now handles RTL formatting
                word_doc_stream = backend.create_word_doc_from_processed_text(
                    processed_text, original_filename, extraction_error, gemini_error_occurred
                )
                if word_doc_stream:
                    processed_doc_streams_for_merge.append((original_filename, word_doc_stream))
                    files_successfully_processed += 1
                    with results_container:
                        success_msg = f"✅ Created intermediate document for '{original_filename}'."
                        if extraction_error: success_msg += " (Note: placeholder used due to Document AI extraction error)"
                        elif gemini_error_occurred: success_msg += " (Note: placeholder or original text used due to Gemini processing error)"
                        elif not processed_text.strip(): success_msg += " (Note: document may be empty or contain placeholder as no text was extracted or processed)"
                        st.success(success_msg)
                else:
                    word_creation_error_occurred = True
                    with results_container: st.error(f"❌ Failed to create intermediate Word file for '{original_filename}'.")
            except Exception as doc_exc:
                word_creation_error_occurred = True
                with results_container: st.error(f"❌ Error during intermediate Word file creation for '{original_filename}': {doc_exc}")

            # Update overall progress
            status_msg_suffix = ""
            if extraction_error or gemini_error_occurred or word_creation_error_occurred: status_msg_suffix = " with issues."
            final_progress_value = (i + 1) / total_files
            final_progress_text = f"Processed {current_file_status}{status_msg_suffix}"
            progress_bar_top.progress(final_progress_value, text=final_progress_text)
            progress_bar_bottom.progress(final_progress_value, text=final_progress_text)

        # --- End of file loop ---

        progress_bar_placeholder_top.empty()
        progress_bar_placeholder_bottom.empty()
        status_text_placeholder_top.empty()
        status_text_placeholder_bottom.empty()

        # 4. Merge Documents
        final_status_message = ""
        rerun_needed = False
        with results_container:
            st.markdown("---")
            if files_successfully_processed > 0:
                st.info(f"💾 Merging {files_successfully_processed} processed Word document(s)...")
                try:
                    merged_buffer = backend.merge_word_documents(processed_doc_streams_for_merge)
                    if merged_buffer:
                        st.session_state.merged_doc_buffer = merged_buffer
                        st.session_state.files_processed_count = files_successfully_processed
                        final_status_message = f"✅ Processing complete! Merged document created from {files_successfully_processed} source file(s)."
                        if files_successfully_processed < total_files: final_status_message += f" ({total_files - files_successfully_processed} file(s) had issues during processing)."
                        st.success(final_status_message)
                        rerun_needed = True
                    else:
                        final_status_message = "❌ Failed to merge Word documents."
                        st.error(final_status_message)
                except Exception as merge_exc:
                    final_status_message = f"❌ Error during document merging: {merge_exc}"
                    logging.error(f"Error during merge_word_documents call: {merge_exc}", exc_info=True)
                    st.error(final_status_message)
            elif total_files > 0:
                 final_status_message = "⚠️ No documents were successfully processed to merge."
                 st.warning(final_status_message)

        st.session_state.processing_complete = True
        st.session_state.processing_started = False
        if rerun_needed: st.rerun()

    else: # Initial checks failed
        st.session_state.processing_started = False

# --- Fallback info message ---
if not st.session_state.ordered_files and not st.session_state.processing_started and not st.session_state.processing_complete:
    st.info("Upload PDF or Image files using the button above.")

# --- Footer ---
st.markdown("---")
st.markdown("Developed with Streamlit, Google Document AI, and Google Gemini.")

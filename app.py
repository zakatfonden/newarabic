
# app.py (Using Document AI Extraction, Translation, and Merging)

import streamlit as st
import backend # Assumes backend_py_docai_merge.py is in the same directory
import os
from io import BytesIO
import logging

# Configure basic logging if needed
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="DocAI Translator", # New title
    page_icon="📄➡️🇦🇪", # New icon
    layout="wide"
)

# --- Initialize Session State ---
default_state = {
    'merged_doc_buffer': None,
    'files_processed_count': 0,
    'processing_complete': False,
    'processing_started': False,
    'ordered_files': [], # List for PDF/Image UploadedFile objects
    # --- NEW: Add state for DocAI config ---
    'docai_project_id': '',
    'docai_location': '',
    'docai_processor_id': '',
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
            if uploaded_file.type in allowed_types:
                file_allowed = True
            else:
                 ext = os.path.splitext(uploaded_file.name)[1].lower()
                 if ext in allowed_ext:
                     file_allowed = True
                     logging.warning(f"File '{uploaded_file.name}' type '{uploaded_file.type}' not in explicit list, but extension '{ext}' is allowed.")
                 else:
                      logging.warning(f"Skipping '{uploaded_file.name}': Type '{uploaded_file.type}' and extension '{ext}' not supported.")


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
        # st.session_state[uploader_key] = [] # Optional: Clear uploader

def clear_all_files_callback():
    st.session_state.ordered_files = []
    uploader_key = "docai_uploader"
    if uploader_key in st.session_state:
        st.session_state[uploader_key] = []
    reset_processing_state()
    st.toast("Removed all files from the list.")


# --- Page Title ---
st.title("📄➡️🇦🇪 Document AI + Gemini Translator") # Changed
st.markdown("Upload PDF or Image files, extract text using Document AI, translate to Arabic using Gemini, and download the merged Word document.") # Changed

# --- Sidebar ---
st.sidebar.header("⚙️ Configuration")

# --- NEW: Document AI Configuration ---
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
# Check if DocAI config seems complete
docai_configured = all([st.session_state.docai_project_id, st.session_state.docai_location, st.session_state.docai_processor_id])
if docai_configured:
    st.sidebar.success("Document AI configured.", icon="✅")
else:
    st.sidebar.warning("Document AI configuration missing.", icon="⚠️")
# ---

st.sidebar.markdown("---")

# Gemini API Key Input (Unchanged)
st.sidebar.subheader("Gemini API Key")
api_key_from_secrets = st.secrets.get("GEMINI_API_KEY", "")
api_key = st.sidebar.text_input(
    "Enter your Google Gemini API Key", type="password",
    help="Required for translation. Get your key from Google AI Studio.", value=api_key_from_secrets or "",
    key="gemini_api_key_input" # Added key for uniqueness
)
if api_key_from_secrets and api_key == api_key_from_secrets: st.sidebar.success("Gemini Key loaded from Secrets.", icon="🔑")
elif not api_key_from_secrets and not api_key: st.sidebar.warning("Gemini Key not found or entered.", icon="❓")
elif api_key and not api_key_from_secrets: st.sidebar.info("Using manually entered Gemini Key.", icon="⌨️")
elif api_key and api_key_from_secrets and api_key != api_key_from_secrets: st.sidebar.info("Using manually entered Gemini Key (overrides secret).", icon="⌨️")

# Model Selection (Unchanged)
st.sidebar.markdown("---")
st.sidebar.header("🧠 AI Model for Translation")
model_options = {
    "Gemini 1.5 Flash (Fastest, Cost-Effective)": "gemini-1.5-flash-latest",
    "Gemini 1.5 Pro (Advanced, Slower, Higher Cost)": "gemini-1.5-pro-latest",
}
selected_model_display_name = st.sidebar.selectbox(
    "Choose the Gemini model for translation:",
    options=list(model_options.keys()), index=0, key="gemini_model_select",
    help="Select the AI model. Pro is better for nuanced translation."
)
selected_model_id = model_options[selected_model_display_name]
st.sidebar.caption(f"Selected model ID: `{selected_model_id}`")

# Translation Rules (Simplified for Translation Only)
st.sidebar.markdown("---")
st.sidebar.header("📜 Translation Rules")
default_rules = """
Translate the following text accurately into Modern Standard Arabic.
The input text might be in Urdu, Farsi, or English.
Preserve the meaning and intent of the original text.
Format the output as clean Arabic paragraphs suitable for a document.
Return ONLY the Arabic translation, without any introductory phrases, explanations, or markdown formatting.
"""
rules_prompt = st.sidebar.text_area(
    "Enter the translation instructions for Gemini:", value=default_rules, height=200,
    help="Instructions for how Gemini should translate the text extracted by Document AI."
)

# --- Main Area ---

st.header("📁 Manage Files for Translation")

# --- CHANGED: File Uploader for PDF/Images ---
st.file_uploader(
    "Choose PDF or Image files to translate:",
    type=["pdf", "png", "jpg", "jpeg", "tiff", "tif", "gif", "bmp", "webp"], # Allowed types
    accept_multiple_files=True,
    key="docai_uploader", # Use a distinct key
    on_change=handle_uploads,
    label_visibility="visible"
)
# ---

st.markdown("---")

# --- TOP: Buttons Area & Progress Indicators ---
st.subheader("🚀 Actions & Progress (Top)")
col_b1_top, col_b2_top = st.columns([3, 2])

with col_b1_top:
    process_button_top_clicked = st.button(
        "✨ Process Files & Merge (Top)", # Changed label slightly
        key="process_button_top_docai", # New key
        use_container_width=True, type="primary",
        # Disable if processing, no files, or DocAI not configured
        disabled=st.session_state.processing_started or not st.session_state.ordered_files or not docai_configured
    )

with col_b2_top:
    if st.session_state.merged_doc_buffer and not st.session_state.processing_started:
        st.download_button(
            label=f"📥 Download Merged ({st.session_state.files_processed_count}) Translations (.docx)",
            data=st.session_state.merged_doc_buffer,
            file_name="merged_docai_translations.docx", # New filename
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            key="download_merged_button_top_docai", # New key
            use_container_width=True
        )
    elif st.session_state.processing_started:
        st.info("Processing in progress...", icon="⏳")
    elif not docai_configured and st.session_state.ordered_files:
         st.warning("Configure Document AI in sidebar to enable processing.", icon="⚙️")
    else:
        st.markdown("*(Download button appears here after processing)*")


# Placeholders for top progress indicators
progress_bar_placeholder_top = st.empty()
status_text_placeholder_top = st.empty()

st.markdown("---") # Separator before file list

# --- Interactive File List (Displays PDF/Image files) ---
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

    # Use unique keys based on index `i`
    for i, file in enumerate(st.session_state.ordered_files):
        col1, col2, col3, col4, col5 = st.columns([0.5, 5, 1, 1, 1])
        with col1: st.write(f"{i+1}")
        with col2: st.write(file.name)
        with col3: st.button("⬆️", key=f"up_docai_{i}", on_click=move_file, args=(i, -1), disabled=(i == 0), help="Move Up")
        with col4: st.button("⬇️", key=f"down_docai_{i}", on_click=move_file, args=(i, 1), disabled=(i == len(st.session_state.ordered_files) - 1), help="Move Down")
        with col5: st.button("❌", key=f"del_docai_{i}", on_click=remove_file, args=(i,), help="Remove")

    st.button("🗑️ Remove All Files",
              key="remove_all_button_docai", # New key
              on_click=clear_all_files_callback,
              help="Click to remove all files from the list.",
              type="secondary")


st.markdown("---") # Separator after file list

# --- BOTTOM: Buttons Area & Progress Indicators ---
st.subheader("🚀 Actions & Progress (Bottom)")
col_b1_bottom, col_b2_bottom = st.columns([3, 2])

with col_b1_bottom:
    process_button_bottom_clicked = st.button(
        "✨ Process Files & Merge (Bottom)", # Changed label slightly
        key="process_button_bottom_docai", # New key
        use_container_width=True, type="primary",
        disabled=st.session_state.processing_started or not st.session_state.ordered_files or not docai_configured
    )

with col_b2_bottom:
    if st.session_state.merged_doc_buffer and not st.session_state.processing_started:
        st.download_button(
            label=f"📥 Download Merged ({st.session_state.files_processed_count}) Translations (.docx)",
            data=st.session_state.merged_doc_buffer,
            file_name="merged_docai_translations.docx", # New filename
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            key="download_merged_button_bottom_docai", # New key
            use_container_width=True
        )
    elif st.session_state.processing_started:
        st.info("Processing in progress...", icon="⏳")
    elif not docai_configured and st.session_state.ordered_files:
         st.warning("Configure Document AI in sidebar to enable processing.", icon="⚙️")
    else:
        st.markdown("*(Download button appears here after processing)*")

# Placeholders for bottom progress indicators
progress_bar_placeholder_bottom = st.empty()
status_text_placeholder_bottom = st.empty()

# --- Container for Individual File Results ---
results_container = st.container()


# --- == Processing Logic (DocAI Extract -> Translate -> Create -> Merge) == ---
if process_button_top_clicked or process_button_bottom_clicked:
    reset_processing_state()
    st.session_state.processing_started = True

    # --- Get DocAI config from session state ---
    project_id = st.session_state.docai_project_id
    location = st.session_state.docai_location
    processor_id = st.session_state.docai_processor_id
    docai_configured = all([project_id, location, processor_id]) # Re-check here

    # Re-check conditions including DocAI config
    if not st.session_state.ordered_files:
        st.warning("⚠️ No files in the list to process.")
        st.session_state.processing_started = False
    elif not docai_configured:
         st.error("❌ Document AI is not configured in the sidebar. Please provide Project ID, Location, and Processor ID.")
         st.session_state.processing_started = False
    elif not api_key:
        st.error("❌ Please enter or configure your Gemini API Key in the sidebar.")
        st.session_state.processing_started = False
    elif not rules_prompt.strip():
        st.warning("⚠️ The 'Translation Rules' field is empty. Using default rules.")
        current_rules = default_rules
    elif not selected_model_id:
        st.error("❌ No Gemini model selected in the sidebar.")
        st.session_state.processing_started = False
    else:
        current_rules = rules_prompt

    # Proceed only if checks passed
    if st.session_state.ordered_files and docai_configured and api_key and st.session_state.processing_started and selected_model_id:

        processed_doc_streams = []
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
            translated_text = ""
            extraction_error = False
            gemini_error_occurred = False
            word_creation_error_occurred = False

            # 1. Extract Text with Document AI
            status_text_placeholder_top.info(f"📄 Extracting text via Document AI for {current_file_status}...")
            status_text_placeholder_bottom.info(f"📄 Extracting text via Document AI for {current_file_status}...")
            try:
                # Use the new backend function for Document AI
                raw_text = backend.extract_text_with_docai(
                    file_to_process, project_id, location, processor_id
                )
                if isinstance(raw_text, str) and raw_text.startswith("Error:"):
                    with results_container: st.error(f"❌ Document AI Error for '{original_filename}': {raw_text}")
                    extraction_error = True
                elif not raw_text or not raw_text.strip():
                    with results_container: st.warning(f"⚠️ No text extracted by Document AI from '{original_filename}'.")
                    # Keep raw_text empty, translation will be skipped
            except Exception as ext_exc:
                with results_container: st.error(f"❌ Unexpected error during Document AI extraction for '{original_filename}': {ext_exc}")
                extraction_error = True

            # 2. Translate with Gemini (if text extracted)
            if not extraction_error:
                if raw_text and raw_text.strip():
                    status_text_placeholder_top.info(f"🤖 Translating text from {current_file_status} via Gemini ({selected_model_display_name})...")
                    status_text_placeholder_bottom.info(f"🤖 Translating text from {current_file_status} via Gemini ({selected_model_display_name})...")
                    try:
                        processed_text_result = backend.process_text_with_gemini(
                            api_key, raw_text, current_rules, selected_model_id
                        )
                        if processed_text_result is None or (isinstance(processed_text_result, str) and processed_text_result.startswith("Error:")):
                            with results_container: st.error(f"❌ Gemini translation error for '{original_filename}': {processed_text_result or 'Unknown API error'}")
                            gemini_error_occurred = True
                            translated_text = ""
                        else:
                            translated_text = processed_text_result
                    except Exception as gem_exc:
                        with results_container: st.error(f"❌ Unexpected error during Gemini translation for '{original_filename}': {gem_exc}")
                        gemini_error_occurred = True
                        translated_text = ""
                else:
                    logging.info(f"Skipping Gemini translation for '{original_filename}' as extracted text was empty.")
                    translated_text = ""

                # 3. Create Individual Word Document with Arabic Translation
                status_text_placeholder_top.info(f"📝 Creating intermediate Word document for {current_file_status}...")
                status_text_placeholder_bottom.info(f"📝 Creating intermediate Word document for {current_file_status}...")
                try:
                    word_doc_stream = backend.create_arabic_word_doc_from_text(
                        translated_text, original_filename
                    )
                    if word_doc_stream:
                        processed_doc_streams.append((original_filename, word_doc_stream))
                        files_successfully_processed += 1
                        with results_container:
                            success_msg = f"✅ Created intermediate document for '{original_filename}'."
                            if not translated_text or not translated_text.strip():
                                # Add appropriate note if placeholder was used
                                if gemini_error_occurred: success_msg += " (Note: placeholder used due to translation error)"
                                elif raw_text is None or not raw_text.strip() and not extraction_error: success_msg += " (Note: placeholder used as no text was extracted by DocAI)"
                                else: success_msg += " (Note: placeholder used as translation was empty)"
                            st.success(success_msg)
                    else:
                        word_creation_error_occurred = True
                        with results_container: st.error(f"❌ Failed to create intermediate Word file for '{original_filename}'.")
                except Exception as doc_exc:
                    word_creation_error_occurred = True
                    with results_container: st.error(f"❌ Error during intermediate Word file creation for '{original_filename}': {doc_exc}")

            else: # Extraction failed critically
                 with results_container: st.warning(f"⏩ Skipping translation and document creation for '{original_filename}' due to extraction errors.")

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
                st.info(f"💾 Merging {files_successfully_processed} translated Word document(s)...")
                try:
                    merged_buffer = backend.merge_word_documents(processed_doc_streams)
                    if merged_buffer:
                        st.session_state.merged_doc_buffer = merged_buffer
                        st.session_state.files_processed_count = files_successfully_processed
                        final_status_message = f"✅ Processing complete! Merged document created from {files_successfully_processed} source file(s)."
                        if files_successfully_processed < total_files:
                             final_status_message += f" ({total_files - files_successfully_processed} file(s) had issues)."
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
            # Else: No files uploaded

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
st.markdown("Developed with Streamlit, Google Document AI, and Google Gemini.") # Updated footer

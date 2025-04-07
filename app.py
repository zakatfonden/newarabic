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
    'docai_location': '', # Default location will be set in the widget if not in secrets
    'docai_processor_id': '',
    # --- Gemini API Key (kept in session state implicitly via widget) ---
}
for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Helper Functions (Unchanged) ---
def reset_processing_state():
    """Resets flags and buffers related to processing results."""
    st.session_state.merged_doc_buffer = None
    st.session_state.files_processed_count = 0
    st.session_state.processing_complete = False
    st.session_state.processing_started = False

def move_file(index, direction):
    """Moves a file up or down in the ordered list."""
    files = st.session_state.ordered_files
    if not (0 <= index < len(files)): return # Index out of bounds
    new_index = index + direction
    if not (0 <= new_index < len(files)): return # New index out of bounds
    # Swap elements
    files[index], files[new_index] = files[new_index], files[index]
    st.session_state.ordered_files = files
    reset_processing_state() # Reset results if order changes

def remove_file(index):
    """Removes a file from the ordered list by its index."""
    files = st.session_state.ordered_files
    if 0 <= index < len(files):
        removed_file = files.pop(index)
        st.toast(f"Removed '{removed_file.name}'.")
        st.session_state.ordered_files = files
        reset_processing_state() # Reset results if file removed
    else:
        st.warning(f"Could not remove file at index {index} (already removed or invalid?).")

def handle_uploads():
    """
    Callback function triggered when files are uploaded via st.file_uploader.
    Adds new, valid files to the session state's ordered list, avoiding duplicates.
    *** Does NOT attempt to clear the uploader widget state directly. ***
    """
    uploader_key = "docai_uploader" # Key for the file uploader widget
    # Check if the uploader key exists in session state and has files
    if uploader_key in st.session_state and st.session_state[uploader_key]:
        current_filenames = {f.name for f in st.session_state.ordered_files}
        new_files_added_count = 0
        skipped_count = 0
        # --- Define allowed types for Document AI ---
        allowed_types = ["application/pdf", "image/jpeg", "image/png", "image/tiff", "image/gif", "image/bmp", "image/webp"]
        allowed_ext = [".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".gif", ".bmp", ".webp"]

        # Iterate through the files currently held by the uploader widget's state
        for uploaded_file in st.session_state[uploader_key]:
            # Check both MIME type and extension for robustness
            file_allowed = False
            file_type = getattr(uploaded_file, 'type', 'unknown') # Get type safely
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()

            if file_type in allowed_types:
                file_allowed = True
            elif file_ext in allowed_ext:
                 file_allowed = True
                 # Log if relying on extension because MIME type might be generic
                 logging.warning(f"File '{uploaded_file.name}' type '{file_type}' not in explicit list, but extension '{file_ext}' is allowed.")
            else:
                 # Log skipped files
                 logging.warning(f"Skipping '{uploaded_file.name}': Type '{file_type}' and extension '{file_ext}' not supported.")


            if file_allowed:
                # Add file only if its name is not already in the list
                if uploaded_file.name not in current_filenames:
                    st.session_state.ordered_files.append(uploaded_file)
                    current_filenames.add(uploaded_file.name) # Update set of current names
                    new_files_added_count += 1
                # else: # Optional: Inform user about duplicates
                #     st.info(f"File '{uploaded_file.name}' is already in the list.")
            else:
                skipped_count += 1

        # Provide feedback to the user
        if new_files_added_count > 0:
            st.toast(f"Added {new_files_added_count} new file(s) to the list.")
            reset_processing_state() # Reset if new files were added
        if skipped_count > 0:
             st.warning(f"Skipped {skipped_count} file(s) due to unsupported type. Allowed types: PDF, JPG, PNG, TIFF, GIF, BMP, WEBP.", icon="⚠️")

        # --- REMOVED THIS LINE to fix the error ---
        # st.session_state[uploader_key] = []
        # The file uploader widget will manage clearing its own state after the callback.

def clear_all_files_callback():
    """Removes all files from the list and resets state."""
    st.session_state.ordered_files = [] # Clear our managed list of files
    uploader_key = "docai_uploader"
    # --- REMOVED THIS LINE to fix the error ---
    # if uploader_key in st.session_state:
    #     st.session_state[uploader_key] = [] # Don't try to clear the widget directly
    reset_processing_state()
    st.toast("Removed all files from the list.")


# --- Page Title and Description ---
st.title("📄✨ Document AI Extractor + Gemini Processor")
st.markdown("Upload PDF/Image files (Arabic/Urdu focus), extract text using Document AI, process the text using Gemini with custom rules (RTL formatting applied), and download the merged Word document.")

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Configuration")

# --- Document AI Configuration Section ---
st.sidebar.subheader("Document AI Settings")
# Input for Google Cloud Project ID
st.session_state.docai_project_id = st.sidebar.text_input(
    "Google Cloud Project ID",
    value=st.session_state.docai_project_id or st.secrets.get("DOCAI_PROJECT_ID", ""),
    help="Your Google Cloud Project ID where the Document AI processor resides."
)
# Input for Processor Location (Region)
st.session_state.docai_location = st.sidebar.text_input(
    "Processor Location",
    # Set default value to "eu" if not found in session state or secrets
    value=st.session_state.docai_location or st.secrets.get("DOCAI_LOCATION", "eu"), # <- CHANGED DEFAULT HERE
    help="The region of your Document AI processor (e.g., 'us', 'eu')."
)
# Input for Processor ID
st.session_state.docai_processor_id = st.sidebar.text_input(
    "Processor ID",
    value=st.session_state.docai_processor_id or st.secrets.get("DOCAI_PROCESSOR_ID", ""),
    help="The specific ID of your Document AI processor (e.g., the Document OCR processor ID)."
)
# Check if all Document AI settings are provided
docai_configured = all([st.session_state.docai_project_id, st.session_state.docai_location, st.session_state.docai_processor_id])
# Display status icon based on configuration completeness
if docai_configured:
    st.sidebar.success("Document AI configured.", icon="✅")
else:
    st.sidebar.warning("Document AI configuration missing.", icon="⚠️")
# --- End Document AI Configuration ---

st.sidebar.markdown("---") # Visual separator

# --- Gemini Configuration Section ---
st.sidebar.subheader("Gemini Settings")
# Input for Gemini API Key (masked as password)
# Check for key in secrets first, then allow manual input
api_key_from_secrets = st.secrets.get("GEMINI_API_KEY", "")
api_key = st.sidebar.text_input(
    "Enter your Google Gemini API Key", type="password",
    help="Required for text processing via Gemini. Get your key from Google AI Studio.",
    value=api_key_from_secrets or "", # Use secret if available, otherwise empty
    key="gemini_api_key_input" # Assign a unique key to the widget
)
# Check if Gemini API key is available and provide feedback
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
    # User manually entered a key, overriding the one from secrets
    st.sidebar.info("Using manually entered Gemini Key (overrides secret).", icon="⌨️")
    gemini_configured = True

# Dropdown for selecting the Gemini model
model_options = {
    "Gemini 1.5 Flash (Fastest, Cost-Effective)": "gemini-1.5-flash-latest",
    "Gemini 1.5 Pro (Advanced, Slower, Higher Cost)": "gemini-1.5-pro-latest",
}
selected_model_display_name = st.sidebar.selectbox(
    "Choose the Gemini model for processing:",
    options=list(model_options.keys()), index=0, key="gemini_model_select",
    help="Select the AI model for processing the extracted text. Pro may handle complex rules better."
)
selected_model_id = model_options[selected_model_display_name] # Get the actual model ID
st.sidebar.caption(f"Selected model ID: `{selected_model_id}`")

# Text area for defining Gemini processing rules
st.sidebar.markdown("---")
st.sidebar.header("📜 Gemini Processing Rules")
# --- Default rules focused on cleaning Arabic/Urdu text ---
default_rules = """Structure the text into paragraphs.
Delete headers (typically signifies the name of a chapter).
Delete footnotes.
Inspect the pdf. There are two lines.: a top line and a bottom line.
The top line is at the top of the page. Within the first 5cm of the page. Remember all the text above the top line.
Remember all the text below the bottom line.
Compare with the extracted text.
Now delete all text that can be identified as above the top line.
Then delete all the text that can be identified as below the bottom line. Everything below the bottom line is footnotes, so it must be deleted."""
# --- End Default Rules ---
rules_prompt = st.sidebar.text_area(
    "Enter the processing instructions for Gemini:",
    value=default_rules, height=250, # Adjust height as needed
    help="Instructions for how Gemini should process the text extracted by Document AI. (Note: Spatial rules like '5cm' might be difficult for the AI to follow precisely based only on text)."
)
# --- End Gemini Configuration ---


# --- Main Application Area ---

st.header("📁 Manage Files for Extraction & Processing")

# --- File Uploader Widget ---
st.file_uploader(
    "Choose PDF or Image files:",
    type=["pdf", "png", "jpg", "jpeg", "tiff", "tif", "gif", "bmp", "webp"], # Allowed file extensions
    accept_multiple_files=True, # Allow uploading multiple files at once
    key="docai_uploader", # Must match the key used in handle_uploads
    on_change=handle_uploads, # Function to call when files are uploaded
    label_visibility="visible" # Make the label "Choose PDF..." visible
)
# --- End File Uploader ---

st.markdown("---") # Visual separator

# --- TOP: Action Buttons and Progress Indicators ---
st.subheader("🚀 Actions & Progress (Top)")
col_b1_top, col_b2_top = st.columns([3, 2]) # Layout columns

with col_b1_top:
    # Determine if the process button should be enabled
    process_button_enabled = (
        not st.session_state.processing_started and # Not already processing
        st.session_state.ordered_files and          # Files are present
        docai_configured and                       # DocAI is configured
        gemini_configured                          # Gemini is configured
    )
    # Process Button
    process_button_top_clicked = st.button(
        "✨ Process Files & Merge (Top)",
        key="process_button_top_combined", # Unique key for this button
        use_container_width=True, type="primary", # Make button prominent
        disabled=not process_button_enabled # Disable if conditions not met
    )

with col_b2_top:
    # Download Button (appears after successful processing)
    if st.session_state.merged_doc_buffer and not st.session_state.processing_started:
        st.download_button(
            label=f"📥 Download Processed Text ({st.session_state.files_processed_count}) (.docx)",
            data=st.session_state.merged_doc_buffer, # The BytesIO buffer from backend
            file_name="merged_processed_text.docx", # Default download filename
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", # MIME type for .docx
            key="download_merged_button_top_combined", # Unique key
            use_container_width=True
        )
    # Status messages while processing or if configuration is missing
    elif st.session_state.processing_started:
        st.info("Processing in progress...", icon="⏳")
    elif not docai_configured and st.session_state.ordered_files:
         st.warning("Configure Document AI in sidebar.", icon="⚙️")
    elif not gemini_configured and st.session_state.ordered_files:
         st.warning("Configure Gemini API Key in sidebar.", icon="🔑")
    elif not st.session_state.ordered_files:
         st.markdown("*(Upload files to enable processing)*")
    else: # Ready to process, but button not clicked yet
        st.markdown("*(Download button appears here after processing)*")


# Placeholders for dynamic progress bar and status text updates (Top)
progress_bar_placeholder_top = st.empty()
status_text_placeholder_top = st.empty()

st.markdown("---") # Visual separator

# --- Interactive File List Display ---
st.subheader(f"Files in Processing Order ({len(st.session_state.ordered_files)}):")

if not st.session_state.ordered_files:
    # Message if no files are uploaded yet
    st.info("Use the uploader above to add PDF or Image files.")
else:
    # Display header row for the file list
    col_h1, col_h2, col_h3, col_h4, col_h5 = st.columns([0.5, 5, 1, 1, 1])
    with col_h1: st.markdown("**#**")
    with col_h2: st.markdown("**Filename**")
    with col_h3: st.markdown("**Up**")
    with col_h4: st.markdown("**Down**")
    with col_h5: st.markdown("**Remove**")

    # Iterate through the ordered files and display controls for each
    for i, file in enumerate(st.session_state.ordered_files):
        col1, col2, col3, col4, col5 = st.columns([0.5, 5, 1, 1, 1])
        with col1: st.write(f"{i+1}") # Display file number
        with col2: st.write(file.name) # Display filename
        # Move Up button (disabled for the first file)
        with col3: st.button("⬆️", key=f"up_combined_{i}", on_click=move_file, args=(i, -1), disabled=(i == 0), help="Move Up")
        # Move Down button (disabled for the last file)
        with col4: st.button("⬇️", key=f"down_combined_{i}", on_click=move_file, args=(i, 1), disabled=(i == len(st.session_state.ordered_files) - 1), help="Move Down")
        # Remove button
        with col5: st.button("❌", key=f"del_combined_{i}", on_click=remove_file, args=(i,), help="Remove")

    # Button to remove all files at once
    st.button("🗑️ Remove All Files",
              key="remove_all_button_combined", # Unique key
              on_click=clear_all_files_callback,
              help="Click to remove all files from the list.",
              type="secondary") # Less prominent style


st.markdown("---") # Visual separator

# --- BOTTOM: Action Buttons and Progress Indicators ---
# Duplicates the top buttons/indicators for convenience at the bottom of the page
st.subheader("🚀 Actions & Progress (Bottom)")
col_b1_bottom, col_b2_bottom = st.columns([3, 2])

with col_b1_bottom:
    # Process Button (Bottom) - uses same enablement logic as top
    process_button_bottom_clicked = st.button(
        "✨ Process Files & Merge (Bottom)",
        key="process_button_bottom_combined", # Unique key
        use_container_width=True, type="primary",
        disabled=not process_button_enabled
    )

with col_b2_bottom:
    # Download Button (Bottom) - uses same logic as top
    if st.session_state.merged_doc_buffer and not st.session_state.processing_started:
        st.download_button(
            label=f"📥 Download Processed Text ({st.session_state.files_processed_count}) (.docx)",
            data=st.session_state.merged_doc_buffer,
            file_name="merged_processed_text.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            key="download_merged_button_bottom_combined", # Unique key
            use_container_width=True
        )
    # Status messages (Bottom)
    elif st.session_state.processing_started:
        st.info("Processing in progress...", icon="⏳")
    elif not docai_configured and st.session_state.ordered_files:
         st.warning("Configure Document AI in sidebar.", icon="⚙️")
    elif not gemini_configured and st.session_state.ordered_files:
         st.warning("Configure Gemini API Key in sidebar.", icon="🔑")
    elif not st.session_state.ordered_files:
         st.markdown("*(Upload files to enable processing)*")
    else:
        st.markdown("*(Download button appears here after processing)*")

# Placeholders for dynamic progress bar and status text updates (Bottom)
progress_bar_placeholder_bottom = st.empty()
status_text_placeholder_bottom = st.empty()

# --- Container for displaying detailed results per file ---
results_container = st.container()


# --- == Main Processing Logic == ---
# This block executes if either the top or bottom "Process" button is clicked
if process_button_top_clicked or process_button_bottom_clicked:
    reset_processing_state() # Clear previous results
    st.session_state.processing_started = True # Set processing flag

    # --- Get Config values needed for processing ---
    project_id = st.session_state.docai_project_id
    location = st.session_state.docai_location
    processor_id = st.session_state.docai_processor_id
    docai_configured = all([project_id, location, processor_id]) # Re-check just before use
    gemini_api_key = api_key # Get from sidebar widget variable
    gemini_configured = bool(gemini_api_key) # Re-check just before use
    current_rules = rules_prompt # Get from sidebar widget variable

    # --- Final Pre-flight Checks before starting the loop ---
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
        current_rules = default_rules # Ensure default rules are used if empty
    elif not selected_model_id:
        st.error("❌ No Gemini model selected in the sidebar.")
        st.session_state.processing_started = False
    # --- End Pre-flight Checks ---

    # Proceed only if processing_started flag is still True after checks
    if st.session_state.processing_started:
        processed_doc_streams_for_merge = [] # List to hold (filename, BytesIO stream) tuples
        files_successfully_processed = 0 # Counter for files that result in a stream (even if placeholder)
        total_files = len(st.session_state.ordered_files)

        # Initialize progress bars
        progress_bar_top = progress_bar_placeholder_top.progress(0, text="Starting processing...")
        progress_bar_bottom = progress_bar_placeholder_bottom.progress(0, text="Starting processing...")

        # --- Loop through each uploaded file in order ---
        for i, file_to_process in enumerate(st.session_state.ordered_files):
            original_filename = file_to_process.name
            current_file_status = f"'{original_filename}' ({i + 1}/{total_files})"
            progress_text = f"Processing {current_file_status}..."

            # Update progress bars and status text
            progress_value = i / total_files
            progress_bar_top.progress(progress_value, text=progress_text)
            progress_bar_bottom.progress(progress_value, text=progress_text)
            status_text_placeholder_top.info(f"🔄 Starting {current_file_status}")
            status_text_placeholder_bottom.info(f"🔄 Starting {current_file_status}")

            # Display header for this file in the results area
            with results_container:
                st.markdown(f"--- \n**Processing: {original_filename}**")

            # Initialize variables for this file's processing steps
            raw_text = None
            processed_text = ""
            extraction_error = False
            gemini_error_occurred = False
            word_creation_error_occurred = False

            # --- Step 1: Extract Text with Document AI ---
            status_text_placeholder_top.info(f"📄 Extracting text via Document AI for {current_file_status}...")
            status_text_placeholder_bottom.info(f"📄 Extracting text via Document AI for {current_file_status}...")
            try:
                # Call backend function
                raw_text_result = backend.extract_text_with_docai(
                    file_to_process, project_id, location, processor_id
                )
                # Check result from backend
                if isinstance(raw_text_result, str) and raw_text_result.startswith("Error:"):
                    # Handle specific error message from backend
                    with results_container: st.error(f"❌ Document AI Error for '{original_filename}': {raw_text_result}")
                    extraction_error = True
                    raw_text = None # Ensure no text proceeds to next step
                elif not raw_text_result or not raw_text_result.strip():
                    # Handle case where extraction succeeded but found no text
                    with results_container: st.warning(f"⚠️ No text extracted by Document AI from '{original_filename}'.")
                    raw_text = "" # Use empty string to signify no text found
                else:
                    # Success: store the extracted text
                    raw_text = raw_text_result
                    with results_container: st.info(f"📄 Text extracted successfully for '{original_filename}'.")
            except Exception as ext_exc:
                # Handle unexpected errors during the backend call
                with results_container: st.error(f"❌ Unexpected error during Document AI extraction for '{original_filename}': {ext_exc}")
                extraction_error = True
                raw_text = None

            # --- Step 2: Process with Gemini ---
            # Proceed only if extraction didn't have a critical error
            if not extraction_error and raw_text is not None:
                # Only call Gemini if there is actual text to process
                if raw_text.strip():
                    status_text_placeholder_top.info(f"✨ Processing text from {current_file_status} via Gemini ({selected_model_display_name})...")
                    status_text_placeholder_bottom.info(f"✨ Processing text from {current_file_status} via Gemini ({selected_model_display_name})...")
                    try:
                        # Call backend function
                        processed_text_result = backend.process_text_with_gemini(
                            gemini_api_key, raw_text, current_rules, selected_model_id
                        )
                        # Check result from backend
                        if isinstance(processed_text_result, str) and processed_text_result.startswith("Error:"):
                            # Handle specific error message from backend
                            with results_container: st.error(f"❌ Gemini processing error for '{original_filename}': {processed_text_result}")
                            gemini_error_occurred = True
                            processed_text = "" # Use empty string for Word creation on error
                        elif processed_text_result is None:
                             # Handle unexpected None response
                             with results_container: st.error(f"❌ Gemini processing error for '{original_filename}': Received None response.")
                             gemini_error_occurred = True
                             processed_text = ""
                        else:
                            # Success: store the processed text
                            processed_text = processed_text_result
                            with results_container: st.success(f"✨ Text processed successfully by Gemini for '{original_filename}'.")
                    except Exception as gem_exc:
                        # Handle unexpected errors during the backend call
                        with results_container: st.error(f"❌ Unexpected error during Gemini processing for '{original_filename}': {gem_exc}")
                        gemini_error_occurred = True
                        processed_text = ""
                else:
                    # If extracted text was empty, log and skip Gemini call
                    logging.info(f"Skipping Gemini processing for '{original_filename}' as extracted text was empty.")
                    processed_text = "" # Ensure processed_text is empty for Word creation
                    with results_container: st.info(f"✨ Skipping Gemini processing for '{original_filename}' (no extracted text).")

            # --- Step 3: Create Individual Word Document ---
            # This step runs regardless of previous errors to ensure a document (even placeholder)
            # is created for each input file, maintaining order for merging.
            status_text_placeholder_top.info(f"📝 Creating intermediate Word document for {current_file_status}...")
            status_text_placeholder_bottom.info(f"📝 Creating intermediate Word document for {current_file_status}...")
            try:
                # Call backend function (which now handles RTL and placeholders)
                word_doc_stream = backend.create_word_doc_from_processed_text(
                    processed_text, original_filename, extraction_error, gemini_error_occurred
                )
                # Check if stream was created successfully
                if word_doc_stream:
                    # Add the stream to the list for merging
                    processed_doc_streams_for_merge.append((original_filename, word_doc_stream))
                    files_successfully_processed += 1 # Increment success counter
                    # Display success message with context about potential issues
                    with results_container:
                        success_msg = f"✅ Created intermediate document for '{original_filename}'."
                        if extraction_error: success_msg += " (Note: placeholder used due to Document AI extraction error)"
                        elif gemini_error_occurred: success_msg += " (Note: placeholder or original text used due to Gemini processing error)"
                        elif not processed_text.strip(): success_msg += " (Note: document may be empty or contain placeholder as no text was extracted or processed)"
                        st.success(success_msg)
                else:
                    # Handle failure to create the stream in the backend
                    word_creation_error_occurred = True
                    with results_container: st.error(f"❌ Failed to create intermediate Word file for '{original_filename}'.")
            except Exception as doc_exc:
                # Handle unexpected errors during the backend call
                word_creation_error_occurred = True
                with results_container: st.error(f"❌ Error during intermediate Word file creation for '{original_filename}': {doc_exc}")

            # --- Update Progress Bar after processing one file ---
            status_msg_suffix = ""
            if extraction_error or gemini_error_occurred or word_creation_error_occurred:
                status_msg_suffix = " with issues."
            final_progress_value = (i + 1) / total_files
            final_progress_text = f"Processed {current_file_status}{status_msg_suffix}"
            progress_bar_top.progress(final_progress_value, text=final_progress_text)
            progress_bar_bottom.progress(final_progress_value, text=final_progress_text)

        # --- End of file processing loop ---

        # Clear progress bars and status text placeholders
        progress_bar_placeholder_top.empty()
        progress_bar_placeholder_bottom.empty()
        status_text_placeholder_top.empty()
        status_text_placeholder_bottom.empty()

        # --- Step 4: Merge Documents ---
        final_status_message = ""
        rerun_needed = False # Flag to trigger st.rerun if download button needs update
        with results_container:
            st.markdown("---") # Separator before final merge status
            # Proceed only if there are successfully created streams to merge
            if files_successfully_processed > 0:
                st.info(f"💾 Merging {files_successfully_processed} processed Word document(s)...")
                try:
                    # Call backend merge function
                    merged_buffer = backend.merge_word_documents(processed_doc_streams_for_merge)
                    # Check if merging was successful
                    if merged_buffer:
                        # Store buffer in session state for download button
                        st.session_state.merged_doc_buffer = merged_buffer
                        st.session_state.files_processed_count = files_successfully_processed
                        # Construct final success message
                        final_status_message = f"✅ Processing complete! Merged document created from {files_successfully_processed} source file(s)."
                        if files_successfully_processed < total_files:
                            final_status_message += f" ({total_files - files_successfully_processed} file(s) had issues during processing)."
                        st.success(final_status_message)
                        rerun_needed = True # Need to rerun to enable download button
                    else:
                        # Handle merge failure reported by backend
                        final_status_message = "❌ Failed to merge Word documents."
                        st.error(final_status_message)
                except Exception as merge_exc:
                    # Handle unexpected errors during merge call
                    final_status_message = f"❌ Error during document merging: {merge_exc}"
                    logging.error(f"Error during merge_word_documents call: {merge_exc}", exc_info=True)
                    st.error(final_status_message)
            elif total_files > 0:
                 # Case where files were uploaded but none processed successfully
                 final_status_message = "⚠️ No documents were successfully processed to merge."
                 st.warning(final_status_message)
            # Else: No files were uploaded initially (handled by initial checks)

        # Final state updates after processing loop finishes
        st.session_state.processing_complete = True
        st.session_state.processing_started = False
        # Rerun the script if needed to update the download button state
        if rerun_needed:
            st.rerun()

    else: # Initial pre-flight checks failed
        # Reset processing flag if checks failed before loop started
        st.session_state.processing_started = False
        # Error messages were already displayed during checks

# --- Fallback informational message ---
# Display if no files are loaded and not currently processing
if not st.session_state.ordered_files and not st.session_state.processing_started and not st.session_state.processing_complete:
    st.info("Upload PDF or Image files using the button above.")

# --- Footer ---
st.markdown("---")
st.markdown("Developed with Streamlit, Google Document AI, and Google Gemini.")


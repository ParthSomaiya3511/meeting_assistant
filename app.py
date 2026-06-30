import os
import tempfile
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder

# Load environment variables
load_dotenv()

from utils.transcriber import transcribe_audio
from utils.analyzer import analyze_transcript

st.set_page_config(
    page_title="AI Meeting Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("AI Meeting Assistant")
st.markdown("Record or upload meeting audio to transcribe, summarize, and manage tasks interactively.")

with st.sidebar:
    st.header("Configuration")
    
    api_key_configured = bool(os.getenv("GROQ_API_KEY"))
    if not api_key_configured:
        st.warning("Please add your GROQ_API_KEY to the .env file to run the app.")

    st.markdown("---")
    st.header("Audio Input Source")
    
    st.subheader("Live Recording")
    recorded_audio = mic_recorder(
        start_prompt="Record Audio",
        stop_prompt="Stop Recording",
        just_once=False,
        use_container_width=True,
        key="recorder"
    )
    
    st.markdown("---")
    
    st.subheader("Upload Audio File")
    uploaded_file = st.file_uploader(
        "Upload a meeting recording (MP3, WAV, M4A)",
        type=["mp3", "wav", "m4a"]
    )

# Retrieve input audio path
audio_path = None
if recorded_audio:
    temp_dir = tempfile.gettempdir()
    audio_path = os.path.join(temp_dir, "recorded_meeting.wav")
    with open(audio_path, "wb") as f:
        f.write(recorded_audio['bytes'])
    st.sidebar.success("Recorded audio captured.")
elif uploaded_file is not None:
    temp_dir = tempfile.gettempdir()
    audio_path = os.path.join(temp_dir, uploaded_file.name)
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.read())
    st.sidebar.success(f"Uploaded: {uploaded_file.name}")

# Process audio
if audio_path:
    if st.button("Process Meeting", type="primary", use_container_width=True):
        if not api_key_configured:
            st.error("GROQ_API_KEY is missing from the environment setup.")
        else:
            with st.spinner("Processing analysis..."):
                try:
                    st.info("Step 1/2: Transcribing Audio...")
                    transcript = transcribe_audio(audio_path)
                    
                    st.info("Step 2/2: Extracting Summary and Action Items...")
                    analysis_result = analyze_transcript(transcript)
                    
                    # Store variables in session state
                    st.session_state["transcript"] = transcript
                    st.session_state["summary"] = analysis_result.summary
                    
                    # Store action items as list of dictionaries for easier pandas conversion
                    st.session_state["tasks_list"] = [
                        {
                            "Completed": False,
                            "Task": item.task,
                            "Assignee": item.assignee,
                            "Deadline": item.deadline
                        }
                        for item in analysis_result.action_items
                    ]
                    
                    st.success("Processing complete.")
                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")
                finally:
                    if audio_path and os.path.exists(audio_path):
                        os.remove(audio_path)

# Display Results
if "transcript" in st.session_state:
    tab1, tab2, tab3 = st.tabs(["📝 Summary & Task Manager", "🗣️ Original Transcript", "💾 Export Data"])
    
    with tab1:
        st.subheader("Meeting Summary")
        st.write(st.session_state["summary"])
        
        st.markdown("---")
        
        st.subheader("Action Items & Task Manager")
        st.markdown(
            "Use the table below to **edit** existing items, **check** completion, "
            "**add** new tasks (click the empty row with the `+` sign), or select rows to **delete** them."
        )
        
        # Ensure tasks_list exists in session state (even if empty)
        if "tasks_list" not in st.session_state:
            st.session_state["tasks_list"] = []

        # Convert to DataFrame for the interactive editor
        df_tasks = pd.DataFrame(st.session_state["tasks_list"])
        
        # Fallback empty dataframe structure if no tasks exist yet
        if df_tasks.empty:
            df_tasks = pd.DataFrame(columns=["Completed", "Task", "Assignee", "Deadline"])
        
        # Interactive data editor
        edited_df = st.data_editor(
            df_tasks,
            num_rows="dynamic",  # Enables adding and deleting rows
            use_container_width=True,
            column_config={
                "Completed": st.column_config.CheckboxColumn(
                    help="Mark task as complete",
                    default=False,
                ),
                "Task": st.column_config.TextColumn(
                    width="large",
                    required=True,
                ),
                "Assignee": st.column_config.TextColumn(
                    default="Unassigned",
                ),
                "Deadline": st.column_config.TextColumn(
                    default="None specified",
                )
            }
        )
        
        # Save modifications back to session state
        st.session_state["tasks_list"] = edited_df.to_dict("records")

    with tab2:
        st.subheader("Full Transcript (Original)")
        st.text_area("Transcription Text", value=st.session_state["transcript"], height=300, disabled=True)

    with tab3:
        st.subheader("Export Results")
        
        # Compile latest modified tasks from session_state for export
        export_tasks = st.session_state.get("tasks_list", [])
        
        # Build Markdown Export
        export_md = f"# Meeting Analysis\n\n"
        export_md += f"## Executive Summary\n{st.session_state['summary']}\n\n"
        export_md += "## Action Items\n"
        
        if export_tasks:
            for item in export_tasks:
                status = "x" if item.get("Completed") else " "
                export_md += f"- [{status}] **Task**: {item.get('Task')} | **Assignee**: {item.get('Assignee')} | **Deadline**: {item.get('Deadline')}\n"
        else:
            export_md += "*No tasks assigned.*\n"
            
        # Download Markdown File
        st.download_button(
            label="Download Analysis (Markdown)",
            data=export_md,
            file_name="meeting_analysis.md",
            mime="text/markdown"
        )
        
        # Download CSV File option for Task Management Import
        if export_tasks:
            df_export = pd.DataFrame(export_tasks)
            csv_data = df_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Tasks Only (CSV)",
                data=csv_data,
                file_name="meeting_tasks.csv",
                mime="text/csv"
            )
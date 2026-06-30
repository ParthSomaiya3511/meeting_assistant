from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

class TaskItem(BaseModel):
    task: str = Field(
        default="No task description", 
        description="The specific action item or task to be executed, translated to English."
    )
    assignee: str = Field(
        default="Unassigned", 
        description="The person responsible for this task. Keep the original name."
    )
    deadline: str = Field(
        default="None specified", 
        description="Due date or timeframe. Translate terms like 'پیر' to 'Monday'."
    )

class MeetingAnalysis(BaseModel):
    summary: str = Field(
        default="The transcript was too short or lacked context to generate a summary.", 
        description="A concise executive summary of the discussion, translated to English."
    )
    action_items: List[TaskItem] = Field(
        default_factory=list, 
        description="A clean list of actionable tasks. Leave empty if no specific tasks are mentioned."
    )

def analyze_transcript(transcript: str) -> MeetingAnalysis:
    """
    Analyzes transcripts (English, Urdu, or bilingual) and returns structured analysis in English.
    """
    # Using Llama 3.3 70B for robust translation and structured understanding
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    structured_llm = llm.with_structured_output(MeetingAnalysis)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a professional administrative assistant. Analyze the provided transcript, "
            "which may be in English, Urdu, or a mix of both (bilingual/Roman Urdu). "
            "1. Generate a clear summary of the discussion in English.\n"
            "2. Extract any action items, translating the tasks and deadlines into English while keeping original names.\n"
            "3. If the input is poetic, philosophical, or lacks business tasks, summarize the philosophical "
            "meaning in English and leave the action items list empty. Do not invent tasks."
        ),
        ("user", "Transcript:\n\n{transcript}")
    ])

    chain = prompt | structured_llm
    return chain.invoke({"transcript": transcript})
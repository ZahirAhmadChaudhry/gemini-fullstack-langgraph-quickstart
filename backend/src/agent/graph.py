import os
import json
from typing import List, Dict, Any

from agent.tools_and_schemas import SegmentsList, TensionExtraction, Categorization, FullAnalysisResult
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig

from agent.state import (
    OverallState,
    SegmentationState,
    AnalysisResult,
)
from agent.configuration import Configuration
from agent.prompts import (
    segmentation_instructions,
    tension_extraction_instructions,
    categorization_instructions,
    synthesis_instructions,
    imaginaire_classification_instructions,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.utils import (
    clean_transcript,
    detect_period,
    determine_theme,
    assign_code,
)

load_dotenv()

if os.getenv("GEMINI_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY is not set")


# Nodes
def segment_transcript(state: OverallState, config: RunnableConfig) -> SegmentationState:
    """LangGraph node that segments the transcript into tension-containing excerpts.

    Uses Gemini 2.0 Flash to identify segments containing paradoxes or tensions
    based on contrastive markers and discourse patterns.

    Args:
        state: Current graph state containing the transcript text
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including segments list
    """
    configurable = Configuration.from_runnable_config(config)

    # Get transcript from preprocessed data or direct input
    transcript_text = state.get("transcript", "")
    if not transcript_text and state.get("preprocessed_data"):
        # Extract text from preprocessed segments
        segments_data = state["preprocessed_data"].get("segments", [])
        transcript_text = "\n".join([seg.get("text", "") for seg in segments_data if seg.get("text")])

    if not transcript_text:
        return {"segments": []}

    # Clean the transcript
    cleaned_text = clean_transcript(transcript_text)

    # init Gemini 2.0 Flash for segmentation
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",  # Use production model (not experimental)
        temperature=0.3,  # Low temperature for consistent segmentation
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    structured_llm = llm.with_structured_output(SegmentsList)

    # Format the prompt for French transcript segmentation
    formatted_prompt = segmentation_instructions.format(
        transcript_text=cleaned_text
    )

    # Generate the segments
    result = structured_llm.invoke(formatted_prompt)

    # Store metadata for each segment
    segments_with_metadata = []
    for i, segment_text in enumerate(result.segments):
        segment_data = {
            "id": f"seg_{i+1:03d}",
            "text": segment_text,
            "period": detect_period(segment_text),
            "theme": determine_theme(segment_text),
        }
        segments_with_metadata.append(segment_data)

    return {"segments": segments_with_metadata}


def continue_to_analysis(state: SegmentationState):
    """LangGraph node that sends segments to analysis nodes.

    This is used to spawn n number of analysis nodes, one for each segment.
    """
    return [
        Send("analyze_segment", {"segment": segment, "segment_id": segment["id"]})
        for segment in state["segments"]
    ]


def analyze_segment(state: Dict[str, Any], config: RunnableConfig) -> OverallState:
    """LangGraph node that analyzes a single segment to extract tension information.

    Performs multi-step analysis of a transcript segment to extract all required fields
    for the French sustainability analysis.

    Args:
        state: Current state containing the segment data
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including analysis results
    """
    configurable = Configuration.from_runnable_config(config)
    segment = state["segment"]
    segment_text = segment["text"]
    segment_id = segment["id"]

    # Initialize Gemini 2.0 Flash
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",  # Use production model
        temperature=0.2,  # Low temperature for consistent analysis
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    # Step 1: Extract tension (original and reformulated items)
    tension_llm = llm.with_structured_output(TensionExtraction)
    tension_prompt = tension_extraction_instructions.format(
        segment_text=segment_text
    )
    tension_result = tension_llm.invoke(tension_prompt)

    # Step 2: Categorize (second-order concept and code)
    categorization_llm = llm.with_structured_output(Categorization)
    categorization_prompt = categorization_instructions.format(
        reformulated_item=tension_result.reformulated_item,
        segment_text=segment_text
    )
    categorization_result = categorization_llm.invoke(categorization_prompt)

    # Step 3: Generate synthesis
    synthesis_prompt = synthesis_instructions.format(
        reformulated_item=tension_result.reformulated_item,
        segment_text=segment_text
    )
    synthesis_result = llm.invoke(synthesis_prompt)

    # Step 4: Classify imaginaire (C/S and IFa/IFr)
    imaginaire_prompt = imaginaire_classification_instructions.format(
        segment_text=segment_text,
        synthesis=synthesis_result.content
    )
    imaginaire_result = llm.invoke(imaginaire_prompt)

    # Step 5: Assign specific code using domain knowledge
    specific_code = assign_code(
        reformulated_item=tension_result.reformulated_item,
        second_order_concept=categorization_result.concept
    )

    # Create the analysis result
    analysis_result = {
        "segment_id": segment_id,
        "concepts_2nd_ordre": categorization_result.concept,
        "items_1er_ordre_reformule": tension_result.reformulated_item,
        "items_1er_ordre_origine": tension_result.original_item,
        "details": segment_text,
        "synthese": synthesis_result.content.strip(),
        "periode": segment.get("period", detect_period(segment_text)),
        "theme": segment.get("theme", determine_theme(segment_text)),
        "code_spe": specific_code,
        "imaginaire": imaginaire_result.content.strip(),
    }

    return {"analysis_results": [analysis_result]}

def finalize_output(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the transcript analysis results.

    Aggregates all analysis results and formats them for output.

    Args:
        state: Current graph state containing all analysis results

    Returns:
        Dictionary with state update, including final formatted results
    """
    analysis_results = state.get("analysis_results", [])

    if not analysis_results:
        return {
            "messages": [AIMessage(content="No tensions were identified in the transcript.")],
            "final_results": []
        }

    # Format results for output
    formatted_results = []
    for result in analysis_results:
        formatted_result = {
            "Concepts de 2nd ordre": result.get("concepts_2nd_ordre", ""),
            "Items de 1er ordre reformulé": result.get("items_1er_ordre_reformule", ""),
            "Items de 1er ordre (intitulé d'origine)": result.get("items_1er_ordre_origine", ""),
            "Détails": result.get("details", ""),
            "Synthèse": result.get("synthese", ""),
            "Période": result.get("periode", ""),
            "Thème": result.get("theme", ""),
            "Code spé": result.get("code_spe", ""),
            "Constat ou stéréotypes (C ou S) (Imaginaire facilitant IFa ou Imaginaire frein IFr)": result.get("imaginaire", ""),
        }
        formatted_results.append(formatted_result)

    # Create summary message
    num_tensions = len(formatted_results)
    themes = [r.get("Thème", "") for r in formatted_results]
    theme_counts = {theme: themes.count(theme) for theme in set(themes) if theme}

    summary_parts = [f"Analyse terminée. {num_tensions} tensions identifiées."]
    if theme_counts:
        theme_summary = ", ".join([f"{theme}: {count}" for theme, count in theme_counts.items()])
        summary_parts.append(f"Répartition par thème: {theme_summary}")

    summary_message = " ".join(summary_parts)

    return {
        "messages": [AIMessage(content=summary_message)],
        "final_results": formatted_results,
    }





# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes for transcript analysis
builder.add_node("segment_transcript", segment_transcript)
builder.add_node("analyze_segment", analyze_segment)
builder.add_node("finalize_output", finalize_output)

# Set the entrypoint as `segment_transcript`
# This means that this node is the first one called
builder.add_edge(START, "segment_transcript")

# Add conditional edge to continue with segment analysis in parallel branches
builder.add_conditional_edges(
    "segment_transcript", continue_to_analysis, ["analyze_segment"]
)

# After all segments are analyzed, finalize the output
builder.add_edge("analyze_segment", "finalize_output")

# End the graph
builder.add_edge("finalize_output", END)

graph = builder.compile(name="french-sustainability-analyzer")

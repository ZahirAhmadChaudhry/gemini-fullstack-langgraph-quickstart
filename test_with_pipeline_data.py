#!/usr/bin/env python3
"""
Test the refactored backend with actual data from the data engineering pipeline.
Uses first 10 segments from Table_A_ml_ready.json to test the workflow.
"""

import os
import sys
import json
from pathlib import Path

# Add the backend source to the path
backend_path = Path(__file__).parent / "backend" / "src"
sys.path.insert(0, str(backend_path))

from agent.graph import graph
from langchain_core.messages import HumanMessage

def load_pipeline_data(file_path: str, max_segments: int = 10):
    """Load and prepare data from the pipeline output."""
    
    print(f"📂 Loading data from: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract first N segments
    segments = data.get('segments', [])[:max_segments]
    
    print(f"📊 Loaded {len(segments)} segments from {data.get('source_file', 'unknown')}")
    
    # Convert to our expected format
    processed_segments = []
    for segment in segments:
        # Extract relevant information from pipeline data
        segment_data = {
            "id": segment["id"],
            "text": segment["text"],
            "period": extract_period_from_features(segment.get("features", {})),
            "theme": extract_theme_from_features(segment.get("features", {})),
            "ml_features": segment.get("features", {}).get("ml_features", {}),
            "tension_indicators": segment.get("features", {}).get("tension_patterns", {}),
        }
        processed_segments.append(segment_data)
    
    return {
        "source_file": data.get("source_file", "unknown"),
        "processed_timestamp": data.get("processed_timestamp", "unknown"),
        "segments": processed_segments
    }

def extract_period_from_features(features):
    """Extract period from pipeline features."""
    temporal_context = features.get("temporal_context", "unknown")
    if temporal_context == "2023":
        return "2023"
    elif temporal_context == "2050":
        return "2050"
    
    # Check ml_features for temporal_period
    ml_features = features.get("ml_features", {})
    temporal_period = ml_features.get("temporal_period", 0)
    if temporal_period == 2023.0:
        return "2023"
    elif temporal_period == 2050.0:
        return "2050"
    
    return ""

def extract_theme_from_features(features):
    """Extract theme from pipeline features."""
    thematic_indicators = features.get("thematic_indicators", {})
    performance_density = thematic_indicators.get("performance_density", 0)
    legitimacy_density = thematic_indicators.get("legitimacy_density", 0)
    
    if performance_density > legitimacy_density:
        return "Performance"
    elif legitimacy_density > performance_density:
        return "Légitimité"
    
    return ""

def test_with_pipeline_data():
    """Test the backend with pipeline data."""
    
    print("🎯 Testing French Sustainability Analyzer with Pipeline Data")
    print("=" * 65)
    
    # Check API key
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY not found in environment variables")
        return False
    
    print("✅ GEMINI_API_KEY found")
    
    # Load pipeline data
    try:
        pipeline_data = load_pipeline_data("Table_A_ml_ready.json", max_segments=10)
    except FileNotFoundError:
        print("❌ Table_A_ml_ready.json not found")
        print("Please make sure the file is in the current directory")
        return False
    except Exception as e:
        print(f"❌ Error loading pipeline data: {e}")
        return False
    
    # Show sample segments
    print(f"\n📋 Sample segments to analyze:")
    for i, segment in enumerate(pipeline_data["segments"][:3]):
        print(f"   {i+1}. {segment['id']}: {segment['text'][:100]}...")
        if segment.get('tension_indicators'):
            print(f"      Pipeline detected tensions: {list(segment['tension_indicators'].keys())}")
    
    # Prepare state for our graph
    initial_state = {
        "messages": [HumanMessage(content="Analyze French sustainability transcript segments")],
        "transcript": "",  # We'll use preprocessed_data instead
        "preprocessed_data": pipeline_data,
        "segments": [],
        "analysis_results": [],
        "final_results": [],
        "max_segments": 10
    }
    
    print(f"\n🚀 Starting analysis of {len(pipeline_data['segments'])} segments...")
    print("⏳ This may take a few minutes with the free API...")
    
    try:
        # Run the graph
        result = graph.invoke(initial_state)
        
        print("\n✅ Analysis completed successfully!")
        
        # Display results summary
        segments_analyzed = len(result.get('segments', []))
        tensions_found = len(result.get('analysis_results', []))
        final_results = len(result.get('final_results', []))
        
        print(f"📊 Results Summary:")
        print(f"   - Segments processed: {segments_analyzed}")
        print(f"   - Tensions identified: {tensions_found}")
        print(f"   - Final results: {final_results}")
        
        # Show detailed results
        if result.get('final_results'):
            print(f"\n📋 Detailed Results:")
            print("-" * 50)
            
            for i, tension in enumerate(result['final_results'][:5]):  # Show first 5
                print(f"\n🔍 Tension {i+1}:")
                print(f"   Concept 2nd ordre: {tension.get('Concepts de 2nd ordre', 'N/A')}")
                print(f"   Reformulé: {tension.get('Items de 1er ordre reformulé', 'N/A')}")
                original_key = "Items de 1er ordre (intitulé d'origine)"
                print(f"   Original: {tension.get(original_key, 'N/A')[:100]}...")
                print(f"   Thème: {tension.get('Thème', 'N/A')}")
                print(f"   Période: {tension.get('Période', 'N/A')}")
                print(f"   Code spé: {tension.get('Code spé', 'N/A')}")
                print(f"   Synthèse: {tension.get('Synthèse', 'N/A')}")
        
        # Show final message
        if result.get('messages'):
            final_message = result['messages'][-1]
            print(f"\n💬 Final Summary: {final_message.content}")
        
        # Compare with pipeline predictions
        print(f"\n🔬 Pipeline vs LLM Comparison:")
        pipeline_tensions = sum(1 for seg in pipeline_data['segments'] 
                              if seg.get('tension_indicators'))
        print(f"   - Pipeline detected tensions in: {pipeline_tensions} segments")
        print(f"   - LLM identified tensions: {tensions_found}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🎯 French Sustainability Transcript Analyzer - Pipeline Integration Test")
    print("=" * 75)
    
    success = test_with_pipeline_data()
    
    if success:
        print("\n🎉 Pipeline integration test successful!")
        print("\nNext steps:")
        print("1. Analyze full dataset")
        print("2. Compare results with expert annotations")
        print("3. Refactor frontend for better visualization")
        print("4. Deploy to university servers")
    else:
        print("\n⚠️  Test failed. Check the errors above.")
        print("\nTroubleshooting:")
        print("1. Ensure GEMINI_API_KEY is set")
        print("2. Check Table_A_ml_ready.json is in current directory")
        print("3. Verify internet connection for API calls")

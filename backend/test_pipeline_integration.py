#!/usr/bin/env python3
"""
Test the refactored backend with actual data from the data engineering pipeline.
Run from backend directory with: uv run python test_pipeline_integration.py
"""

import os
import sys
import json
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent.graph import graph
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
from typing import Dict, Any, List
from langchain_core.outputs import LLMResult

class TokenUsageTracker(BaseCallbackHandler):
    """Callback handler to track token usage across all LLM calls."""

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
        self.call_details = []

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Called when LLM ends running."""
        self.total_calls += 1

        # Try multiple ways to extract token usage
        input_tokens = 0
        output_tokens = 0

        # Method 1: Check llm_output for usage_metadata
        if response.llm_output and 'usage_metadata' in response.llm_output:
            usage = response.llm_output['usage_metadata']
            input_tokens = usage.get('input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)

        # Method 2: Check for token_usage in llm_output
        elif response.llm_output and 'token_usage' in response.llm_output:
            usage = response.llm_output['token_usage']
            input_tokens = usage.get('prompt_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0)

        # Method 3: Estimate based on text length (rough approximation)
        else:
            # Rough estimation: ~4 characters per token for text
            for generation in response.generations:
                for gen in generation:
                    if hasattr(gen, 'text') and gen.text:
                        output_tokens += len(gen.text) // 4

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        self.call_details.append({
            'call': self.total_calls,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'method': 'usage_metadata' if input_tokens > 0 else 'estimated'
        })

    def get_summary(self):
        """Get token usage summary."""
        total_tokens = self.total_input_tokens + self.total_output_tokens
        return {
            'total_calls': self.total_calls,
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': total_tokens,
            'call_details': self.call_details
        }

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

def estimate_tokens(text: str) -> int:
    """Rough estimation of tokens in text (1 token ≈ 4 characters for most languages)."""
    return len(text) // 4

def estimate_prompt_tokens(segments_data: dict) -> dict:
    """Estimate tokens that will be used in prompts."""
    total_text = ""

    # Estimate segmentation prompt tokens
    for segment in segments_data["segments"]:
        total_text += segment["text"]

    segmentation_tokens = estimate_tokens(total_text)

    # Estimate analysis tokens per segment (multiple prompts per segment)
    analysis_tokens_per_segment = 0
    for segment in segments_data["segments"]:
        segment_text = segment["text"]
        # Each segment goes through multiple analysis steps
        analysis_tokens_per_segment += estimate_tokens(segment_text) * 4  # 4 different prompts

    return {
        "estimated_segmentation_input": segmentation_tokens,
        "estimated_analysis_input": analysis_tokens_per_segment,
        "estimated_total_input": segmentation_tokens + analysis_tokens_per_segment,
        "estimated_segments": len(segments_data["segments"])
    }

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
    pipeline_file = "../Table_A_ml_ready.json"
    try:
        pipeline_data = load_pipeline_data(pipeline_file, max_segments=3)  # Start with 3 segments for token analysis
    except FileNotFoundError:
        print(f"❌ {pipeline_file} not found")
        print("Please make sure the file is in the parent directory")
        return False
    except Exception as e:
        print(f"❌ Error loading pipeline data: {e}")
        return False
    
    # Show sample segments
    print(f"\n📋 Sample segments to analyze:")
    for i, segment in enumerate(pipeline_data["segments"][:3]):
        print(f"   {i+1}. {segment['id']}")
        print(f"      Text: {segment['text'][:80]}...")
        print(f"      Period: {segment.get('period', 'Unknown')}")
        print(f"      Theme: {segment.get('theme', 'Unknown')}")
        if segment.get('tension_indicators'):
            print(f"      Pipeline tensions: {list(segment['tension_indicators'].keys())}")
        print()
    
    # Prepare state for our graph
    initial_state = {
        "messages": [HumanMessage(content="Analyze French sustainability transcript segments")],
        "transcript": "",  # We'll use preprocessed_data instead
        "preprocessed_data": pipeline_data,
        "segments": [],
        "analysis_results": [],
        "final_results": [],
        "max_segments": 3
    }
    
    # Estimate token usage before starting
    token_estimates = estimate_prompt_tokens(pipeline_data)
    print(f"\n📊 Token Usage Estimates:")
    print(f"   - Estimated input tokens: ~{token_estimates['estimated_total_input']:,}")
    print(f"   - Estimated output tokens: ~{token_estimates['estimated_total_input'] // 4:,} (rough guess)")
    print(f"   - Total estimated: ~{token_estimates['estimated_total_input'] + token_estimates['estimated_total_input'] // 4:,}")

    print(f"\n🚀 Starting analysis of {len(pipeline_data['segments'])} segments...")
    print("⏳ This may take a few minutes with the free API...")

    # Create token usage tracker
    token_tracker = TokenUsageTracker()

    try:
        # Run the graph with token tracking
        result = graph.invoke(
            initial_state,
            config={"callbacks": [token_tracker]}
        )
        
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
            
            for i, tension in enumerate(result['final_results'][:3]):  # Show first 3
                print(f"\n🔍 Tension {i+1}:")
                print(f"   Concept 2nd ordre: {tension.get('Concepts de 2nd ordre', 'N/A')}")
                print(f"   Reformulé: {tension.get('Items de 1er ordre reformulé', 'N/A')}")
                original_key = "Items de 1er ordre (intitulé d'origine)"
                print(f"   Original: {tension.get(original_key, 'N/A')[:80]}...")
                print(f"   Thème: {tension.get('Thème', 'N/A')}")
                print(f"   Période: {tension.get('Période', 'N/A')}")
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

        # Token usage summary
        token_summary = token_tracker.get_summary()
        print(f"\n💰 Token Usage Summary:")
        print(f"   - Total LLM calls: {token_summary['total_calls']}")
        print(f"   - Input tokens: {token_summary['total_input_tokens']:,}")
        print(f"   - Output tokens: {token_summary['total_output_tokens']:,}")
        print(f"   - Total tokens: {token_summary['total_tokens']:,}")

        # Cost estimation (Gemini 2.0 Flash pricing)
        # Free tier: 15 RPM, 1M TPM, 1,500 RPD
        # Paid tier: $0.075 per 1M input tokens, $0.30 per 1M output tokens
        if token_summary['total_tokens'] > 0:
            input_cost = (token_summary['total_input_tokens'] / 1_000_000) * 0.075
            output_cost = (token_summary['total_output_tokens'] / 1_000_000) * 0.30
            total_cost = input_cost + output_cost

            print(f"   - Estimated cost (if paid): ${total_cost:.4f}")
            print(f"     • Input cost: ${input_cost:.4f}")
            print(f"     • Output cost: ${output_cost:.4f}")

            # Free tier usage
            daily_limit = 1500  # requests per day
            monthly_token_limit = 1_000_000 * 30  # rough monthly estimate

            print(f"   - Free tier usage:")
            print(f"     • Requests used: {token_summary['total_calls']}/{daily_limit} daily limit")
            print(f"     • Tokens used: {token_summary['total_tokens']:,} (monthly ~{monthly_token_limit:,})")

        # Detailed call breakdown
        if token_summary['call_details']:
            print(f"\n📊 Detailed Call Breakdown:")
            for call in token_summary['call_details']:
                print(f"   Call {call['call']}: {call['input_tokens']} in + {call['output_tokens']} out = {call['total_tokens']} total")

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
    
    # Check if we're in the right directory
    if not Path("src/agent").exists():
        print("❌ Please run this script from the backend directory")
        print("   cd backend && uv run python test_pipeline_integration.py")
        exit(1)
    
    success = test_with_pipeline_data()
    
    if success:
        print("\n🎉 Pipeline integration test successful!")
        print("\nNext steps:")
        print("1. Test with more segments (increase max_segments)")
        print("2. Compare results with expert annotations")
        print("3. Refactor frontend for better visualization")
        print("4. Deploy to university servers")
    else:
        print("\n⚠️  Test failed. Check the errors above.")
        print("\nTroubleshooting:")
        print("1. Ensure GEMINI_API_KEY is set in .env file")
        print("2. Check Table_A_ml_ready.json is in parent directory")
        print("3. Verify internet connection for API calls")

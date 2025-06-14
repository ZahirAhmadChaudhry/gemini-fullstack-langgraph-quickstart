#!/usr/bin/env python3
"""
Quick test with production Gemini 2.0 Flash model to verify higher rate limits.
Run from backend directory with: uv run python test_production_model.py
"""

import os
import sys
import json
import time
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent.graph import graph
from langchain_core.messages import HumanMessage

def load_sample_data(file_path: str, max_segments: int = 10):
    """Load sample data for testing."""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    segments = data.get('segments', [])[:max_segments]
    
    processed_segments = []
    for segment in segments:
        segment_data = {
            "id": segment["id"],
            "text": segment["text"],
            "period": extract_period_from_features(segment.get("features", {})),
            "theme": extract_theme_from_features(segment.get("features", {})),
            "ml_features": segment.get("features", {}).get("ml_features", {}),
            "tension_indicators": segment.get("features", {}).get("tension_patterns", {}),
        }
        processed_segments.append(segment_data)
    
    return processed_segments

def extract_period_from_features(features):
    """Extract period from pipeline features."""
    temporal_context = features.get("temporal_context", "unknown")
    if temporal_context in ["2023", "2050"]:
        return temporal_context
    
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

def main():
    """Test production model with higher rate limits."""
    
    print("🏭 Testing Production Gemini 2.0 Flash Model")
    print("=" * 50)
    
    # Check prerequisites
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY not found in environment variables")
        return
    
    # Load sample data
    pipeline_file = "../Table_A_ml_ready.json"
    try:
        segments = load_sample_data(pipeline_file, max_segments=10)
        print(f"📊 Loaded {len(segments)} segments for testing")
    except Exception as e:
        print(f"❌ Error loading pipeline data: {e}")
        return
    
    # Test with production model
    initial_state = {
        "messages": [HumanMessage(content="Test production model with 10 segments")],
        "transcript": "",
        "preprocessed_data": {"segments": segments},
        "segments": [],
        "analysis_results": [],
        "final_results": [],
        "max_segments": 10
    }
    
    print(f"🚀 Testing with production model: gemini-2.0-flash")
    print(f"⏱️  Expected: Higher rate limits than experimental model")
    
    start_time = time.time()
    
    try:
        result = graph.invoke(initial_state)
        end_time = time.time()
        
        processing_time = end_time - start_time
        segments_processed = len(result.get('segments', []))
        tensions_found = len(result.get('final_results', []))
        
        print(f"\n✅ SUCCESS with production model!")
        print(f"   - Segments processed: {segments_processed}")
        print(f"   - Tensions found: {tensions_found}")
        print(f"   - Processing time: {processing_time:.1f} seconds")
        print(f"   - Time per segment: {processing_time/max(segments_processed,1):.1f}s")
        
        # Show sample results
        if result.get('final_results'):
            print(f"\n📋 Sample Results:")
            for i, tension in enumerate(result['final_results'][:2]):
                print(f"   Tension {i+1}:")
                print(f"   - Concept: {tension.get('Concepts de 2nd ordre', 'N/A')}")
                print(f"   - Reformulated: {tension.get('Items de 1er ordre reformulé', 'N/A')}")
                print(f"   - Theme: {tension.get('Thème', 'N/A')}")
        
        # Rate limit assessment
        if processing_time < 60:  # Less than 1 minute for 10 segments
            print(f"\n🎉 PRODUCTION MODEL WORKING!")
            print(f"   ✅ No significant rate limiting detected")
            print(f"   ✅ Ready for larger scale testing")
            print(f"   💰 Cost estimate: ~$0.0005 for this test")
            
            # Projections
            cost_per_segment = 0.000053  # From previous estimate
            full_dataset_cost = 302 * cost_per_segment
            print(f"\n🔮 Full Dataset Projections:")
            print(f"   - 302 segments total cost: ${full_dataset_cost:.4f}")
            print(f"   - Processing time: ~{(302 * processing_time/segments_processed)/3600:.1f} hours")
            print(f"   - University budget: VERY AFFORDABLE!")
            
        else:
            print(f"\n⚠️  Still experiencing rate limits")
            print(f"   - Consider smaller batches")
            print(f"   - May need to use different model")
        
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"❌ Test failed after {processing_time:.1f}s")
        print(f"Error: {str(e)}")
        
        if "429" in str(e) or "quota" in str(e).lower():
            print(f"\n🚫 Still hitting rate limits with production model")
            print(f"   - May need to use gemini-1.5-flash instead")
            print(f"   - Or implement slower processing")
        else:
            print(f"\n🔧 Other error - check configuration")

if __name__ == "__main__":
    main()

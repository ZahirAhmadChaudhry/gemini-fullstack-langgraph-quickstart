#!/usr/bin/env python3
"""
Final speed test with parallel processing to maximize throughput.
Test how many segments we can process simultaneously within rate limits.

Run from backend directory with: uv run python test_final_speed.py
"""

import os
import sys
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent.graph import graph
from langchain_core.messages import HumanMessage

def load_test_data(file_path: str, max_segments: int = 50):
    """Load test data for final speed test."""
    
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

def process_single_segment(segment_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single segment (thread-safe)."""
    
    segment_id = segment_data.get('id', 'unknown')
    start_time = time.time()
    
    try:
        # Prepare state for single segment
        initial_state = {
            "messages": [HumanMessage(content=f"Analyze segment {segment_id}")],
            "transcript": "",
            "preprocessed_data": {"segments": [segment_data]},
            "segments": [],
            "analysis_results": [],
            "final_results": [],
            "max_segments": 1
        }
        
        # Process segment
        result = graph.invoke(initial_state)
        end_time = time.time()
        
        final_results = result.get('final_results', [])
        
        return {
            "success": True,
            "segment_id": segment_id,
            "tensions_found": len(final_results),
            "processing_time": end_time - start_time,
            "error": None,
            "result_data": final_results
        }
        
    except Exception as e:
        end_time = time.time()
        return {
            "success": False,
            "segment_id": segment_id,
            "tensions_found": 0,
            "processing_time": end_time - start_time,
            "error": str(e),
            "result_data": []
        }

def run_parallel_speed_test(segments: List[Dict], max_workers: int) -> Dict[str, Any]:
    """Run parallel speed test with specified workers."""
    
    print(f"\n🚀 PARALLEL SPEED TEST: {max_workers} workers, {len(segments)} segments")
    print(f"   Theoretical max: {max_workers * 60 / 4.2:.0f} segments/min")
    
    start_time = time.time()
    results = []
    completed_count = 0
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all segments for processing
        future_to_segment = {
            executor.submit(process_single_segment, segment): segment 
            for segment in segments
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_segment):
            segment = future_to_segment[future]
            try:
                result = future.result()
                results.append(result)
                completed_count += 1
                
                if result["success"]:
                    print(f"   ✅ {completed_count}/{len(segments)}: {result['segment_id']} - {result['tensions_found']} tensions ({result['processing_time']:.1f}s)")
                else:
                    print(f"   ❌ {completed_count}/{len(segments)}: {result['segment_id']} - {result['error']}")
                
                # Progress update every 10 segments
                if completed_count % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = completed_count / elapsed * 60
                    print(f"   📊 Progress: {completed_count}/{len(segments)} - Current rate: {rate:.1f} seg/min")
                    
            except Exception as e:
                print(f"   ❌ {segment['id']}: Exception - {e}")
                results.append({
                    "success": False,
                    "segment_id": segment['id'],
                    "tensions_found": 0,
                    "processing_time": 0,
                    "error": str(e),
                    "result_data": []
                })
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    successful_results = [r for r in results if r["success"]]
    segments_per_minute = (len(successful_results) / processing_time) * 60
    
    return {
        "max_workers": max_workers,
        "segments_tested": len(segments),
        "successful_segments": len(successful_results),
        "processing_time": processing_time,
        "segments_per_minute": segments_per_minute,
        "success_rate": len(successful_results) / len(segments),
        "results": results,
        "sample_results": [r["result_data"][0] if r["result_data"] else None for r in successful_results[:3]]
    }

def save_final_results(test_result: Dict[str, Any], output_file: str = "final_speed_test.json"):
    """Save final speed test results with sample outputs."""
    
    # Calculate cost
    successful_count = test_result["successful_segments"]
    total_tokens = successful_count * 2260  # From previous tests
    total_cost = (total_tokens / 1_000_000) * 0.105  # Combined rate
    
    output_data = {
        "test_date": datetime.now().isoformat(),
        "performance_summary": {
            "max_workers": test_result["max_workers"],
            "segments_tested": test_result["segments_tested"],
            "successful_segments": test_result["successful_segments"],
            "success_rate": test_result["success_rate"],
            "processing_time_minutes": test_result["processing_time"] / 60,
            "segments_per_minute": test_result["segments_per_minute"],
            "total_cost_usd": total_cost,
            "cost_per_segment": total_cost / max(successful_count, 1)
        },
        "full_dataset_projections": {
            "total_segments": 302,
            "projected_time_minutes": 302 / test_result["segments_per_minute"],
            "projected_cost_usd": (302 * total_cost) / max(successful_count, 1),
            "speed_improvement_vs_sequential": test_result["segments_per_minute"] / 14.1
        },
        "sample_outputs": [r for r in test_result["sample_results"] if r is not None]
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"📁 Final speed test results saved to: {output_file}")
    return output_file

def main():
    """Run final speed optimization test."""
    
    print("🏁 French Sustainability Transcript Analyzer - FINAL SPEED TEST")
    print("=" * 75)
    
    # Check prerequisites
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY not found in environment variables")
        return
    
    # Load test data
    pipeline_file = "../Table_A_ml_ready.json"
    try:
        segments = load_test_data(pipeline_file, max_segments=50)  # Test with 50 segments
        print(f"📊 Loaded {len(segments)} segments for final speed test")
    except Exception as e:
        print(f"❌ Error loading pipeline data: {e}")
        return
    
    print(f"🎯 Goal: Maximize speed while maintaining quality")
    print(f"📈 Current baseline: 14.1 segments/min (sequential)")
    print(f"🎯 Target: >30 segments/min (2x improvement)")
    print(f"⚠️  Rate limit: 2000 RPM (allows ~400 segments/min theoretical)")
    
    # Test with optimal parallel workers
    # With 4.2s per segment, theoretical max is ~14 parallel workers
    # But let's test with 10 workers for safety margin
    optimal_workers = 10
    
    try:
        print(f"\n🚀 Running final speed test with {optimal_workers} parallel workers...")
        result = run_parallel_speed_test(segments, optimal_workers)
        
        # Generate final report
        print(f"\n" + "="*75)
        print(f"🏁 FINAL SPEED TEST RESULTS")
        print(f"="*75)
        
        print(f"\n📊 Performance Summary:")
        print(f"   - Workers: {result['max_workers']}")
        print(f"   - Segments tested: {result['segments_tested']}")
        print(f"   - Successful: {result['successful_segments']}")
        print(f"   - Success rate: {result['success_rate']*100:.1f}%")
        print(f"   - Processing time: {result['processing_time']/60:.1f} minutes")
        print(f"   - Speed: {result['segments_per_minute']:.1f} segments/minute")
        
        # Speed improvement analysis
        baseline_speed = 14.1
        speed_improvement = result['segments_per_minute'] / baseline_speed
        print(f"\n🚀 Speed Improvement:")
        print(f"   - Baseline (sequential): {baseline_speed} segments/min")
        print(f"   - Optimized (parallel): {result['segments_per_minute']:.1f} segments/min")
        print(f"   - Improvement factor: {speed_improvement:.1f}x")
        print(f"   - Time reduction: {(1 - 1/speed_improvement)*100:.1f}%")
        
        # Full dataset projections
        total_segments = 302
        optimized_time = total_segments / result['segments_per_minute']
        baseline_time = 21.5  # From previous tests
        
        print(f"\n🔮 Full Dataset Projections ({total_segments} segments):")
        print(f"   - Optimized time: {optimized_time:.1f} minutes ({optimized_time/60:.1f} hours)")
        print(f"   - Baseline time: {baseline_time:.1f} minutes")
        print(f"   - Time saved: {baseline_time - optimized_time:.1f} minutes")
        
        # Cost analysis
        successful_count = result['successful_segments']
        total_tokens = successful_count * 2260
        total_cost = (total_tokens / 1_000_000) * 0.105
        full_dataset_cost = (302 * total_cost) / successful_count
        
        print(f"\n💰 Cost Analysis:")
        print(f"   - Cost for test: ${total_cost:.4f}")
        print(f"   - Cost per segment: ${total_cost/successful_count:.6f}")
        print(f"   - Full dataset cost: ${full_dataset_cost:.4f}")
        
        # Final recommendations
        print(f"\n💡 Final Recommendations:")
        if speed_improvement >= 2.0:
            print(f"   🎉 EXCELLENT: {speed_improvement:.1f}x speed improvement achieved!")
            print(f"   ✅ Can process full dataset in {optimized_time:.1f} minutes")
        elif speed_improvement >= 1.5:
            print(f"   ✅ GOOD: {speed_improvement:.1f}x speed improvement")
            print(f"   ✅ Significant time savings achieved")
        else:
            print(f"   ⚠️  MODERATE: {speed_improvement:.1f}x improvement")
            print(f"   📝 Consider different optimization strategies")
        
        if result['success_rate'] >= 0.95:
            print(f"   ✅ High reliability maintained ({result['success_rate']*100:.1f}% success)")
        else:
            print(f"   ⚠️  Success rate: {result['success_rate']*100:.1f}% - monitor for issues")
        
        # Save results
        save_final_results(result)
        
        print(f"\n🎯 CONCLUSION:")
        if optimized_time < 15:
            print(f"   🚀 PRODUCTION READY: Full dataset in under 15 minutes!")
        elif optimized_time < 20:
            print(f"   ✅ GOOD PERFORMANCE: Full dataset in under 20 minutes")
        else:
            print(f"   📝 BASELINE PERFORMANCE: Consider further optimization")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Final speed test interrupted")
    except Exception as e:
        print(f"❌ Final speed test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

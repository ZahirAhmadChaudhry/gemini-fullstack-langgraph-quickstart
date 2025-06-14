#!/usr/bin/env python3
"""
Parallel processing optimization - process multiple segments simultaneously.
This should dramatically improve speed by utilizing our 2000 RPM limit effectively.

Run from backend directory with: uv run python test_parallel_processing.py
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
import threading

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent.graph import graph
from langchain_core.messages import HumanMessage

@dataclass
class ParallelTestResult:
    """Results from parallel processing test."""
    config_name: str
    segments_tested: int
    processing_time_seconds: float
    segments_per_minute: float
    success_rate: float
    parallel_workers: int
    sample_results: List[Dict[str, Any]]

def load_test_data(file_path: str, max_segments: int = 30):
    """Load test data for parallel processing."""
    
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
    """Process a single segment (for parallel execution)."""
    
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

def run_parallel_test(segments: List[Dict], max_workers: int, config_name: str) -> ParallelTestResult:
    """Run parallel processing test with specified number of workers."""
    
    print(f"\n🚀 Testing {config_name} with {max_workers} parallel workers...")
    print(f"   - Segments: {len(segments)}")
    print(f"   - Theoretical max: {max_workers * 60 / 4.2:.0f} segments/min")
    
    start_time = time.time()
    results = []
    
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
                
                if result["success"]:
                    print(f"   ✅ {result['segment_id']}: {result['tensions_found']} tensions in {result['processing_time']:.1f}s")
                else:
                    print(f"   ❌ {result['segment_id']}: {result['error']}")
                    
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
    
    # Get sample results
    sample_results = []
    for result in successful_results[:3]:  # First 3 successful results
        if result["result_data"]:
            sample_results.extend(result["result_data"][:1])  # First tension from each
    
    return ParallelTestResult(
        config_name=config_name,
        segments_tested=len(segments),
        processing_time_seconds=processing_time,
        segments_per_minute=segments_per_minute,
        success_rate=len(successful_results) / len(segments),
        parallel_workers=max_workers,
        sample_results=sample_results
    )

def save_parallel_outputs(results: List[ParallelTestResult], output_file: str = "parallel_test_results.json"):
    """Save parallel test results and sample outputs."""
    
    output_data = {
        "test_date": datetime.now().isoformat(),
        "test_summary": {
            "configurations_tested": len(results),
            "best_speed": max(r.segments_per_minute for r in results if r.segments_per_minute > 0),
            "best_config": max(results, key=lambda r: r.segments_per_minute).config_name if results else "None",
            "speed_improvement": "TBD"
        },
        "results": {}
    }
    
    baseline_speed = 14.1  # From previous sequential test
    
    for result in results:
        speed_improvement = result.segments_per_minute / baseline_speed if baseline_speed > 0 else 0
        
        output_data["results"][result.config_name] = {
            "performance": {
                "segments_per_minute": result.segments_per_minute,
                "success_rate": result.success_rate,
                "parallel_workers": result.parallel_workers,
                "processing_time_seconds": result.processing_time_seconds,
                "speed_improvement_factor": speed_improvement
            },
            "sample_tensions": result.sample_results
        }
    
    # Update summary
    if results:
        best_result = max(results, key=lambda r: r.segments_per_minute)
        output_data["test_summary"]["speed_improvement"] = f"{best_result.segments_per_minute / baseline_speed:.1f}x faster"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"📁 Parallel test results saved to: {output_file}")
    return output_file

def main():
    """Run parallel processing optimization tests."""
    
    print("🚀 French Sustainability Transcript Analyzer - PARALLEL PROCESSING")
    print("=" * 75)
    
    # Check prerequisites
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY not found in environment variables")
        return
    
    # Load test data
    pipeline_file = "../Table_A_ml_ready.json"
    try:
        segments = load_test_data(pipeline_file, max_segments=30)  # Test with 30 segments
        print(f"📊 Loaded {len(segments)} segments for parallel testing")
    except Exception as e:
        print(f"❌ Error loading pipeline data: {e}")
        return
    
    # Define parallel configurations
    # With 2000 RPM limit and 5 calls per segment = 400 segments/min theoretical max
    # But LLM response time is ~4.2s, so realistic max is ~14 parallel workers
    parallel_configs = [
        (2, "Parallel_2_Workers"),
        (4, "Parallel_4_Workers"), 
        (8, "Parallel_8_Workers"),
        (12, "Parallel_12_Workers"),
    ]
    
    print(f"🧪 Testing {len(parallel_configs)} parallel configurations...")
    print(f"⚠️  Note: 2000 RPM limit allows ~400 segments/min theoretical")
    print(f"⚠️  But LLM response time (~4.2s) limits practical parallelism")
    
    # Run parallel tests
    results = []
    for workers, name in parallel_configs:
        try:
            result = run_parallel_test(segments, workers, name)
            results.append(result)
            
            print(f"\n✅ {name} completed:")
            print(f"   - Speed: {result.segments_per_minute:.1f} segments/min")
            print(f"   - Success rate: {result.success_rate*100:.1f}%")
            print(f"   - Total time: {result.processing_time_seconds:.1f}s")
            
            # Brief pause between tests
            time.sleep(10)
            
        except KeyboardInterrupt:
            print(f"\n⏹️  Parallel testing interrupted")
            break
        except Exception as e:
            print(f"   ❌ {name} failed: {e}")
    
    # Generate parallel optimization report
    if results:
        generate_parallel_report(results, segments)
        save_parallel_outputs(results)

def generate_parallel_report(results: List[ParallelTestResult], segments: List[Dict]):
    """Generate comprehensive parallel processing report."""
    
    print(f"\n" + "="*75)
    print(f"🚀 PARALLEL PROCESSING RESULTS")
    print(f"="*75)
    
    # Results table
    print(f"\n📊 Parallel Test Results:")
    print(f"-" * 75)
    print(f"{'Config':<20} {'Workers':<8} {'Seg/Min':<10} {'Success%':<10} {'Speedup':<10}")
    print(f"-" * 75)
    
    baseline_speed = 14.1  # From sequential test
    successful_results = [r for r in results if r.segments_per_minute > 0]
    
    for result in results:
        if result.segments_per_minute > 0:
            speedup = result.segments_per_minute / baseline_speed
            print(f"{result.config_name:<20} {result.parallel_workers:<8} "
                  f"{result.segments_per_minute:<10.1f} {result.success_rate*100:<10.1f} "
                  f"{speedup:<10.1f}x")
        else:
            print(f"{result.config_name:<20} {result.parallel_workers:<8} "
                  f"{'FAILED':<10} {'--':<10} {'--':<10}")
    
    if successful_results:
        # Find best configuration
        best_result = max(successful_results, key=lambda r: r.segments_per_minute)
        speedup = best_result.segments_per_minute / baseline_speed
        
        print(f"\n🏆 Best Parallel Configuration: {best_result.config_name}")
        print(f"   - Workers: {best_result.parallel_workers}")
        print(f"   - Speed: {best_result.segments_per_minute:.1f} segments/minute")
        print(f"   - Speedup: {speedup:.1f}x faster than sequential")
        print(f"   - Success rate: {best_result.success_rate*100:.1f}%")
        print(f"   - Processing time: {best_result.processing_time_seconds:.1f}s")
        
        # Full dataset projections with parallel processing
        total_segments = 302
        parallel_time_minutes = total_segments / best_result.segments_per_minute
        
        print(f"\n🔮 Optimized Full Dataset Projections ({total_segments} segments):")
        print(f"   - Processing time: {parallel_time_minutes:.1f} minutes ({parallel_time_minutes/60:.1f} hours)")
        print(f"   - Time reduction: {(1 - parallel_time_minutes/21.5)*100:.1f}% vs sequential")
        print(f"   - Segments per hour: {best_result.segments_per_minute * 60:.0f}")
        
        # Efficiency analysis
        theoretical_max = best_result.parallel_workers * (60 / 4.2)  # 4.2s per segment
        efficiency = (best_result.segments_per_minute / theoretical_max) * 100
        
        print(f"\n📈 Parallel Efficiency Analysis:")
        print(f"   - Theoretical max: {theoretical_max:.1f} segments/min")
        print(f"   - Actual performance: {best_result.segments_per_minute:.1f} segments/min")
        print(f"   - Parallel efficiency: {efficiency:.1f}%")
        
        # Recommendations
        print(f"\n💡 Parallel Processing Recommendations:")
        if efficiency > 80:
            print(f"   ✅ Excellent parallel efficiency - optimal configuration")
        elif efficiency > 60:
            print(f"   ✅ Good parallel efficiency - consider slight optimization")
        else:
            print(f"   ⚠️  Lower efficiency - may hit rate limits or resource constraints")
        
        if best_result.success_rate >= 0.95:
            print(f"   ✅ High reliability maintained with parallel processing")
        else:
            print(f"   ⚠️  Success rate dropped - balance speed vs reliability")
        
        if parallel_time_minutes < 10:
            print(f"   🚀 EXCELLENT: Full dataset in under 10 minutes!")
        elif parallel_time_minutes < 15:
            print(f"   ✅ GOOD: Full dataset in under 15 minutes")
        else:
            print(f"   ⚠️  Moderate improvement - consider more workers")
    
    else:
        print(f"\n❌ No successful parallel configurations")
        print(f"   - All tests failed - likely hitting rate limits")
        print(f"   - Consider fewer parallel workers")

if __name__ == "__main__":
    main()

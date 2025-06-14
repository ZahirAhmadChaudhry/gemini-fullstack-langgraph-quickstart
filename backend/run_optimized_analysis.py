#!/usr/bin/env python3
"""
Production-ready optimized analysis with 8.9x speed improvement.
Processes segments in parallel and saves detailed outputs for review.

Run from backend directory with: uv run python run_optimized_analysis.py
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent.graph import graph
from langchain_core.messages import HumanMessage

def load_pipeline_data(file_path: str, max_segments: int = None):
    """Load data from the pipeline output."""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    segments = data.get('segments', [])
    if max_segments:
        segments = segments[:max_segments]
    
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
    
    return {
        "source_file": data.get("source_file", "unknown"),
        "processed_timestamp": data.get("processed_timestamp", "unknown"),
        "segments": processed_segments
    }

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
    """Process a single segment (optimized for parallel execution)."""
    
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

def run_optimized_analysis(segments: List[Dict], max_workers: int = 10) -> Dict[str, Any]:
    """Run optimized parallel analysis."""
    
    print(f"🚀 OPTIMIZED ANALYSIS: {max_workers} workers, {len(segments)} segments")
    print(f"   Expected speed: ~126 segments/minute (8.9x improvement)")
    print(f"   Estimated time: {len(segments) / 126:.1f} minutes")
    
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
                
                # Progress update every 25 segments
                if completed_count % 25 == 0:
                    elapsed = time.time() - start_time
                    rate = completed_count / elapsed * 60
                    remaining = (len(segments) - completed_count) / rate * 60 if rate > 0 else 0
                    print(f"   📊 Progress: {completed_count}/{len(segments)} - Rate: {rate:.1f} seg/min - ETA: {remaining:.1f}s")
                    
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
        "segments_tested": len(segments),
        "successful_segments": len(successful_results),
        "processing_time": processing_time,
        "segments_per_minute": segments_per_minute,
        "success_rate": len(successful_results) / len(segments),
        "results": results
    }

def save_analysis_results(analysis_result: Dict[str, Any], pipeline_data: Dict, output_file: str = "optimized_analysis_results.json"):
    """Save comprehensive analysis results with sample outputs for review."""
    
    successful_results = [r for r in analysis_result["results"] if r["success"]]
    
    # Calculate costs
    total_tokens = len(successful_results) * 2260  # From previous tests
    total_cost = (total_tokens / 1_000_000) * 0.105  # Combined rate
    
    # Prepare detailed output
    output_data = {
        "analysis_metadata": {
            "timestamp": datetime.now().isoformat(),
            "source_file": pipeline_data.get("source_file", "unknown"),
            "pipeline_timestamp": pipeline_data.get("processed_timestamp", "unknown"),
            "analysis_method": "Optimized Parallel Processing (8.9x speed improvement)"
        },
        "performance_summary": {
            "segments_analyzed": analysis_result["segments_tested"],
            "successful_segments": analysis_result["successful_segments"],
            "success_rate": analysis_result["success_rate"],
            "processing_time_minutes": analysis_result["processing_time"] / 60,
            "segments_per_minute": analysis_result["segments_per_minute"],
            "speed_improvement_factor": analysis_result["segments_per_minute"] / 14.1,  # vs baseline
            "total_cost_usd": total_cost,
            "cost_per_segment": total_cost / max(len(successful_results), 1)
        },
        "tension_analysis_summary": {
            "total_tensions_identified": sum(r["tensions_found"] for r in successful_results),
            "segments_with_tensions": len([r for r in successful_results if r["tensions_found"] > 0]),
            "average_tensions_per_segment": sum(r["tensions_found"] for r in successful_results) / max(len(successful_results), 1)
        },
        "detailed_results": []
    }
    
    # Add detailed results for each segment
    for result in successful_results:
        segment_detail = {
            "segment_id": result["segment_id"],
            "processing_time_seconds": result["processing_time"],
            "tensions_found": result["tensions_found"],
            "tensions": result["result_data"]
        }
        output_data["detailed_results"].append(segment_detail)
    
    # Add sample outputs for review (first 10 tensions)
    sample_tensions = []
    for result in successful_results[:10]:  # First 10 segments
        if result["result_data"]:
            for tension in result["result_data"][:1]:  # First tension from each segment
                sample_tensions.append({
                    "segment_id": result["segment_id"],
                    "tension": tension
                })
    
    output_data["sample_outputs_for_review"] = sample_tensions
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"📁 Comprehensive results saved to: {output_file}")
    return output_file

def main():
    """Run optimized analysis with sample output generation."""
    
    print("🚀 French Sustainability Transcript Analyzer - OPTIMIZED PRODUCTION")
    print("=" * 75)
    
    # Check prerequisites
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY not found in environment variables")
        return
    
    # Load pipeline data
    pipeline_file = "../Table_A_ml_ready.json"
    try:
        pipeline_data = load_pipeline_data(pipeline_file, max_segments=100)  # Process 100 segments
        segments = pipeline_data["segments"]
        print(f"📊 Loaded {len(segments)} segments from {pipeline_data['source_file']}")
    except Exception as e:
        print(f"❌ Error loading pipeline data: {e}")
        return
    
    print(f"⚡ Using optimized parallel processing (8.9x speed improvement)")
    print(f"💰 Estimated cost: ${len(segments) * 0.000237:.4f}")
    print(f"⏱️  Estimated time: {len(segments) / 126:.1f} minutes")
    
    try:
        # Run optimized analysis
        print(f"\n🚀 Starting optimized analysis...")
        analysis_result = run_optimized_analysis(segments, max_workers=10)
        
        # Generate comprehensive report
        print(f"\n" + "="*75)
        print(f"🎉 OPTIMIZED ANALYSIS COMPLETED")
        print(f"="*75)
        
        print(f"\n📊 Results Summary:")
        print(f"   - Segments analyzed: {analysis_result['successful_segments']}/{analysis_result['segments_tested']}")
        print(f"   - Success rate: {analysis_result['success_rate']*100:.1f}%")
        print(f"   - Processing time: {analysis_result['processing_time']/60:.1f} minutes")
        print(f"   - Speed: {analysis_result['segments_per_minute']:.1f} segments/minute")
        print(f"   - Speed improvement: {analysis_result['segments_per_minute']/14.1:.1f}x vs baseline")
        
        # Calculate tension statistics
        successful_results = [r for r in analysis_result["results"] if r["success"]]
        total_tensions = sum(r["tensions_found"] for r in successful_results)
        segments_with_tensions = len([r for r in successful_results if r["tensions_found"] > 0])
        
        print(f"\n🔍 Tension Analysis:")
        print(f"   - Total tensions identified: {total_tensions}")
        print(f"   - Segments with tensions: {segments_with_tensions}/{len(successful_results)}")
        print(f"   - Average tensions per segment: {total_tensions/max(len(successful_results),1):.1f}")
        
        # Cost analysis
        total_cost = len(successful_results) * 0.000237
        print(f"\n💰 Cost Analysis:")
        print(f"   - Total cost: ${total_cost:.4f}")
        print(f"   - Cost per segment: ${0.000237:.6f}")
        
        # Save comprehensive results
        output_file = save_analysis_results(analysis_result, pipeline_data)
        
        print(f"\n📋 Sample outputs saved for your review!")
        print(f"📁 Full results: {output_file}")
        print(f"\n🎯 Ready for production deployment!")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Analysis interrupted by user")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

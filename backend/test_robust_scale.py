#!/usr/bin/env python3
"""
Robust scale testing with proper rate limiting and batch processing.
Designed to work within free tier constraints.

Run from backend directory with: uv run python test_robust_scale.py
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

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent.graph import graph
from langchain_core.messages import HumanMessage

@dataclass
class RobustTestResult:
    """Container for robust test results."""
    segments_tested: int
    tensions_found: int
    processing_time: float
    successful_segments: int
    failed_segments: int
    rate_limit_hits: int
    average_time_per_segment: float
    success_rate: float

class RateLimitedProcessor:
    """Processor that respects API rate limits."""
    
    def __init__(self, requests_per_minute: int = 8):  # Conservative: 8/10 limit
        self.requests_per_minute = requests_per_minute
        self.min_delay = 60.0 / requests_per_minute  # Minimum delay between requests
        self.last_request_time = 0
        self.request_count = 0
        self.rate_limit_hits = 0
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_delay:
            wait_time = self.min_delay - time_since_last
            print(f"   ⏳ Rate limiting: waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def handle_rate_limit_error(self, error_msg: str):
        """Handle rate limit errors with exponential backoff."""
        self.rate_limit_hits += 1
        
        # Extract retry delay from error message if available
        retry_delay = 60  # Default 1 minute
        if "retry_delay" in error_msg and "seconds:" in error_msg:
            try:
                delay_part = error_msg.split("seconds:")[1].split("}")[0].strip()
                retry_delay = int(delay_part)
            except:
                pass
        
        print(f"   🚫 Rate limit hit #{self.rate_limit_hits}. Waiting {retry_delay}s...")
        time.sleep(retry_delay + 5)  # Add 5s buffer

def load_pipeline_data(file_path: str, max_segments: int = None):
    """Load and prepare data from the pipeline output."""
    
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

def process_single_segment(segment_data: dict, processor: RateLimitedProcessor) -> dict:
    """Process a single segment with rate limiting."""
    
    print(f"   📝 Processing {segment_data['id']}...")
    
    # Wait for rate limiting
    processor.wait_if_needed()
    
    # Prepare state for single segment
    initial_state = {
        "messages": [HumanMessage(content=f"Analyze segment {segment_data['id']}")],
        "transcript": "",
        "preprocessed_data": {
            "segments": [segment_data]
        },
        "segments": [],
        "analysis_results": [],
        "final_results": [],
        "max_segments": 1
    }
    
    start_time = time.time()
    
    try:
        result = graph.invoke(initial_state)
        end_time = time.time()
        
        return {
            "success": True,
            "segment_id": segment_data['id'],
            "tensions_found": len(result.get('final_results', [])),
            "processing_time": end_time - start_time,
            "error": None,
            "result": result.get('final_results', [])
        }
        
    except Exception as e:
        end_time = time.time()
        error_msg = str(e)
        
        # Handle rate limit errors
        if "429" in error_msg or "quota" in error_msg.lower():
            processor.handle_rate_limit_error(error_msg)
            return {
                "success": False,
                "segment_id": segment_data['id'],
                "tensions_found": 0,
                "processing_time": end_time - start_time,
                "error": "rate_limit",
                "result": []
            }
        else:
            return {
                "success": False,
                "segment_id": segment_data['id'],
                "tensions_found": 0,
                "processing_time": end_time - start_time,
                "error": error_msg,
                "result": []
            }

def run_robust_scale_test(segments_count: int, pipeline_data: dict) -> RobustTestResult:
    """Run robust scale test with proper rate limiting."""
    
    print(f"\n🛡️  Running ROBUST scale test with {segments_count} segments...")
    print(f"   📊 Rate limit: 8 requests/minute (conservative)")
    print(f"   ⏱️  Estimated time: {(segments_count * 5 * 60) / 8 / 60:.1f} minutes")
    
    processor = RateLimitedProcessor(requests_per_minute=8)
    test_segments = pipeline_data["segments"][:segments_count]
    
    results = []
    successful_segments = 0
    total_tensions = 0
    start_time = time.time()
    
    for i, segment in enumerate(test_segments, 1):
        print(f"\n   📍 Segment {i}/{segments_count}")
        
        result = process_single_segment(segment, processor)
        results.append(result)
        
        if result["success"]:
            successful_segments += 1
            total_tensions += result["tensions_found"]
            print(f"      ✅ Success: {result['tensions_found']} tensions in {result['processing_time']:.1f}s")
        else:
            print(f"      ❌ Failed: {result['error']}")
        
        # Progress update
        if i % 5 == 0 or i == segments_count:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = (segments_count - i) * avg_time
            print(f"   📈 Progress: {i}/{segments_count} ({i/segments_count*100:.1f}%) - "
                  f"ETA: {remaining/60:.1f}min")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return RobustTestResult(
        segments_tested=segments_count,
        tensions_found=total_tensions,
        processing_time=total_time,
        successful_segments=successful_segments,
        failed_segments=segments_count - successful_segments,
        rate_limit_hits=processor.rate_limit_hits,
        average_time_per_segment=total_time / segments_count,
        success_rate=successful_segments / segments_count
    )

def main():
    """Run robust scale testing."""
    
    print("🛡️  French Sustainability Transcript Analyzer - ROBUST Scale Testing")
    print("=" * 75)
    
    # Check prerequisites
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY not found in environment variables")
        return
    
    # Load pipeline data
    pipeline_file = "../Table_A_ml_ready.json"
    try:
        full_pipeline_data = load_pipeline_data(pipeline_file)
        total_available = len(full_pipeline_data["segments"])
        print(f"📊 Total segments available: {total_available}")
    except Exception as e:
        print(f"❌ Error loading pipeline data: {e}")
        return
    
    # Conservative test scales for free tier
    test_scales = [3, 5, 8]  # Start very conservative
    
    print(f"🧪 Running robust tests with scales: {test_scales}")
    print("⏳ This will take time due to rate limiting...")
    
    # Run tests
    results = []
    for scale in test_scales:
        try:
            result = run_robust_scale_test(scale, full_pipeline_data)
            results.append(result)
            
            print(f"\n✅ COMPLETED {scale} segments:")
            print(f"   - Success rate: {result.success_rate*100:.1f}%")
            print(f"   - Tensions found: {result.tensions_found}")
            print(f"   - Rate limit hits: {result.rate_limit_hits}")
            print(f"   - Total time: {result.processing_time/60:.1f} minutes")
            
        except KeyboardInterrupt:
            print(f"\n⏹️  Test interrupted by user")
            break
        except Exception as e:
            print(f"❌ Test with {scale} segments failed: {e}")
    
    # Generate report
    if results:
        generate_robust_report(results, full_pipeline_data)

def generate_robust_report(results: List[RobustTestResult], pipeline_data: dict):
    """Generate robust testing report."""
    
    print("\n" + "="*75)
    print("🛡️  ROBUST SCALE TESTING REPORT")
    print("="*75)
    
    # Summary table
    print("\n📋 Robust Test Results:")
    print("-" * 75)
    print(f"{'Segments':<10} {'Success%':<10} {'Tensions':<10} {'Time(min)':<12} {'Rate Hits':<10}")
    print("-" * 75)
    
    for result in results:
        print(f"{result.segments_tested:<10} {result.success_rate*100:<10.1f} "
              f"{result.tensions_found:<10} {result.processing_time/60:<12.1f} "
              f"{result.rate_limit_hits:<10}")
    
    # Analysis
    if results:
        best_result = max(results, key=lambda r: r.segments_tested)
        
        print(f"\n🔍 Analysis:")
        print(f"   - Largest successful test: {best_result.segments_tested} segments")
        print(f"   - Best success rate: {max(r.success_rate for r in results)*100:.1f}%")
        print(f"   - Average time per segment: {best_result.average_time_per_segment:.1f}s")
        print(f"   - Rate limiting effectiveness: {best_result.rate_limit_hits} hits")
        
        # Realistic projections
        total_segments = len(pipeline_data["segments"])
        if best_result.success_rate > 0.8:
            realistic_time = (total_segments * best_result.average_time_per_segment) / 3600
            print(f"\n🎯 Realistic Full Dataset Projections ({total_segments} segments):")
            print(f"   - Estimated processing time: {realistic_time:.1f} hours")
            print(f"   - Recommended batch size: {best_result.segments_tested}")
            print(f"   - Number of batches needed: {total_segments // best_result.segments_tested}")
        
        print(f"\n💡 Production Recommendations:")
        if best_result.success_rate > 0.9:
            print("   ✅ System is robust and production-ready")
            print("   ✅ Rate limiting strategy is effective")
            print("   📈 Can scale to full dataset with batch processing")
        else:
            print("   ⚠️  Need to improve error handling")
            print("   📉 Consider smaller batch sizes")

if __name__ == "__main__":
    main()

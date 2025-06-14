#!/usr/bin/env python3
"""
Speed optimization test - push the limits of processing speed.
Test different configurations to find optimal speed/quality balance.

Run from backend directory with: uv run python test_speed_optimization.py
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

from agent.batch_processor import ProductionBatchProcessor, BatchConfig

@dataclass
class SpeedTestResult:
    """Results from speed optimization test."""
    config_name: str
    segments_tested: int
    processing_time_seconds: float
    segments_per_minute: float
    success_rate: float
    total_cost: float
    cost_per_segment: float
    rate_limit_hits: int
    sample_results: List[Dict[str, Any]]

def load_test_data(file_path: str, max_segments: int = 20):
    """Load test data for speed optimization."""
    
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

def run_speed_test(config: BatchConfig, segments: List[Dict], config_name: str) -> SpeedTestResult:
    """Run speed test with specific configuration."""
    
    print(f"\n🏃‍♂️ Testing {config_name}...")
    print(f"   - Rate limit: {config.max_requests_per_minute} RPM")
    print(f"   - Segments: {len(segments)}")
    
    processor = ProductionBatchProcessor(config)
    start_time = time.time()
    
    try:
        progress = processor.process_batch(segments, resume=False)
        end_time = time.time()
        
        processing_time = end_time - start_time
        segments_per_minute = (progress.successful_segments / processing_time) * 60
        
        # Calculate cost (rough estimate)
        total_tokens = progress.successful_segments * 2260  # From previous test
        total_cost = (total_tokens / 1_000_000) * 0.105  # Combined rate
        
        # Get sample results
        sample_results = []
        for result in progress.results[:3]:  # First 3 results
            if result.success and result.result_data:
                sample_results.extend(result.result_data[:1])  # First tension from each
        
        return SpeedTestResult(
            config_name=config_name,
            segments_tested=len(segments),
            processing_time_seconds=processing_time,
            segments_per_minute=segments_per_minute,
            success_rate=progress.successful_segments / progress.total_segments,
            total_cost=total_cost,
            cost_per_segment=total_cost / max(progress.successful_segments, 1),
            rate_limit_hits=processor.rate_limit_hits,
            sample_results=sample_results
        )
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return SpeedTestResult(
            config_name=config_name,
            segments_tested=len(segments),
            processing_time_seconds=0,
            segments_per_minute=0,
            success_rate=0,
            total_cost=0,
            cost_per_segment=0,
            rate_limit_hits=0,
            sample_results=[]
        )

def save_sample_outputs(results: List[SpeedTestResult], output_file: str = "speed_test_samples.json"):
    """Save sample outputs for review."""
    
    sample_data = {
        "test_date": datetime.now().isoformat(),
        "test_summary": {
            "configurations_tested": len(results),
            "best_speed": max(r.segments_per_minute for r in results if r.segments_per_minute > 0),
            "best_config": max(results, key=lambda r: r.segments_per_minute).config_name if results else "None"
        },
        "sample_outputs": {}
    }
    
    for result in results:
        if result.sample_results:
            sample_data["sample_outputs"][result.config_name] = {
                "performance": {
                    "segments_per_minute": result.segments_per_minute,
                    "success_rate": result.success_rate,
                    "cost_per_segment": result.cost_per_segment
                },
                "sample_tensions": result.sample_results
            }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"📁 Sample outputs saved to: {output_file}")
    return output_file

def main():
    """Run comprehensive speed optimization tests."""
    
    print("🏃‍♂️ French Sustainability Transcript Analyzer - SPEED OPTIMIZATION")
    print("=" * 75)
    
    # Check prerequisites
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY not found in environment variables")
        return
    
    # Load test data
    pipeline_file = "../Table_A_ml_ready.json"
    try:
        segments = load_test_data(pipeline_file, max_segments=20)  # Test with 20 segments
        print(f"📊 Loaded {len(segments)} segments for speed testing")
    except Exception as e:
        print(f"❌ Error loading pipeline data: {e}")
        return
    
    # Define speed test configurations
    speed_configs = [
        # Current baseline
        (BatchConfig(max_requests_per_minute=300, retry_attempts=2, retry_delay=5, progress_file="speed_test_300.json"), "Baseline_300RPM"),
        
        # Aggressive speed tests
        (BatchConfig(max_requests_per_minute=600, retry_attempts=2, retry_delay=3, progress_file="speed_test_600.json"), "Fast_600RPM"),
        
        (BatchConfig(max_requests_per_minute=1000, retry_attempts=2, retry_delay=2, progress_file="speed_test_1000.json"), "Faster_1000RPM"),
        
        (BatchConfig(max_requests_per_minute=1500, retry_attempts=1, retry_delay=1, progress_file="speed_test_1500.json"), "Maximum_1500RPM"),
    ]
    
    print(f"🧪 Testing {len(speed_configs)} speed configurations...")
    print(f"⚠️  Note: We have 2000 RPM limit, testing up to 1500 RPM")
    
    # Run speed tests
    results = []
    for config, name in speed_configs:
        try:
            result = run_speed_test(config, segments, name)
            results.append(result)
            
            if result.segments_per_minute > 0:
                print(f"   ✅ {name}: {result.segments_per_minute:.1f} segments/min "
                      f"({result.success_rate*100:.1f}% success, {result.rate_limit_hits} rate hits)")
            else:
                print(f"   ❌ {name}: Failed")
            
            # Brief pause between tests
            time.sleep(5)
            
        except KeyboardInterrupt:
            print(f"\n⏹️  Speed testing interrupted")
            break
        except Exception as e:
            print(f"   ❌ {name} failed: {e}")
    
    # Generate speed optimization report
    if results:
        generate_speed_report(results, segments)
        save_sample_outputs(results)

def generate_speed_report(results: List[SpeedTestResult], segments: List[Dict]):
    """Generate comprehensive speed optimization report."""
    
    print(f"\n" + "="*75)
    print(f"🏃‍♂️ SPEED OPTIMIZATION RESULTS")
    print(f"="*75)
    
    # Results table
    print(f"\n📊 Speed Test Results:")
    print(f"-" * 75)
    print(f"{'Config':<15} {'Seg/Min':<10} {'Success%':<10} {'Cost/Seg':<12} {'Rate Hits':<10}")
    print(f"-" * 75)
    
    successful_results = [r for r in results if r.segments_per_minute > 0]
    
    for result in results:
        if result.segments_per_minute > 0:
            print(f"{result.config_name:<15} {result.segments_per_minute:<10.1f} "
                  f"{result.success_rate*100:<10.1f} ${result.cost_per_segment:<11.6f} "
                  f"{result.rate_limit_hits:<10}")
        else:
            print(f"{result.config_name:<15} {'FAILED':<10} {'--':<10} {'--':<12} {'--':<10}")
    
    if successful_results:
        # Find best configuration
        best_result = max(successful_results, key=lambda r: r.segments_per_minute)
        
        print(f"\n🏆 Best Configuration: {best_result.config_name}")
        print(f"   - Speed: {best_result.segments_per_minute:.1f} segments/minute")
        print(f"   - Success rate: {best_result.success_rate*100:.1f}%")
        print(f"   - Cost per segment: ${best_result.cost_per_segment:.6f}")
        print(f"   - Rate limit hits: {best_result.rate_limit_hits}")
        
        # Speed improvement analysis
        baseline = next((r for r in results if "Baseline" in r.config_name), None)
        if baseline and baseline.segments_per_minute > 0:
            speed_improvement = best_result.segments_per_minute / baseline.segments_per_minute
            print(f"\n📈 Speed Improvement:")
            print(f"   - {speed_improvement:.1f}x faster than baseline")
            print(f"   - Time reduction: {(1 - 1/speed_improvement)*100:.1f}%")
        
        # Full dataset projections with optimized speed
        total_segments = 302
        optimized_time_minutes = total_segments / best_result.segments_per_minute
        optimized_cost = total_segments * best_result.cost_per_segment
        
        print(f"\n🔮 Optimized Full Dataset Projections ({total_segments} segments):")
        print(f"   - Processing time: {optimized_time_minutes:.1f} minutes ({optimized_time_minutes/60:.1f} hours)")
        print(f"   - Total cost: ${optimized_cost:.4f}")
        print(f"   - Segments per hour: {best_result.segments_per_minute * 60:.0f}")
        
        # Recommendations
        print(f"\n💡 Speed Optimization Recommendations:")
        if best_result.rate_limit_hits == 0:
            print(f"   ✅ Optimal configuration found with no rate limiting")
            print(f"   🚀 Can process full dataset in {optimized_time_minutes:.1f} minutes")
        else:
            print(f"   ⚠️  Some rate limiting detected - consider slightly lower rate")
        
        if best_result.success_rate >= 0.95:
            print(f"   ✅ High reliability maintained at optimized speed")
        else:
            print(f"   ⚠️  Success rate dropped - balance speed vs reliability")
    
    else:
        print(f"\n❌ No successful speed configurations")
        print(f"   - All tests failed or hit rate limits")
        print(f"   - Consider more conservative settings")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Comprehensive scale testing for the French sustainability transcript analyzer.
Tests performance, accuracy, and robustness with larger datasets.

Run from backend directory with: uv run python test_scale_validation.py
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent.graph import graph
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

@dataclass
class TestResult:
    """Container for test results."""
    segments_tested: int
    tensions_found: int
    processing_time: float
    total_tokens: int
    input_tokens: int
    output_tokens: int
    api_calls: int
    errors: List[str]
    success_rate: float
    cost_estimate: float

class ScaleTestTracker(BaseCallbackHandler):
    """Enhanced callback handler for scale testing."""
    
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
        self.call_details = []
        self.errors = []
        self.start_time = None
        self.end_time = None
    
    def start_test(self):
        """Mark the start of testing."""
        self.start_time = time.time()
    
    def end_test(self):
        """Mark the end of testing."""
        self.end_time = time.time()
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        """Called when LLM starts running."""
        pass
    
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
        
        # Method 3: Estimate based on text length
        else:
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
            'timestamp': time.time()
        })
    
    def on_llm_error(self, error: Exception, **kwargs: Any) -> Any:
        """Called when LLM encounters an error."""
        self.errors.append(str(error))
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive test summary."""
        total_tokens = self.total_input_tokens + self.total_output_tokens
        processing_time = (self.end_time - self.start_time) if self.start_time and self.end_time else 0
        
        return {
            'total_calls': self.total_calls,
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': total_tokens,
            'processing_time': processing_time,
            'errors': self.errors,
            'error_count': len(self.errors),
            'success_rate': (self.total_calls - len(self.errors)) / max(self.total_calls, 1),
            'call_details': self.call_details,
            'tokens_per_second': total_tokens / max(processing_time, 1),
            'calls_per_minute': (self.total_calls * 60) / max(processing_time, 1)
        }

def load_pipeline_data(file_path: str, max_segments: int = None):
    """Load and prepare data from the pipeline output."""
    
    print(f"📂 Loading data from: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract segments
    segments = data.get('segments', [])
    if max_segments:
        segments = segments[:max_segments]
    
    print(f"📊 Loaded {len(segments)} segments from {data.get('source_file', 'unknown')}")
    
    # Convert to our expected format
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
    if temporal_context == "2023":
        return "2023"
    elif temporal_context == "2050":
        return "2050"
    
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

def run_scale_test(segments_count: int, pipeline_data: dict) -> TestResult:
    """Run a scale test with specified number of segments."""
    
    print(f"\n🧪 Running scale test with {segments_count} segments...")
    
    # Create tracker
    tracker = ScaleTestTracker()
    tracker.start_test()
    
    # Prepare state
    test_segments = pipeline_data["segments"][:segments_count]
    initial_state = {
        "messages": [HumanMessage(content=f"Analyze {segments_count} French sustainability transcript segments")],
        "transcript": "",
        "preprocessed_data": {
            **pipeline_data,
            "segments": test_segments
        },
        "segments": [],
        "analysis_results": [],
        "final_results": [],
        "max_segments": segments_count
    }
    
    try:
        # Run the analysis
        result = graph.invoke(
            initial_state,
            config={"callbacks": [tracker]}
        )
        
        tracker.end_test()
        summary = tracker.get_summary()
        
        # Calculate cost estimate
        input_cost = (summary['total_input_tokens'] / 1_000_000) * 0.075
        output_cost = (summary['total_output_tokens'] / 1_000_000) * 0.30
        total_cost = input_cost + output_cost
        
        return TestResult(
            segments_tested=segments_count,
            tensions_found=len(result.get('final_results', [])),
            processing_time=summary['processing_time'],
            total_tokens=summary['total_tokens'],
            input_tokens=summary['total_input_tokens'],
            output_tokens=summary['total_output_tokens'],
            api_calls=summary['total_calls'],
            errors=summary['errors'],
            success_rate=summary['success_rate'],
            cost_estimate=total_cost
        )
        
    except Exception as e:
        tracker.end_test()
        summary = tracker.get_summary()
        
        return TestResult(
            segments_tested=segments_count,
            tensions_found=0,
            processing_time=summary['processing_time'],
            total_tokens=summary['total_tokens'],
            input_tokens=summary['total_input_tokens'],
            output_tokens=summary['total_output_tokens'],
            api_calls=summary['total_calls'],
            errors=summary['errors'] + [str(e)],
            success_rate=0.0,
            cost_estimate=0.0
        )

def main():
    """Run comprehensive scale testing."""
    
    print("🎯 French Sustainability Transcript Analyzer - Scale Testing")
    print("=" * 70)
    
    # Check prerequisites
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY not found in environment variables")
        return
    
    if not Path("src/agent").exists():
        print("❌ Please run this script from the backend directory")
        return
    
    # Load pipeline data
    pipeline_file = "../Table_A_ml_ready.json"
    try:
        full_pipeline_data = load_pipeline_data(pipeline_file, max_segments=None)
        total_available = len(full_pipeline_data["segments"])
        print(f"📊 Total segments available: {total_available}")
    except FileNotFoundError:
        print(f"❌ {pipeline_file} not found")
        return
    except Exception as e:
        print(f"❌ Error loading pipeline data: {e}")
        return
    
    # Define test scales
    test_scales = [5, 10, 20, 30]  # Start conservative for free tier
    if total_available < max(test_scales):
        test_scales = [min(s, total_available) for s in test_scales if s <= total_available]
    
    print(f"🧪 Running tests with scales: {test_scales}")
    print("⏳ This will take several minutes...")
    
    # Run tests
    results = []
    for scale in test_scales:
        try:
            result = run_scale_test(scale, full_pipeline_data)
            results.append(result)
            
            print(f"✅ {scale} segments: {result.tensions_found} tensions, "
                  f"{result.processing_time:.1f}s, {result.total_tokens} tokens")
            
            # Brief pause between tests to respect rate limits
            time.sleep(2)
            
        except Exception as e:
            print(f"❌ Test with {scale} segments failed: {e}")
            traceback.print_exc()
    
    # Generate comprehensive report
    generate_scale_report(results, full_pipeline_data)

def generate_scale_report(results: List[TestResult], pipeline_data: dict):
    """Generate comprehensive scale testing report."""
    
    print("\n" + "="*70)
    print("📊 SCALE TESTING REPORT")
    print("="*70)
    
    if not results:
        print("❌ No successful tests to report")
        return
    
    # Summary table
    print("\n📋 Test Results Summary:")
    print("-" * 70)
    print(f"{'Segments':<10} {'Tensions':<10} {'Time(s)':<10} {'Tokens':<10} {'Cost($)':<10} {'Success%':<10}")
    print("-" * 70)
    
    for result in results:
        print(f"{result.segments_tested:<10} {result.tensions_found:<10} "
              f"{result.processing_time:<10.1f} {result.total_tokens:<10} "
              f"{result.cost_estimate:<10.4f} {result.success_rate*100:<10.1f}")
    
    # Performance analysis
    print(f"\n🚀 Performance Analysis:")
    if len(results) > 1:
        # Calculate scaling metrics
        largest_test = max(results, key=lambda r: r.segments_tested)
        smallest_test = min(results, key=lambda r: r.segments_tested)
        
        time_per_segment = largest_test.processing_time / largest_test.segments_tested
        tokens_per_segment = largest_test.total_tokens / largest_test.segments_tested
        cost_per_segment = largest_test.cost_estimate / largest_test.segments_tested
        
        print(f"   - Time per segment: {time_per_segment:.1f} seconds")
        print(f"   - Tokens per segment: {tokens_per_segment:.0f}")
        print(f"   - Cost per segment: ${cost_per_segment:.4f}")
        print(f"   - API calls per segment: {largest_test.api_calls / largest_test.segments_tested:.1f}")
    
    # Projection for full dataset
    total_segments = len(pipeline_data["segments"])
    if results:
        best_result = max(results, key=lambda r: r.segments_tested)
        segments_ratio = total_segments / best_result.segments_tested
        
        projected_time = best_result.processing_time * segments_ratio
        projected_tokens = best_result.total_tokens * segments_ratio
        projected_cost = best_result.cost_estimate * segments_ratio
        projected_calls = best_result.api_calls * segments_ratio
        
        print(f"\n🔮 Full Dataset Projections ({total_segments} segments):")
        print(f"   - Estimated processing time: {projected_time/3600:.1f} hours")
        print(f"   - Estimated total tokens: {projected_tokens:,.0f}")
        print(f"   - Estimated total cost: ${projected_cost:.2f}")
        print(f"   - Estimated API calls: {projected_calls:.0f}")
        
        # Free tier analysis
        daily_request_limit = 1500
        days_needed = projected_calls / daily_request_limit
        print(f"   - Days needed (free tier): {days_needed:.1f}")
    
    # Error analysis
    all_errors = []
    for result in results:
        all_errors.extend(result.errors)
    
    if all_errors:
        print(f"\n⚠️  Error Analysis:")
        print(f"   - Total errors: {len(all_errors)}")
        unique_errors = list(set(all_errors))
        for error in unique_errors[:5]:  # Show first 5 unique errors
            print(f"   - {error}")
    else:
        print(f"\n✅ No errors encountered during testing!")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if results:
        avg_success_rate = sum(r.success_rate for r in results) / len(results)
        if avg_success_rate > 0.95:
            print("   ✅ System is robust and ready for production")
        elif avg_success_rate > 0.8:
            print("   ⚠️  System mostly stable, monitor error rates")
        else:
            print("   ❌ System needs improvement before production")
        
        largest_test = max(results, key=lambda r: r.segments_tested)
        if largest_test.segments_tested >= 20:
            print("   ✅ Successfully tested at medium scale")
        else:
            print("   📈 Consider testing with larger datasets")

if __name__ == "__main__":
    main()

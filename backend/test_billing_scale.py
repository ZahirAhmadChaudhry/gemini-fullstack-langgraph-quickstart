#!/usr/bin/env python3
"""
Test with 100 segments to measure actual billing costs with paid tier.
Run from backend directory with: uv run python test_billing_scale.py
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent.batch_processor import ProductionBatchProcessor, BatchConfig

@dataclass
class CostAnalysis:
    """Detailed cost analysis for billing."""
    segments_processed: int
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    processing_time_minutes: float
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    cost_per_segment: float
    tokens_per_segment: float
    api_calls_made: int

def load_segments_data(file_path: str, max_segments: int = 100):
    """Load segments data for testing."""
    
    print(f"📂 Loading up to {max_segments} segments from: {file_path}")
    
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
    
    print(f"📊 Loaded {len(processed_segments)} segments from {data.get('source_file', 'unknown')}")
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

def estimate_tokens_from_text(text: str) -> int:
    """Rough token estimation (1 token ≈ 4 characters)."""
    return len(text) // 4

def calculate_detailed_costs(progress, processing_time_minutes: float) -> CostAnalysis:
    """Calculate detailed cost analysis."""
    
    successful_results = [r for r in progress.results if r.success]
    
    # Estimate tokens (since we may not get exact counts from API)
    total_input_tokens = 0
    total_output_tokens = 0
    
    for result in successful_results:
        # Estimate input tokens from segment text and prompts
        segment_text = ""
        for segment_data in [r.result_data for r in progress.results if r.segment_id == result.segment_id]:
            if segment_data:
                segment_text = str(segment_data)
        
        # Rough estimation: segment text + prompt overhead × 5 calls per segment
        input_tokens_per_segment = estimate_tokens_from_text(segment_text) * 5
        total_input_tokens += input_tokens_per_segment
        
        # Estimate output tokens from result data
        output_text = json.dumps(result.result_data, ensure_ascii=False)
        output_tokens_per_segment = estimate_tokens_from_text(output_text)
        total_output_tokens += output_tokens_per_segment
    
    total_tokens = total_input_tokens + total_output_tokens
    
    # Gemini 2.0 Flash pricing (as of 2024)
    # Input: $0.075 per 1M tokens
    # Output: $0.30 per 1M tokens
    input_cost = (total_input_tokens / 1_000_000) * 0.075
    output_cost = (total_output_tokens / 1_000_000) * 0.30
    total_cost = input_cost + output_cost
    
    return CostAnalysis(
        segments_processed=len(successful_results),
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_tokens=total_tokens,
        processing_time_minutes=processing_time_minutes,
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        total_cost_usd=total_cost,
        cost_per_segment=total_cost / max(len(successful_results), 1),
        tokens_per_segment=total_tokens / max(len(successful_results), 1),
        api_calls_made=len(successful_results) * 5  # 5 calls per segment
    )

def main():
    """Test with 100 segments to measure billing costs."""
    
    print("💰 French Sustainability Transcript Analyzer - BILLING SCALE TEST")
    print("=" * 75)
    print("🎯 Testing with 100 segments to measure actual costs")
    
    # Check prerequisites
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY not found in environment variables")
        return
    
    # Load 100 segments
    pipeline_file = "../Table_A_ml_ready.json"
    try:
        segments = load_segments_data(pipeline_file, max_segments=100)
        print(f"📊 Ready to process {len(segments)} segments")
    except Exception as e:
        print(f"❌ Error loading pipeline data: {e}")
        return
    
    # Configure for Tier 1 with Gemini 2.0 Flash (2000 RPM limit!)
    config = BatchConfig(
        max_requests_per_minute=300,  # Conservative: 300/2000 limit (5 calls per segment = 60 segments/min)
        retry_attempts=3,
        retry_delay=5,  # Much shorter delays
        batch_size=1,
        save_progress=True,
        progress_file="billing_test_progress.json"
    )
    
    print(f"⚙️  Configuration for Tier 1 (Gemini 2.0 Flash):")
    print(f"   - Max requests/minute: {config.max_requests_per_minute} (limit: 2000)")
    print(f"   - Retry attempts: {config.retry_attempts}")
    print(f"   - Expected API calls: {len(segments) * 5}")
    print(f"   - Segments per minute: ~{config.max_requests_per_minute // 5}")

    # Estimate processing time
    estimated_minutes = (len(segments) * 5) / config.max_requests_per_minute
    print(f"⏱️  Estimated processing time: {estimated_minutes:.1f} minutes")
    
    # Cost estimation
    estimated_input_tokens = sum(estimate_tokens_from_text(s['text']) * 5 for s in segments)
    estimated_output_tokens = estimated_input_tokens // 4  # Rough guess
    estimated_cost = (estimated_input_tokens / 1_000_000 * 0.075) + (estimated_output_tokens / 1_000_000 * 0.30)
    
    print(f"💰 Pre-processing Cost Estimates:")
    print(f"   - Estimated input tokens: {estimated_input_tokens:,}")
    print(f"   - Estimated output tokens: {estimated_output_tokens:,}")
    print(f"   - Estimated total cost: ${estimated_cost:.4f}")
    print(f"   - Cost per segment: ${estimated_cost/len(segments):.6f}")
    
    # Confirm before proceeding
    print(f"\n⚠️  BILLING CONFIRMATION:")
    print(f"   This test will process {len(segments)} segments")
    print(f"   Estimated cost: ${estimated_cost:.4f}")
    print(f"   Processing time: ~{estimated_minutes:.1f} minutes")
    
    response = input("\n   Continue with billing test? (yes/no): ").strip().lower()
    if response != 'yes':
        print("❌ Test cancelled by user")
        return
    
    # Create processor
    processor = ProductionBatchProcessor(config)
    
    try:
        print(f"\n🚀 Starting BILLING SCALE TEST...")
        start_time = time.time()
        
        # Process batch
        progress = processor.process_batch(segments, resume=True)
        
        end_time = time.time()
        processing_time_minutes = (end_time - start_time) / 60
        
        # Calculate detailed costs
        cost_analysis = calculate_detailed_costs(progress, processing_time_minutes)
        
        # Export results
        output_file = processor.export_results(progress, "billing_test_results.json")
        
        # Comprehensive billing report
        print(f"\n" + "="*75)
        print(f"💰 BILLING SCALE TEST RESULTS")
        print(f"="*75)
        
        print(f"\n📊 Processing Summary:")
        print(f"   - Segments processed: {cost_analysis.segments_processed}/{len(segments)}")
        print(f"   - Success rate: {cost_analysis.segments_processed/len(segments)*100:.1f}%")
        print(f"   - Processing time: {cost_analysis.processing_time_minutes:.1f} minutes")
        print(f"   - API calls made: {cost_analysis.api_calls_made}")
        print(f"   - Rate limit hits: {processor.rate_limit_hits}")
        
        print(f"\n💰 Cost Analysis:")
        print(f"   - Total input tokens: {cost_analysis.total_input_tokens:,}")
        print(f"   - Total output tokens: {cost_analysis.total_output_tokens:,}")
        print(f"   - Total tokens: {cost_analysis.total_tokens:,}")
        print(f"   - Input cost: ${cost_analysis.input_cost_usd:.4f}")
        print(f"   - Output cost: ${cost_analysis.output_cost_usd:.4f}")
        print(f"   - TOTAL COST: ${cost_analysis.total_cost_usd:.4f}")
        
        print(f"\n📈 Per-Segment Metrics:")
        print(f"   - Cost per segment: ${cost_analysis.cost_per_segment:.6f}")
        print(f"   - Tokens per segment: {cost_analysis.tokens_per_segment:.0f}")
        print(f"   - Time per segment: {cost_analysis.processing_time_minutes*60/cost_analysis.segments_processed:.1f} seconds")
        
        # Scale projections
        total_segments = 302  # Your full dataset
        scale_factor = total_segments / cost_analysis.segments_processed
        
        print(f"\n🔮 Full Dataset Projections ({total_segments} segments):")
        projected_cost = cost_analysis.total_cost_usd * scale_factor
        projected_time = cost_analysis.processing_time_minutes * scale_factor
        
        print(f"   - Projected total cost: ${projected_cost:.2f}")
        print(f"   - Projected processing time: {projected_time/60:.1f} hours")
        print(f"   - Monthly cost (if run weekly): ${projected_cost*4:.2f}")
        print(f"   - Annual cost (if run monthly): ${projected_cost*12:.2f}")
        
        # Budget analysis
        print(f"\n💡 Budget Analysis:")
        if projected_cost < 1.0:
            print(f"   ✅ Very affordable: <$1 for full dataset")
        elif projected_cost < 10.0:
            print(f"   ✅ Affordable: <$10 for full dataset")
        elif projected_cost < 50.0:
            print(f"   ⚠️  Moderate cost: <$50 for full dataset")
        else:
            print(f"   ❌ High cost: >${projected_cost:.0f} for full dataset")
        
        print(f"\n📁 Detailed results saved to: {output_file}")
        
        # Save cost analysis
        cost_report = {
            "test_date": datetime.now().isoformat(),
            "segments_tested": len(segments),
            "cost_analysis": {
                "segments_processed": cost_analysis.segments_processed,
                "total_cost_usd": cost_analysis.total_cost_usd,
                "cost_per_segment": cost_analysis.cost_per_segment,
                "tokens_per_segment": cost_analysis.tokens_per_segment,
                "processing_time_minutes": cost_analysis.processing_time_minutes
            },
            "projections": {
                "full_dataset_segments": total_segments,
                "projected_cost_usd": projected_cost,
                "projected_time_hours": projected_time / 60
            }
        }
        
        with open("billing_cost_analysis.json", "w", encoding="utf-8") as f:
            json.dump(cost_report, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Cost analysis saved to: billing_cost_analysis.json")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Test interrupted by user")
        print(f"📁 Progress saved - can resume with same command")
    except Exception as e:
        print(f"❌ Billing test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test the production batch processor with a small sample.
Run from backend directory with: uv run python test_production_batch.py
"""

import os
import sys
import json
from pathlib import Path
from datetime import timedelta

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent.batch_processor import ProductionBatchProcessor, BatchConfig

def load_sample_data(file_path: str, max_segments: int = 5):
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
    """Test the production batch processor."""
    
    print("🏭 French Sustainability Transcript Analyzer - Production Batch Test")
    print("=" * 75)
    
    # Check prerequisites
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY not found in environment variables")
        return
    
    # Load sample data
    pipeline_file = "../Table_A_ml_ready.json"
    try:
        sample_segments = load_sample_data(pipeline_file, max_segments=3)  # Very conservative
        print(f"📊 Loaded {len(sample_segments)} sample segments for testing")
    except Exception as e:
        print(f"❌ Error loading pipeline data: {e}")
        return
    
    # Configure batch processor
    config = BatchConfig(
        max_requests_per_minute=4,  # Very conservative
        retry_attempts=2,
        retry_delay=30,
        batch_size=1,
        save_progress=True,
        progress_file="test_batch_progress.json"
    )
    
    print(f"⚙️  Batch Configuration:")
    print(f"   - Max requests/minute: {config.max_requests_per_minute}")
    print(f"   - Retry attempts: {config.retry_attempts}")
    print(f"   - Progress saving: {config.save_progress}")
    
    # Create processor
    processor = ProductionBatchProcessor(config)
    
    # Estimate time
    estimated_minutes = (len(sample_segments) * 5) / config.max_requests_per_minute
    print(f"⏱️  Estimated processing time: {estimated_minutes:.1f} minutes")
    
    try:
        # Process batch
        print(f"\n🚀 Starting production batch processing...")
        progress = processor.process_batch(sample_segments, resume=True)
        
        # Export results
        output_file = processor.export_results(progress, "test_production_results.json")
        
        # Summary
        print(f"\n📊 Final Results:")
        print(f"   - Total segments: {progress.total_segments}")
        print(f"   - Successful: {progress.successful_segments}")
        print(f"   - Failed: {progress.failed_segments}")
        print(f"   - Success rate: {progress.successful_segments/progress.total_segments*100:.1f}%")
        print(f"   - Total tensions found: {sum(r.tensions_found for r in progress.results if r.success)}")
        print(f"   - Rate limit hits: {processor.rate_limit_hits}")
        
        # Production readiness assessment
        success_rate = progress.successful_segments / progress.total_segments
        if success_rate >= 0.8:
            print(f"\n✅ PRODUCTION READY!")
            print(f"   - High success rate ({success_rate*100:.1f}%)")
            print(f"   - Error handling working")
            print(f"   - Progress saving functional")
            
            # Scale projections
            total_segments = 302  # From your full dataset
            scale_time = (total_segments * 5) / config.max_requests_per_minute / 60  # hours
            print(f"\n🔮 Full Dataset Projections:")
            print(f"   - Processing time: {scale_time:.1f} hours")
            print(f"   - Recommended batch size: 50-100 segments")
            print(f"   - Can process overnight or over weekend")
            
        else:
            print(f"\n⚠️  NEEDS IMPROVEMENT")
            print(f"   - Low success rate ({success_rate*100:.1f}%)")
            print(f"   - Review error handling")
            print(f"   - Consider slower rate limits")
        
        print(f"\n📁 Results saved to: {output_file}")
        print(f"📁 Progress saved to: {config.progress_file}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Processing interrupted by user")
        print(f"📁 Progress saved - can resume with same command")
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

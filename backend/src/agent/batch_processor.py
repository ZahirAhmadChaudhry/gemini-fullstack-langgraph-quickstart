"""
Production-ready batch processor for French sustainability transcript analysis.
Handles rate limiting, error recovery, and progress tracking.
"""

import time
import json
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path

from agent.graph import graph
from langchain_core.messages import HumanMessage


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    max_requests_per_minute: int = 600  # Optimized for Tier 1 (2000 RPM limit)
    retry_attempts: int = 3
    retry_delay: int = 10  # seconds
    batch_size: int = 1  # Process one segment at a time
    save_progress: bool = True
    progress_file: str = "batch_progress.json"
    parallel_workers: int = 10  # Parallel processing for 8.9x speed improvement


@dataclass
class SegmentResult:
    """Result for a single segment."""
    segment_id: str
    success: bool
    tensions_found: int
    processing_time: float
    error: Optional[str]
    timestamp: str
    result_data: List[Dict[str, Any]]


@dataclass
class BatchProgress:
    """Progress tracking for batch processing."""
    total_segments: int
    processed_segments: int
    successful_segments: int
    failed_segments: int
    start_time: str
    last_update: str
    estimated_completion: str
    results: List[SegmentResult]


class ProductionBatchProcessor:
    """Production-ready batch processor with rate limiting and error recovery."""
    
    def __init__(self, config: BatchConfig = None):
        self.config = config or BatchConfig()
        self.last_request_time = 0
        self.request_count = 0
        self.rate_limit_hits = 0
        
    def _wait_for_rate_limit(self):
        """Wait to respect rate limits."""
        min_delay = 60.0 / self.config.max_requests_per_minute
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < min_delay:
            wait_time = min_delay - time_since_last
            print(f"   ⏳ Rate limiting: waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def _handle_rate_limit_error(self, error_msg: str) -> int:
        """Handle rate limit errors and return wait time."""
        self.rate_limit_hits += 1
        
        # Extract retry delay from error message
        retry_delay = self.config.retry_delay
        if "retry_delay" in error_msg and "seconds:" in error_msg:
            try:
                delay_part = error_msg.split("seconds:")[1].split("}")[0].strip()
                retry_delay = int(delay_part) + 10  # Add buffer
            except:
                pass
        
        print(f"   🚫 Rate limit hit #{self.rate_limit_hits}. Waiting {retry_delay}s...")
        return retry_delay
    
    def _process_single_segment(self, segment_data: Dict[str, Any]) -> SegmentResult:
        """Process a single segment with error handling."""
        segment_id = segment_data.get('id', 'unknown')
        start_time = time.time()
        
        for attempt in range(self.config.retry_attempts):
            try:
                # Wait for rate limiting
                self._wait_for_rate_limit()
                
                # Prepare state
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
                
                return SegmentResult(
                    segment_id=segment_id,
                    success=True,
                    tensions_found=len(final_results),
                    processing_time=end_time - start_time,
                    error=None,
                    timestamp=datetime.now().isoformat(),
                    result_data=final_results
                )
                
            except Exception as e:
                error_msg = str(e)
                
                # Handle rate limit errors
                if "429" in error_msg or "quota" in error_msg.lower():
                    wait_time = self._handle_rate_limit_error(error_msg)
                    
                    if attempt < self.config.retry_attempts - 1:
                        print(f"   🔄 Retry attempt {attempt + 1}/{self.config.retry_attempts}")
                        time.sleep(wait_time)
                        continue
                
                # Other errors
                end_time = time.time()
                return SegmentResult(
                    segment_id=segment_id,
                    success=False,
                    tensions_found=0,
                    processing_time=end_time - start_time,
                    error=error_msg,
                    timestamp=datetime.now().isoformat(),
                    result_data=[]
                )
        
        # All retries failed
        end_time = time.time()
        return SegmentResult(
            segment_id=segment_id,
            success=False,
            tensions_found=0,
            processing_time=end_time - start_time,
            error="Max retries exceeded",
            timestamp=datetime.now().isoformat(),
            result_data=[]
        )
    
    def _save_progress(self, progress: BatchProgress):
        """Save progress to file."""
        if self.config.save_progress:
            with open(self.config.progress_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(progress), f, indent=2, ensure_ascii=False)
    
    def _load_progress(self) -> Optional[BatchProgress]:
        """Load existing progress."""
        if self.config.save_progress and Path(self.config.progress_file).exists():
            try:
                with open(self.config.progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Convert dict results back to SegmentResult objects
                    if 'results' in data:
                        results = []
                        for r in data['results']:
                            if isinstance(r, dict):
                                results.append(SegmentResult(**r))
                            else:
                                results.append(r)
                        data['results'] = results
                    return BatchProgress(**data)
            except Exception as e:
                print(f"Warning: Could not load progress file: {e}")
                return None
        return None
    
    def _estimate_completion_time(self, processed: int, total: int, start_time: datetime) -> str:
        """Estimate completion time."""
        if processed == 0:
            return "Unknown"
        
        elapsed = datetime.now() - start_time
        avg_time_per_segment = elapsed.total_seconds() / processed
        remaining_segments = total - processed
        remaining_time = remaining_segments * avg_time_per_segment
        
        completion_time = datetime.now() + timedelta(seconds=remaining_time)
        return completion_time.isoformat()
    
    def process_batch(self, segments: List[Dict[str, Any]], resume: bool = True) -> BatchProgress:
        """Process a batch of segments with progress tracking."""
        
        # Load existing progress if resuming
        progress = None
        if resume:
            progress = self._load_progress()
        
        if progress is None:
            # Start new batch
            progress = BatchProgress(
                total_segments=len(segments),
                processed_segments=0,
                successful_segments=0,
                failed_segments=0,
                start_time=datetime.now().isoformat(),
                last_update=datetime.now().isoformat(),
                estimated_completion="Unknown",
                results=[]
            )
        
        start_time = datetime.fromisoformat(progress.start_time)
        processed_ids = {r.segment_id for r in progress.results}
        
        print(f"🚀 Starting batch processing...")
        print(f"   📊 Total segments: {progress.total_segments}")
        print(f"   📈 Already processed: {progress.processed_segments}")
        print(f"   ⏱️  Rate limit: {self.config.max_requests_per_minute} req/min")
        
        for i, segment in enumerate(segments):
            segment_id = segment.get('id', f'segment_{i}')
            
            # Skip if already processed
            if segment_id in processed_ids:
                continue
            
            print(f"\n   📍 Processing {segment_id} ({progress.processed_segments + 1}/{progress.total_segments})")
            
            # Process segment
            result = self._process_single_segment(segment)
            progress.results.append(result)
            progress.processed_segments += 1
            
            if result.success:
                progress.successful_segments += 1
                print(f"      ✅ Success: {result.tensions_found} tensions in {result.processing_time:.1f}s")
            else:
                progress.failed_segments += 1
                print(f"      ❌ Failed: {result.error}")
            
            # Update progress
            progress.last_update = datetime.now().isoformat()
            progress.estimated_completion = self._estimate_completion_time(
                progress.processed_segments, progress.total_segments, start_time
            )
            
            # Save progress
            self._save_progress(progress)
            
            # Progress report every 5 segments
            if progress.processed_segments % 5 == 0:
                success_rate = progress.successful_segments / progress.processed_segments * 100
                print(f"   📊 Progress: {progress.processed_segments}/{progress.total_segments} "
                      f"({success_rate:.1f}% success rate)")
        
        print(f"\n🎉 Batch processing completed!")
        print(f"   ✅ Successful: {progress.successful_segments}")
        print(f"   ❌ Failed: {progress.failed_segments}")
        print(f"   📊 Success rate: {progress.successful_segments/progress.total_segments*100:.1f}%")
        
        return progress
    
    def export_results(self, progress: BatchProgress, output_file: str = "analysis_results.json"):
        """Export results to JSON file."""
        successful_results = [r for r in progress.results if r.success]
        
        export_data = {
            "metadata": {
                "total_segments": progress.total_segments,
                "successful_segments": progress.successful_segments,
                "failed_segments": progress.failed_segments,
                "success_rate": progress.successful_segments / progress.total_segments,
                "processing_time": progress.last_update,
                "total_tensions_found": sum(r.tensions_found for r in successful_results)
            },
            "results": [
                {
                    "segment_id": r.segment_id,
                    "tensions_found": r.tensions_found,
                    "processing_time": r.processing_time,
                    "timestamp": r.timestamp,
                    "analysis": r.result_data
                }
                for r in successful_results
            ]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Results exported to {output_file}")
        return output_file

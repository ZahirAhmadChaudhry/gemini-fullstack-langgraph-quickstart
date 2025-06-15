#!/usr/bin/env python
"""
Performance Benchmarking Script for French Transcript Preprocessing

This script conducts performance benchmarking on the preprocessing pipeline
with a focus on large document sets. It measures:
- Processing time
- Memory usage
- CPU utilization
- Scaling characteristics with document size

The benchmarks are run with various configurations to identify optimal settings.

Usage:
    python run_benchmark.py [--large] [--heavy-load]

Options:
    --large       Run benchmarks on large documents (>5MB)
    --heavy-load  Simulate concurrent processing load
"""

import os
import sys
import json
import time
import logging
import psutil
import argparse
import multiprocessing
from pathlib import Path
from datetime import datetime
import concurrent.futures
import matplotlib.pyplot as plt
import numpy as np

# Adjust Python path to allow importing from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import required modules
import preprocess_transcripts as pipeline

# Configure logging
logging.basicConfig(
    filename=Path(__file__).parent / 'logs' / 'benchmark.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('benchmark')

class PerformanceBenchmark:
    """Benchmark runner for the preprocessing pipeline performance."""
    
    def __init__(self, large_docs=False, heavy_load=False):
        """Initialize benchmark environment.
        
        Args:
            large_docs (bool): Whether to include large documents in benchmark
            heavy_load (bool): Whether to simulate heavy system load during benchmark
        """
        self.test_dir = Path(__file__).parent
        self.benchmark_dir = self.test_dir / 'benchmark'
        self.results_dir = self.benchmark_dir / 'results'
        self.charts_dir = self.benchmark_dir / 'charts'
        
        # Data directories for benchmarking
        self.main_data_dir = Path(__file__).parent.parent / 'data'
        self.files_dir = Path(__file__).parent.parent / 'files'
        self.data_renamed_dir = Path(__file__).parent.parent / 'data_renamed'
        
        # Benchmark parameters
        self.large_docs = large_docs
        self.heavy_load = heavy_load
        self.cpu_count = multiprocessing.cpu_count()
        
        # Ensure directories exist
        self.ensure_directories_exist()
        
        # Initialize results container
        self.benchmark_results = {
            'environment': {
                'cpu_count': self.cpu_count,
                'os': sys.platform,
                'python_version': sys.version,
                'timestamp': datetime.now().isoformat(),
                'large_docs': large_docs,
                'heavy_load': heavy_load
            },
            'results': {
                'individual_files': {},
                'batch_processing': [],
                'scaling_tests': []
            }
        }
    
    def ensure_directories_exist(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.benchmark_dir, self.results_dir, self.charts_dir]:
            directory.mkdir(exist_ok=True, parents=True)
    
    def collect_files_by_size(self):
        """Collect files for benchmarking, categorized by size.
        
        Returns:
            dict: Categorized files by size
        """
        all_files = []
        
        # Collect files from all relevant directories
        for directory in [self.main_data_dir, self.files_dir, self.data_renamed_dir]:
            if directory.exists():
                all_files.extend(list(directory.glob('*.*')))
        
        # Filter for common document types
        supported_extensions = ['.txt', '.docx', '.pdf', '.md']
        all_files = [f for f in all_files if f.suffix.lower() in supported_extensions]
        
        # Categorize by size
        files_by_size = {
            'small': [],    # <100KB
            'medium': [],   # 100KB-1MB
            'large': []     # >1MB
        }
        
        for file_path in all_files:
            size_kb = file_path.stat().st_size / 1024
            
            if size_kb < 100:
                files_by_size['small'].append(file_path)
            elif size_kb < 1024:
                files_by_size['medium'].append(file_path)
            else:
                files_by_size['large'].append(file_path)
        
        # If not using large docs, exclude them
        if not self.large_docs:
            files_by_size['large'] = []
        
        # Log the collection results
        logger.info(f"Collected {len(all_files)} files for benchmarking:")
        logger.info(f"  Small (<100KB): {len(files_by_size['small'])} files")
        logger.info(f"  Medium (100KB-1MB): {len(files_by_size['medium'])} files")
        logger.info(f"  Large (>1MB): {len(files_by_size['large'])} files")
        
        return files_by_size
    
    def track_system_resources(self, duration=1.0, interval=0.1):
        """Track system resource usage.
        
        Args:
            duration (float): Duration to track in seconds
            interval (float): Measurement interval in seconds
            
        Returns:
            dict: Resource usage statistics
        """
        measurements = []
        end_time = time.time() + duration
        
        while time.time() < end_time:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory_info = psutil.Process(os.getpid()).memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            measurements.append({
                'timestamp': time.time(),
                'cpu_percent': cpu_percent,
                'memory_mb': memory_mb
            })
            
            time.sleep(interval)
        
        # Calculate statistics
        if not measurements:
            return {
                'cpu_avg': 0,
                'cpu_max': 0,
                'memory_avg_mb': 0,
                'memory_max_mb': 0
            }
        
        cpu_values = [m['cpu_percent'] for m in measurements]
        memory_values = [m['memory_mb'] for m in measurements]
        
        return {
            'cpu_avg': sum(cpu_values) / len(cpu_values),
            'cpu_max': max(cpu_values),
            'memory_avg_mb': sum(memory_values) / len(memory_values),
            'memory_max_mb': max(memory_values)
        }
    
    def simulate_load(self, cpu_target=80, duration=None):
        """Simulate CPU load for benchmarking.
        
        Args:
            cpu_target (int): Target CPU percentage
            duration (float): Duration to run in seconds, or None for continuous
            
        Returns:
            multiprocessing.Process: The load process
        """
        def cpu_load(stop_event, target_pct):
            """Function to generate CPU load."""
            while not stop_event.is_set():
                start_time = time.time()
                # Burn CPU cycles
                while time.time() - start_time < 0.1:
                    _ = [i**2 for i in range(10000)]
                    
                # Sleep to achieve target percentage
                time.sleep((0.1 * (100 - target_pct)) / target_pct)
        
        stop_event = multiprocessing.Event()
        process = multiprocessing.Process(
            target=cpu_load, 
            args=(stop_event, cpu_target)
        )
        
        process.start()
        logger.info(f"Started load simulation at {cpu_target}% CPU")
        
        # If duration specified, schedule stop
        if duration:
            def stop_after_duration():
                time.sleep(duration)
                stop_event.set()
                logger.info("Stopped load simulation")
            
            stop_thread = threading.Thread(target=stop_after_duration)
            stop_thread.daemon = True
            stop_thread.start()
        
        return process, stop_event
    
    def benchmark_single_file(self, file_path, category):
        """Benchmark processing of a single file.
        
        Args:
            file_path (Path): Path to the file to benchmark
            category (str): Size category of the file
            
        Returns:
            dict: Benchmark results
        """
        file_name = file_path.name
        logger.info(f"Benchmarking {category} file: {file_name}")
        
        output_file = self.benchmark_dir / f"{file_path.stem}_benchmark.json"
        
        # Prepare benchmark data
        benchmark_data = {
            'file_name': file_name,
            'file_path': str(file_path),
            'file_size_kb': file_path.stat().st_size / 1024,
            'file_type': file_path.suffix,
            'category': category,
            'success': False,
            'error': None
        }
        
        try:
            # Track initial state
            initial_resources = self.track_system_resources(duration=0.5)
            
            # Process the file and time it
            start_time = time.time()
            
            # Call the preprocessing pipeline
            pipeline_args = {
                'input_file': str(file_path),
                'output_file': str(output_file),
                'ml_ready_format': True,
                'verbose': False
            }
              # Create preprocessor instance and process file directly
            preprocessor = pipeline.TranscriptPreprocessor()
            
            # Process the individual file
            result = preprocessor.preprocess_transcript(file_path, file_path.name)
                
            # Save results manually to the specified location
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
              # Create preprocessor instance
            preprocessor = pipeline.TranscriptPreprocessor()
            
            # Create output directory if it doesn't exist
            output_dir = output_file.parent
            output_dir.mkdir(exist_ok=True, parents=True)
            
            # Process the individual file
            result = preprocessor.preprocess_transcript(file_path, file_path.name)
                
            # Save results manually to the specified location
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            benchmark_data['processing_time_sec'] = processing_time
            
            # Track resource usage after processing
            final_resources = self.track_system_resources(duration=0.5)
            
            # Add resource metrics
            benchmark_data.update({
                'initial_cpu_percent': initial_resources['cpu_avg'],
                'final_cpu_percent': final_resources['cpu_avg'],
                'initial_memory_mb': initial_resources['memory_avg_mb'],
                'final_memory_mb': final_resources['memory_avg_mb'],
                'memory_diff_mb': final_resources['memory_avg_mb'] - initial_resources['memory_avg_mb'],
                'success': output_file.exists()
            })
            
            # Calculate processing speed
            if benchmark_data['file_size_kb'] > 0:
                benchmark_data['processing_speed_kb_per_sec'] = benchmark_data['file_size_kb'] / processing_time
            
            logger.info(f"Completed benchmark for {file_name}: {processing_time:.2f} seconds")
            
            # Clean up output file
            if output_file.exists():
                output_file.unlink()
        
        except Exception as e:
            benchmark_data['error'] = str(e)
            logger.error(f"Benchmark error for {file_name}: {e}", exc_info=True)
        
        return benchmark_data
    
    def benchmark_batch_processing(self, files_by_size, batch_sizes=[1, 2, 4]):
        """Benchmark batch processing with different sizes.
        
        Args:
            files_by_size (dict): Files categorized by size
            batch_sizes (list): Batch sizes to test
            
        Returns:
            list: Batch benchmark results
        """
        results = []
        
        # Prepare a mixed batch of files for testing
        test_batch = []
        for category, files in files_by_size.items():
            if files:  # Add up to 3 files from each category
                test_batch.extend(files[:min(3, len(files))])
        
        if not test_batch:
            logger.warning("No files available for batch processing benchmark")
            return results
        
        # Get file information for the selected batch
        batch_info = []
        for file_path in test_batch:
            batch_info.append({
                'file_name': file_path.name,
                'file_size_kb': file_path.stat().st_size / 1024
            })
        
        # Run benchmarks with different batch sizes
        for batch_size in batch_sizes:
            if batch_size > len(test_batch):
                continue
                
            test_files = test_batch[:batch_size]
            
            logger.info(f"Benchmarking batch processing with batch size: {batch_size}")
            
            # Prepare benchmark data
            benchmark_data = {
                'batch_size': batch_size,
                'total_files': len(test_files),
                'batch_info': batch_info[:batch_size],
                'success': False,
                'error': None
            }
            
            try:
                # Track initial state
                initial_resources = self.track_system_resources(duration=0.5)
                
                # Process the batch and time it
                start_time = time.time()
                
                # Process files in parallel
                with concurrent.futures.ProcessPoolExecutor(max_workers=batch_size) as executor:
                    future_to_file = {
                        executor.submit(self._process_file_for_batch, file_path): file_path
                        for file_path in test_files
                    }
                    
                    # Wait for all futures to complete
                    for future in concurrent.futures.as_completed(future_to_file):
                        file_path = future_to_file[future]
                        try:
                            _ = future.result()
                        except Exception as e:
                            logger.error(f"Error processing {file_path.name}: {e}")
                
                # Calculate processing time
                processing_time = time.time() - start_time
                benchmark_data['total_processing_time_sec'] = processing_time
                
                # Track resource usage after processing
                final_resources = self.track_system_resources(duration=0.5)
                
                # Add resource metrics
                benchmark_data.update({
                    'initial_cpu_percent': initial_resources['cpu_avg'],
                    'peak_cpu_percent': final_resources['cpu_max'],
                    'initial_memory_mb': initial_resources['memory_avg_mb'],
                    'peak_memory_mb': final_resources['memory_max_mb'],
                    'success': True
                })
                
                # Calculate average time per file
                benchmark_data['avg_time_per_file_sec'] = processing_time / len(test_files)
                
                logger.info(f"Completed batch benchmark with batch size {batch_size}: "
                           f"{processing_time:.2f} seconds total, "
                           f"{benchmark_data['avg_time_per_file_sec']:.2f} seconds per file")
                
            except Exception as e:
                benchmark_data['error'] = str(e)
                logger.error(f"Batch benchmark error: {e}", exc_info=True)
            
            results.append(benchmark_data)
        
        return results
      def _process_file_for_batch(self, file_path):
        """Helper function to process a file for batch benchmarking."""
        output_file = self.benchmark_dir / f"{file_path.stem}_batch_benchmark.json"
        
        # Create preprocessor instance
        preprocessor = pipeline.TranscriptPreprocessor()
        
        # Create output directory if it doesn't exist
        output_dir = output_file.parent
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Process the individual file
        result = preprocessor.preprocess_transcript(file_path, file_path.name)
        
        # Write the result to the output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        # Save results manually to the specified location
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # Clean up output file
        if output_file.exists():
            output_file.unlink()
        
        return True
    
    def benchmark_scaling(self, files_by_size):
        """Benchmark how performance scales with file size.
        
        Args:
            files_by_size (dict): Files categorized by size
            
        Returns:
            list: Scaling benchmark results
        """
        results = []
        
        # Select files for the scaling test
        # Try to get files with increasing sizes
        test_files = []
        
        # Get some small files
        if files_by_size['small']:
            test_files.extend(sorted(files_by_size['small'], 
                                    key=lambda f: f.stat().st_size)[:2])
        
        # Get some medium files
        if files_by_size['medium']:
            # Get smallest, middle, and largest medium file
            sorted_medium = sorted(files_by_size['medium'], key=lambda f: f.stat().st_size)
            if len(sorted_medium) >= 3:
                test_files.append(sorted_medium[0])  # Smallest
                test_files.append(sorted_medium[len(sorted_medium) // 2])  # Middle
                test_files.append(sorted_medium[-1])  # Largest
            else:
                test_files.extend(sorted_medium)
        
        # Get large files if enabled
        if self.large_docs and files_by_size['large']:
            test_files.extend(sorted(files_by_size['large'], 
                                    key=lambda f: f.stat().st_size)[:2])
        
        # Run benchmark for each selected file
        for file_path in test_files:
            size_kb = file_path.stat().st_size / 1024
            
            # Determine category
            if size_kb < 100:
                category = 'small'
            elif size_kb < 1024:
                category = 'medium'
            else:
                category = 'large'
            
            # Benchmark this file
            result = self.benchmark_single_file(file_path, category)
            
            # Add to scaling results
            results.append(result)
        
        return results
    
    def run_benchmarks(self):
        """Run all benchmarks."""
        # Collect files
        files_by_size = self.collect_files_by_size()
        
        # Start load simulation if requested
        load_process = None
        stop_event = None
        if self.heavy_load:
            import threading
            load_process, stop_event = self.simulate_load(cpu_target=70)
        
        try:
            # Benchmark individual files (sample from each category)
            logger.info("Starting individual file benchmarks")
            for category, files in files_by_size.items():
                # Take up to 3 files from each category
                for file_path in files[:min(3, len(files))]:
                    result = self.benchmark_single_file(file_path, category)
                    self.benchmark_results['results']['individual_files'][str(file_path)] = result
            
            # Benchmark batch processing
            logger.info("Starting batch processing benchmarks")
            batch_results = self.benchmark_batch_processing(files_by_size)
            self.benchmark_results['results']['batch_processing'] = batch_results
            
            # Benchmark scaling with file size
            logger.info("Starting scaling benchmarks")
            scaling_results = self.benchmark_scaling(files_by_size)
            self.benchmark_results['results']['scaling_tests'] = scaling_results
            
            # Save results
            self.save_results()
            
            # Generate charts
            self.generate_charts()
            
            logger.info("All benchmarks completed successfully")
            
        finally:
            # Stop load simulation if running
            if self.heavy_load and load_process and stop_event:
                stop_event.set()
                load_process.join(timeout=2)
                if load_process.is_alive():
                    load_process.terminate()
    
    def save_results(self):
        """Save the benchmark results to a file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.results_dir / f'benchmark_results_{timestamp}.json'
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.benchmark_results, f, indent=2)
        
        logger.info(f"Saved benchmark results to {results_path}")
    
    def generate_charts(self):
        """Generate charts from benchmark data."""
        try:
            # Import matplotlib here to avoid issues if it's not installed
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            # File size vs. processing time chart
            plt.figure(figsize=(10, 6))
            
            # Collect data points from scaling tests
            sizes = []
            times = []
            categories = []
            
            for result in self.benchmark_results['results']['scaling_tests']:
                if result['success'] and 'file_size_kb' in result and 'processing_time_sec' in result:
                    sizes.append(result['file_size_kb'])
                    times.append(result['processing_time_sec'])
                    categories.append(result['category'])
            
            # If we have data, plot it
            if sizes:
                # Create scatter plot with different colors by category
                for category in ['small', 'medium', 'large']:
                    cat_sizes = [sizes[i] for i in range(len(sizes)) if categories[i] == category]
                    cat_times = [times[i] for i in range(len(times)) if categories[i] == category]
                    if cat_sizes:
                        plt.scatter(cat_sizes, cat_times, label=category)
                
                # Add best fit line
                if len(sizes) > 1:
                    z = np.polyfit(sizes, times, 1)
                    p = np.poly1d(z)
                    plt.plot(sizes, p(sizes), "r--", alpha=0.8)
                
                plt.xlabel('File Size (KB)')
                plt.ylabel('Processing Time (seconds)')
                plt.title('File Size vs. Processing Time')
                plt.legend()
                plt.grid(True)
                
                # Save the figure
                chart_path = self.charts_dir / 'size_vs_time.png'
                plt.savefig(chart_path)
                plt.close()
                logger.info(f"Generated chart: {chart_path}")
            
            # Batch size vs. processing efficiency chart
            plt.figure(figsize=(10, 6))
            
            # Collect data from batch processing results
            batch_sizes = []
            avg_times = []
            
            for result in self.benchmark_results['results']['batch_processing']:
                if result['success'] and 'batch_size' in result and 'avg_time_per_file_sec' in result:
                    batch_sizes.append(result['batch_size'])
                    avg_times.append(result['avg_time_per_file_sec'])
            
            # If we have data, plot it
            if batch_sizes:
                plt.plot(batch_sizes, avg_times, 'bo-')
                plt.xlabel('Batch Size')
                plt.ylabel('Average Processing Time per File (seconds)')
                plt.title('Batch Size vs. Processing Efficiency')
                plt.grid(True)
                
                # Save the figure
                chart_path = self.charts_dir / 'batch_efficiency.png'
                plt.savefig(chart_path)
                plt.close()
                logger.info(f"Generated chart: {chart_path}")
        
        except ImportError:
            logger.warning("Matplotlib not installed. Skipping chart generation.")
        except Exception as e:
            logger.error(f"Error generating charts: {e}", exc_info=True)

def main():
    parser = argparse.ArgumentParser(description='Run performance benchmarks on the preprocessing pipeline')
    parser.add_argument('--large', action='store_true', help='Include large documents in benchmark')
    parser.add_argument('--heavy-load', action='store_true', help='Simulate heavy system load during benchmark')
    
    args = parser.parse_args()
    
    try:
        benchmark = PerformanceBenchmark(large_docs=args.large, heavy_load=args.heavy_load)
        benchmark.run_benchmarks()
        
        print("\n=== BENCHMARK COMPLETED ===")
        print(f"Results saved to: {benchmark.results_dir}")
        print(f"Charts saved to: {benchmark.charts_dir}")
        
        return 0
    except Exception as e:
        logger.critical(f"Benchmark failed: {e}", exc_info=True)
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

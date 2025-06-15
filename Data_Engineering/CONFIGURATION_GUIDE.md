# Configuration Guide - Enhanced French Transcript Preprocessing Pipeline v2.0.0

## Overview

The Enhanced French Transcript Preprocessing Pipeline v2.0.0 features a comprehensive configuration management system that allows fine-tuning of all pipeline components for different environments and use cases.

## Configuration Architecture

### Configuration Hierarchy

1. **Default Configuration**: Built-in defaults for all settings
2. **Configuration Files**: JSON files for environment-specific settings
3. **Environment Variables**: Runtime overrides for deployment
4. **Command Line Arguments**: Session-specific overrides

### Configuration Files

#### Available Configuration Files

- `config_default.json`: Production-ready default configuration
- `config_development.json`: Development environment with debug settings
- `config_custom.json`: Your custom configuration (create as needed)

#### Configuration Structure

```json
{
  "mode": "production|development|testing|benchmark",
  "segmentation": {
    "strategy": "auto|sentence_based|word_based|hybrid",
    "min_segment_lines": 2,
    "max_segment_lines": 10,
    "target_words_per_segment": 150,
    "min_words_per_segment": 80,
    "max_words_per_segment": 300,
    "transcript_detection_threshold": 500.0,
    "coherence_threshold": 0.8
  },
  "ml": {
    "enable_enhanced_features": true,
    "enable_target_format": true,
    "enable_tension_detection": true,
    "enable_thematic_analysis": true,
    "enable_temporal_classification": true,
    "enable_conceptual_markers": true,
    "quality_threshold": 0.5,
    "confidence_threshold": 0.7
  },
  "processing": {
    "supported_formats": [".txt", ".docx", ".pdf", ".json"],
    "max_file_size_mb": 100,
    "encoding_detection_confidence": 0.8,
    "enable_memory_optimization": true,
    "enable_parallel_processing": false,
    "max_workers": 4
  },
  "output": {
    "base_output_dir": "preprocessed_data",
    "create_standard_output": true,
    "create_ml_ready_output": true,
    "create_target_format_output": true,
    "preserve_backward_compatibility": true,
    "enable_progress_tracking": true
  },
  "logging": {
    "level": "INFO|DEBUG|WARNING|ERROR",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_path": "preprocessing.log",
    "enable_console_output": true,
    "enable_file_output": true,
    "max_file_size_mb": 10,
    "backup_count": 5
  },
  "nlp": {
    "spacy_model": "fr_core_news_lg",
    "enable_stanza": false,
    "stanza_model": "fr",
    "enable_noun_phrase_extraction": true,
    "enable_discourse_analysis": true,
    "enable_temporal_analysis": true
  }
}
```

## Configuration Options Reference

### Processing Modes

#### Development Mode
- **Purpose**: Development and debugging
- **Features**: Debug logging, progress tracking, reduced thresholds
- **Usage**: `python main.py --mode development`

#### Production Mode
- **Purpose**: Production deployment
- **Features**: Optimized performance, parallel processing, minimal logging
- **Usage**: `python main.py --mode production`

#### Testing Mode
- **Purpose**: Automated testing
- **Features**: Isolated output, warning-level logging, no parallel processing
- **Usage**: `python main.py --mode testing`

#### Benchmark Mode
- **Purpose**: Performance benchmarking
- **Features**: Minimal logging, maximum performance, no progress tracking
- **Usage**: `python main.py --mode benchmark`

### Segmentation Configuration

#### Strategy Options

- **auto**: Automatically detect content type and choose appropriate strategy
- **sentence_based**: Traditional sentence-based segmentation for documents
- **word_based**: Word-based segmentation for transcripts (150-300 words)
- **hybrid**: Combination approach using both strategies

#### Key Parameters

- `target_words_per_segment`: Target segment size in words (default: 150)
- `transcript_detection_threshold`: Average words per sentence to detect transcripts (default: 500)
- `coherence_threshold`: Minimum coherence score for segment boundaries (default: 0.8)

### ML Configuration

#### Feature Extraction Controls

- `enable_enhanced_features`: Enable comprehensive feature extraction
- `enable_target_format`: Generate data.json compatible output
- `enable_tension_detection`: Detect opposing concept patterns
- `enable_thematic_analysis`: Performance vs Legitimacy classification
- `enable_temporal_classification`: 2023/2050 temporal context detection
- `enable_conceptual_markers`: Second-order concept classification

#### Quality Thresholds

- `quality_threshold`: Minimum ML readiness score (default: 0.5)
- `confidence_threshold`: Minimum confidence for classifications (default: 0.7)

### Processing Configuration

#### Performance Settings

- `enable_parallel_processing`: Enable multi-worker processing
- `max_workers`: Number of parallel workers (default: 4)
- `enable_memory_optimization`: Enable memory optimization features

#### File Handling

- `supported_formats`: List of supported file extensions
- `max_file_size_mb`: Maximum file size for processing (default: 100MB)
- `encoding_detection_confidence`: Minimum confidence for encoding detection

## Environment Variables

### Core Variables

```bash
# Processing mode
export PIPELINE_MODE=production

# Output directory
export PIPELINE_OUTPUT_DIR=/path/to/output

# Logging configuration
export PIPELINE_LOG_LEVEL=DEBUG

# Parallel processing
export PIPELINE_PARALLEL=true
export PIPELINE_MAX_WORKERS=8
```

### Advanced Variables

```bash
# Memory optimization
export PIPELINE_MEMORY_OPTIMIZATION=true

# Feature extraction
export PIPELINE_ENABLE_TARGET_FORMAT=true
export PIPELINE_ENABLE_TENSION_DETECTION=true

# Quality thresholds
export PIPELINE_QUALITY_THRESHOLD=0.7
export PIPELINE_CONFIDENCE_THRESHOLD=0.8
```

## Usage Examples

### Basic Configuration Usage

```bash
# Use default production configuration
python main.py

# Use development configuration
python main.py --config config_development.json

# Override specific settings
python main.py --mode development --workers 2 --log-level DEBUG
```

### Custom Configuration Creation

```bash
# Generate sample configuration
python main.py --generate-config my_config.json

# Edit the generated file
# Use custom configuration
python main.py --config my_config.json
```

### Environment-Specific Deployment

```bash
# Development environment
export PIPELINE_MODE=development
export PIPELINE_LOG_LEVEL=DEBUG
python main.py

# Production environment
export PIPELINE_MODE=production
export PIPELINE_PARALLEL=true
export PIPELINE_MAX_WORKERS=8
python main.py
```

### Programmatic Configuration

```python
from config import PipelineConfig, ProcessingMode, SegmentationStrategy

# Create custom configuration
config = PipelineConfig(mode=ProcessingMode.PRODUCTION)

# Customize segmentation
config.segmentation.strategy = SegmentationStrategy.WORD_BASED
config.segmentation.target_words_per_segment = 200

# Customize ML features
config.ml.enable_target_format = True
config.ml.quality_threshold = 0.8

# Customize processing
config.processing.enable_parallel_processing = True
config.processing.max_workers = 6

# Validate configuration
issues = config.validate()
if issues:
    print("Configuration issues:", issues)
else:
    print("Configuration is valid")

# Use configuration
from preprocess_transcripts import TranscriptPreprocessor
preprocessor = TranscriptPreprocessor(
    input_dir="data",
    output_dir="results",
    config=config
)
```

## Configuration Validation

### Automatic Validation

The pipeline automatically validates configuration on startup and reports any issues:

```bash
python main.py --validate-config
```

### Common Validation Issues

1. **Invalid segment size ranges**: `min_words_per_segment >= max_words_per_segment`
2. **Invalid worker count**: `max_workers < 1`
3. **Invalid paths**: Non-existent or inaccessible directories
4. **Invalid thresholds**: Values outside valid ranges (0.0-1.0)

### Manual Validation

```python
from config import PipelineConfig

config = PipelineConfig()
issues = config.validate()

if issues:
    for issue in issues:
        print(f"Configuration issue: {issue}")
else:
    print("Configuration is valid")
```

## Best Practices

### Development Environment

- Use `config_development.json` as starting point
- Enable debug logging: `"level": "DEBUG"`
- Disable parallel processing for easier debugging
- Use smaller segment sizes for faster iteration

### Production Environment

- Use `config_default.json` as starting point
- Enable parallel processing for performance
- Set appropriate memory limits
- Use INFO or WARNING log levels
- Enable all ML features for comprehensive output

### Testing Environment

- Use isolated output directories
- Disable progress tracking
- Use minimal logging
- Set deterministic parameters for reproducible results

### Performance Optimization

- Enable parallel processing: `"enable_parallel_processing": true`
- Optimize worker count based on CPU cores
- Enable memory optimization for large files
- Adjust segment sizes based on content type

## Troubleshooting Configuration

### Common Issues

1. **Configuration file not found**: Check file path and permissions
2. **Invalid JSON format**: Validate JSON syntax
3. **Environment variable conflicts**: Check for conflicting environment variables
4. **Permission issues**: Ensure write access to output directories

### Debug Configuration

```bash
# Enable verbose configuration debugging
python main.py --mode development --log-level DEBUG --verbose

# Validate configuration without processing
python main.py --validate-config --verbose

# Generate and inspect default configuration
python main.py --generate-config debug_config.json
cat debug_config.json
```

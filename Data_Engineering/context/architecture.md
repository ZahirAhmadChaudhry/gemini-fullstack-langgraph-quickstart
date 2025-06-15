# Enhanced French Transcript Preprocessing Pipeline Architecture v2.0.0

## System Overview

The Enhanced French Transcript Preprocessing Pipeline v2.0.0 is a production-ready system designed to transform raw French transcripts into structured, ML-ready data compatible with human-annotated reference formats. The system features intelligent content detection, advanced segmentation strategies, comprehensive configuration management, and target format generation capabilities.

**Key Enhancement**: The v2.0.0 architecture generates output so well-structured and feature-rich that downstream ML pipelines can produce results matching the sophistication of human-annotated data.json reference.

## Enhanced Architecture Diagram v2.0.0

```
┌─────────────────────────┐      ┌──────────────────────┐      ┌───────────────────────┐
│                         │      │                      │      │                       │
│  Configuration          │──────▶  Document Processing │──────▶  Intelligent          │
│  Management             │      │                      │      │  Content Detection    │
│                         │      │                      │      │                       │
└─────────────────────────┘      └──────────────────────┘      └───────────────────────┘
         │                               │                              │
         │                               │                              │
         ▼                               ▼                              ▼
┌─────────────────────────┐      ┌──────────────────────┐      ┌───────────────────────┐
│                         │      │                      │      │                       │
│  PipelineConfig         │      │  OptimizedDocx       │      │  Advanced             │
│  ProcessingMode         │      │  OptimizedPdf        │      │  Segmentation         │
│  Environment Variables  │      │  RobustEncoding      │      │  (Word/Sentence)      │
│                         │      │  Content Cleaning    │      │                       │
└─────────────────────────┘      └──────────────────────┘      └───────────────────────┘
                                                                         │
                                                                         │
                                                                         ▼
┌─────────────────────────┐      ┌──────────────────────┐      ┌───────────────────────┐
│                         │      │                      │      │                       │
│  Target Format          │◀─────│  Enhanced ML         │◀─────│  Enhanced Feature     │
│  Generator              │      │  Ready Formatter     │      │  Extraction           │
│                         │      │                      │      │                       │
└─────────────────────────┘      └──────────────────────┘      └───────────────────────┘
         │                               │                              │
         │                               │                              │
         ▼                               ▼                              ▼
┌─────────────────────────┐      ┌──────────────────────┐      ┌───────────────────────┐
│                         │      │                      │      │                       │
│  data.json Compatible   │      │  ML-Ready JSON       │      │  Temporal Context     │
│  Output                 │      │  with Quality        │      │  Thematic Indicators  │
│  (7 Columns)            │      │  Scoring             │      │  Tension Patterns     │
│                         │      │                      │      │  Conceptual Markers   │
└─────────────────────────┘      └──────────────────────┘      └───────────────────────┘
```

## Enhanced Component Descriptions v2.0.0

### 1. Configuration Management (NEW)

#### PipelineConfig
- **Purpose**: Centralized configuration management for all pipeline components
- **Key Features**:
  - Environment-specific configurations (development, production, testing)
  - Runtime parameter adjustment
  - Environment variable overrides
  - JSON configuration file support
- **Core Methods**:
  - `__init__(mode, config_file)`: Initialize with mode and optional config file
  - `load_from_file(config_file)`: Load configuration from JSON
  - `validate()`: Validate configuration and return issues
  - `setup_logging()`: Configure logging based on settings
- **Interactions**: Provides configuration to all pipeline components

#### Main Entry Point (NEW)
- **Purpose**: Streamlined CLI interface for pipeline operations
- **Key Features**:
  - Command-line argument parsing
  - Configuration validation
  - Multiple processing modes
  - Utility operations (dry-run, list-files, etc.)
- **Core Methods**:
  - `create_argument_parser()`: Setup CLI arguments
  - `apply_cli_overrides()`: Apply command-line overrides
  - `main()`: Main entry point with error handling
- **Interactions**: Orchestrates entire pipeline execution

### 2. Enhanced Document Processing

#### OptimizedDocxProcessor (Enhanced)
- **Purpose**: Memory-efficient processing of DOCX files
- **Key Features**:
  - Controlled memory usage during table processing
  - Garbage collection triggers for large documents
  - Memory monitoring
  - Enhanced content cleaning
- **Core Methods**:
  - `extract_text(docx_path)`: Extracts text with memory optimization
  - `_is_memory_high()`: Monitors memory usage
- **Interactions**: Provides clean text content to the main pipeline

#### Content Cleaning (Enhanced)
- **Purpose**: Comprehensive removal of content annotations
- **Key Features**:
  - Removes [Music], [Applause], [Laughter] annotations ✅ **FIXED**
  - Comprehensive pattern matching for French and English annotations
  - Preserves text structure while removing artifacts
- **Core Methods**:
  - `_remove_timestamps_and_speakers()`: Enhanced cleaning with content annotations
- **Interactions**: Ensures clean text for downstream processing

### 2. Text Processing

#### ImprovedSentenceTokenizer
- **Purpose**: Enhanced sentence tokenization for French text
- **Key Features**:
  - Handles unpunctuated YouTube transcripts
  - French-specific sentence boundary detection
  - Heuristic segmentation
- **Core Methods**:
  - `tokenize(text)`: Tokenizes text into sentences
  - `_tokenize_unpunctuated(text)`: Handles unpunctuated text
  - `_has_sufficient_punctuation(text)`: Detects punctuation levels
- **Interactions**: Provides segmented sentences for feature extraction

### 3. Feature Extraction

#### MlReadyFormatter
- **Purpose**: Creates standardized ML-ready output
- **Key Features**:
  - Four-column structure (segments, features, metadata, IDs)
  - Feature extraction for temporal markers
  - Metadata tracking
- **Core Methods**:
  - `format_segments(segments)`: Formats segments into ML-ready structure
  - `_extract_linguistic_features(text)`: Extracts features from text
  - `_get_temporal_context(segment)`: Identifies temporal context
- **Interactions**: Produces the final JSON output

## Data Flow

1. **Input**: Raw transcript files (DOCX, PDF, TXT)
2. **Document Processing**: Files are loaded and decoded with memory optimization
3. **Text Processing**: Text is tokenized and segmented appropriately
4. **Feature Extraction**: Linguistic features are extracted from segments
5. **Output**: Structured JSON with segments and features for ML pipeline

## Input/Output Specifications

### Input Formats
- **DOCX**: Word documents with tables and formatting
- **PDF**: PDF documents with text content
- **TXT**: Plain text files (potentially with encoding issues)
- **YouTube Transcripts**: Text files with timestamp patterns and minimal punctuation

### Output Format
```json
{
  "source_file": "example.txt",
  "processed_timestamp": "2025-06-11T08:30:42.123Z",
  "segments": [
    {
      "id": "example_seg_001",
      "text": "Segment text content",
      "features": {
        "temporal_context": "2023|2050|unknown",
        "discourse_markers": ["marker_type"],
        "sentence_count": 1,
        "word_count": 10,
        "noun_phrases": ["phrase1", "phrase2"]
      },
      "metadata": {
        "source": "example.txt",
        "segment_lines": 1,
        "position": {
          "start": 0,
          "end": 1
        }
      }
    }
  ]
}
```

## Key Design Decisions

### Memory Management
- **Decision**: Use incremental processing instead of loading entire documents
- **Rationale**: Prevents memory explosion with large documents
- **Implementation**: Page-by-page processing, garbage collection triggers

### Encoding Detection
- **Decision**: Use a "UTF-8 first" approach with French-specific validation
- **Rationale**: Most modern files are UTF-8, but legacy French files need special handling
- **Implementation**: Multi-stage detection with fallbacks and mojibake repair

### YouTube Transcript Handling
- **Decision**: Implement heuristic segmentation for unpunctuated text
- **Rationale**: YouTube transcripts often lack proper punctuation
- **Implementation**: Pattern recognition for sentence boundaries based on timing, pauses, and semantic clues

### ML-Ready Format
- **Decision**: Standardize on a 4-column JSON structure
- **Rationale**: Provides consistent format for ML pipeline while capturing all necessary data
- **Implementation**: `MlReadyFormatter` class with consistent structure

## Configuration Management

The system uses the following configuration mechanisms:

- **Command-line arguments**: For specifying input/output files and processing options
- **Log configuration**: For controlling logging levels and destinations
- **Memory thresholds**: For controlling garbage collection triggers

## Extension Points

### Adding New Document Types
1. Create a new processor class implementing the document processor interface
2. Add detection logic to the document type detection system
3. Register the processor in the main pipeline

### Adding New Features
1. Add feature extraction logic to the MlReadyFormatter
2. Update the output schema documentation
3. Add validation logic to the testing framework

### Supporting New Languages
1. Update the encoding detector with language-specific patterns
2. Modify the sentence tokenizer for language-specific rules
3. Update the feature extractor for language-specific features

## Testing Strategy

### Unit Testing
- Tests for individual components (e.g., encoding detection, sentence tokenization)
- Isolated tests with mock inputs and outputs

### Integration Testing
- Tests for component interactions (e.g., document processing to feature extraction)
- Uses sample files from different categories

### End-to-End Testing
- Full pipeline tests from raw files to ML-ready output
- Validation of output format and content

### Performance Testing
- Benchmarks for memory usage and processing time
- Scaling tests with different file sizes
- Load tests for concurrent processing

## Deployment Considerations

### Dependencies
- Python 3.8+ environment
- Required libraries in requirements.txt
- Sufficient memory based on document size (minimum recommended: 8GB)

### Performance Tuning
- Memory threshold adjustment based on available system resources
- Batch size configuration for large document sets
- Concurrent processing configuration

### Monitoring
- Log output for tracking processing status
- Memory usage monitoring
- Processing time tracking

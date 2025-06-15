# Refactor Plan: French Transcript Preprocessing Pipeline

This refactor plan addresses the issues identified in the preprocessing pipeline, focusing on optimizing it to extract only the 4 key columns needed instead of all 12. The goal is to create a more robust, memory-efficient pipeline specifically for French language transcripts.

## 1. Current State Analysis

### Current Pipeline Structure
The current preprocessing pipeline handles:
- File loading (DOCX, PDF, TXT)
- Encoding normalization
- Text cleaning (removing timestamps, speakers)
- Sentence tokenization
- Segmentation based on discourse markers
- Temporal marker identification (2023/2050)

### Critical Issues Identified
1. **Memory leaks and inefficient processing**:
   - Python-docx table cell recalculation causing O(n²) performance degradation
   - PyMuPDF memory leaks in document processing
   - High memory consumption in spaCy French models

2. **French-specific encoding issues**:
   - Chardet library's false positives in encoding detection
   - Mojibake corruption in accented characters

3. **French NLP accuracy limitations**:
   - spaCy's lemmatizer performs worse than Stanza for morphologically rich French
   - Slow loading times for French models
   - Poor handling of dialectal variations

4. **Missing key functionality for target output**:
   - No tension/paradox extraction ("A vs B" detection)
   - No second-order concept assignment
   - No first-order reformulation mechanisms

## 2. Refactor Goals and Focus

### Required Pipeline Output (4 Key Columns):
1. **Concepts de 2nd ordre** (Second-Order Concepts): High-level socio-economic themes
2. **Items de 1er ordre reformulé** (First-Order Reformulated Items): Tensions in "A vs B" format
3. **Items de 1er ordre (intitulé d'origine)** (Original First-Order Items): Raw quotes
4. **Détails** (Details): Extended excerpts providing context (2-10 lines)

### Performance and Technical Goals:
- Fix memory leaks in document processing
- Implement robust encoding detection for French text
- Optimize segmentation for tension identification
- Balance spaCy and Stanza usage for optimal French processing
- Implement mechanisms to handle larger file sizes efficiently

## 3. Technical Implementation Plan

### 3.1 Memory and Encoding Optimizations

#### A. Memory-Safe Document Processing
```python
class OptimizedDocxProcessor:
    def extract_text(self, docx_path: str) -> str:
        try:
            document = docx.Document(docx_path)
            text = []
            
            # Process document with controlled memory usage
            for para in document.paragraphs:
                text.append(para.text)
                
                # Monitor memory and trigger garbage collection if needed
                if self._is_memory_high():
                    document = None
                    gc.collect()
                    document = docx.Document(docx_path)
            
            return "\n".join(text)
        finally:
            gc.collect()  # Force cleanup
```

#### B. Robust Encoding Detection for French
```python
class RobustEncodingDetector:
    def detect_encoding(self, file_path: str) -> Tuple[str, float]:
        # Try UTF-8 first (most reliable)
        with open(file_path, 'rb') as f:
            raw_bytes = f.read()
            
        try:
            raw_bytes.decode('utf-8')
            return 'utf-8', 1.0
        except UnicodeDecodeError:
            pass
        
        # Cross-validate with multiple libraries for French text
        chardet_result = chardet.detect(raw_bytes)
        
        # Validate encoding is appropriate for French text
        if self._is_valid_french_encoding(chardet_result['encoding']):
            return chardet_result['encoding'], chardet_result['confidence']
        else:
            return 'utf-8', 0.7  # Default to UTF-8 as safest option
```

#### C. Memory Monitoring and Management
```python
def _is_memory_high(self, threshold_mb: int = 100) -> bool:
    """Check if memory usage is above threshold."""
    current_process = psutil.Process(os.getpid())
    memory_info = current_process.memory_info()
    memory_usage_mb = memory_info.rss / 1024 / 1024
    
    return memory_usage_mb > threshold_mb
```

### 3.2 YouTube Transcript Processing

#### A. YouTube Transcript Detection
```python
def _is_youtube_transcript(self, file_path: Path, text_content: str) -> bool:
    """Check if the file is likely a YouTube transcript."""
    # Check filename for YouTube indicators
    filename = file_path.name.lower()
    if "youtube" in filename or "yt" in filename or "transcript" in filename:
        return True
        
    # Check content for YouTube-specific patterns
    youtube_patterns = [
        r'\[\d{1,2}:\d{2}\]',  # [0:00] timestamp format
        r'\d{1,2}:\d{2} - \d{1,2}:\d{2}',  # 0:00 - 0:15 timestamps
        r'YouTube\s*[Tt]ranscript',
        r'Generated automatically by YouTube'
    ]
    
    for pattern in youtube_patterns:
        if re.search(pattern, text_content):
            return True
            
    return False
```

#### B. Improved Sentence Tokenization
```python
class ImprovedSentenceTokenizer:
    def tokenize(self, text: str, aggressive: bool = False) -> List[str]:
        """Tokenize text with enhanced handling for unpunctuated transcripts."""
        # If text has sufficient punctuation, use standard tokenization
        if self._has_sufficient_punctuation(text):
            return self._tokenize_with_spacy(text)
        
        # For YouTube-style unpunctuated text, use enhanced approach
        return self._tokenize_unpunctuated(text, aggressive)
```

### 3.3 ML-Ready Data Formatting

#### A. Standardized Output Structure
```python
class MlReadyFormatter:
    def format_segments(self, segments: List[Dict[str, Any]], 
                       source_file: str, 
                       nlp_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Format segments into ML-ready structure with metadata and features."""
        formatted_data = {
            "source_file": source_file,
            "processed_timestamp": datetime.now().isoformat(),
            "segments": []
        }
        
        for i, segment in enumerate(segments):
            # Extract segment text
            text = segment.get("segment_text", "")
            
            # Generate segment ID
            segment_id = f"{os.path.splitext(source_file)[0]}_seg_{i+1:03d}"
            
            # Extract or calculate features
            features = {
                "temporal_context": self._get_temporal_context(segment),
                "discourse_markers": self._extract_discourse_markers(segment),
                "sentence_count": len(segment.get("text", [])) if isinstance(segment.get("text", []), list) else 1,
                "word_count": len(text.split())
            }
            
            # Add noun phrases if available
            if nlp_results and "noun_phrases" in nlp_results:
                features["noun_phrases"] = [phrase for phrase in nlp_results["noun_phrases"] 
                                          if phrase.lower() in text.lower()]
            
            # Create metadata
            metadata = {
                "source": source_file,
                "segment_lines": features["sentence_count"],
                "position": {
                    "start": segment.get("start_sentence_index", i),
                    "end": segment.get("end_sentence_index", i + features["sentence_count"])
                }
            }
            
            # Create the formatted segment
            formatted_segment = {
                "id": segment_id,
                "text": text,
                "features": features,
                "metadata": metadata
            }
            
            formatted_data["segments"].append(formatted_segment)
        
        return formatted_data
```

#### B. Feature Extraction for ML
```python
def _extract_linguistic_features(self, text: str) -> Dict[str, Any]:
    """Extract linguistic features useful for ML processing."""
    doc = self.nlp_spacy(text)
    
    features = {
        # Basic counts
        "sentence_count": len(list(doc.sents)),
        "word_count": len([token for token in doc if not token.is_punct]),
        
        # NLP features
        "noun_phrases": [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1],
        "temporal_markers": self._identify_temporal_markers(text)
    }
    
    return features
```

#### C. Metadata Tracking
```python
def _extract_metadata(self, segment: Dict[str, Any], source_file: str, index: int) -> Dict[str, Any]:
    """Extract metadata for tracking and provenance."""
    return {
        "source": source_file,
        "original_line_numbers": (segment.get("start_line", index), 
                                 segment.get("end_line", index + 1)),
        "processing_date": datetime.now().isoformat(),
        "is_youtube": segment.get("is_youtube", False),
        "segment_position": index + 1,
        "total_segments": segment.get("total_segments", 1)
    }
```

## 4. Pipeline Component Integration

### 4.1 Core Pipeline Components
1. **TensionPipeline**: Main orchestrator class
2. **RobustDocumentLoader**: Handles different file formats with memory management
3. **FrenchTextCleaner**: French-specific text normalization
4. **TensionSegmenter**: Extract meaningful segments containing tensions
5. **TensionExtractor**: Identify and format paradoxes/tensions
6. **ConceptMapper**: Assign second-order concepts

### 4.2 Pipeline Flow
1. Load and normalize document
2. Clean and preprocess text
3. Segment into meaningful chunks
4. For each segment:
   - Identify temporal context
   - Extract linguistic features (discourse markers, noun phrases)
   - Track position information and metadata
   - Generate unique segment IDs
5. Output ML-ready data with standardized format

## 5. Implementation Priorities

### Priority 1: Core Functionality & Memory Fixes
- Fix memory leaks in document processing
- Implement robust encoding detection
- Add basic tension detection with contrastive markers

### Priority 2: French Language Optimizations
- Implement hybrid spaCy/Stanza approach
- Enhance French temporal marker detection
- Add French-specific text normalization

### Priority 3: Advanced Features
- Improve tension extraction with dependency parsing
- Add sophisticated concept mapping
- Implement validation frameworks

## 6. Testing and Validation

### Basic Test Suite:
1. Memory profiling tests to verify leak fixes
2. French encoding robustness tests with various character sets
3. Benchmark French NLP performance against test cases
4. Test tension extraction on known paradox examples

### Advanced Validation:
1. Compare with manual annotations on subset of data
2. Measure accuracy of second-order concept assignment
3. Validate temporal context detection accuracy
4. Assess overall system performance on unseen data

## 7. Next Steps

1. **Immediate Actions**:
   - Implement memory-safe document processing
   - Add robust French encoding detection
   - Integrate basic tension extraction

2. **Short-Term Goals** (1-2 weeks):
   - Complete all core functionality for 4 columns
   - Test on representative sample datasets
   - Measure and optimize performance

3. **Medium-Term Goals** (2-4 weeks):
   - Enhance accuracy of French-specific components
   - Add advanced validation mechanisms
   - Create detailed documentation

This refactor plan outlines a comprehensive approach to addressing the critical issues in the French transcript preprocessing pipeline while focusing specifically on the 4 key columns required. By prioritizing memory optimization, French language processing enhancements, and targeted functionality for tension extraction and categorization, we can deliver a more efficient and accurate solution for French language opinion analysis.

## 8. June 11, 2025 Update: Revised Approach

After reassessing project requirements, we've shifted focus from analysis to ML-ready data preparation. The preprocessing pipeline is now focused on:

1. **Efficient Document Processing**: Implemented memory-optimized document processors
2. **Robust French Encoding**: Developed French-specific encoding detection and fixing
3. **YouTube Transcript Handling**: Created specialized processing for unpunctuated transcripts 
4. **ML-Ready Formatting**: Standardized JSON output with the 4-column structure

### Updated Component Structure
- **TranscriptPreprocessor**: Main orchestrator class
- **OptimizedDocxProcessor**: Memory-efficient DOCX processing
- **OptimizedPdfProcessor**: Memory-efficient PDF processing 
- **RobustEncodingDetector**: French-specific encoding detection
- **ImprovedSentenceTokenizer**: Enhanced tokenization for YouTube transcripts
- **MlReadyFormatter**: ML-ready data formatting

### ML-Ready Output Format
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

### Next Steps
1. Run full pipeline test with sample transcript collection
2. Perform performance benchmarking on large document sets
3. Coordinate with ML team for data format validation
4. Final documentation review and updates

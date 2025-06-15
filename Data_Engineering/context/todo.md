# TODO: French Transcript Preprocessing Pipeline Refactor

## Immediate Tasks (Week 1)

### 1. Memory Optimization
- [x] Replace `python-docx` table handling with optimized version
- [x] Fix PyMuPDF memory leaks in document processing
- [x] Implement garbage collection triggers after processing large documents
- [x] Add memory usage monitoring in large batch operations

### 2. Encoding Improvements
- [x] Create `RobustEncodingDetector` class with UTF-8 first approach
- [x] Add cross-validation with multiple detection libraries
- [x] Implement French-specific encoding validation patterns
- [x] Fix mojibake patterns for French diacritics

### 3. YouTube Transcript Processing  
- [x] Create `ImprovedSentenceTokenizer` for handling unpunctuated text
- [x] Implement heuristic sentence boundary detection
- [x] Add automatic YouTube transcript format detection
- [x] Create test cases for YouTube transcript processing

## Short-term Tasks (Week 2)

### 4. French Language Processing Enhancements
- [x] Optimize French temporal marker detection
- [x] Implement French text normalization with diacritics handling
- [x] Enhance segmentation for YouTube transcripts
- [ ] Set up hybrid spaCy/Stanza processing pipeline (Optional)

### 5. ML-Ready Output Structure
- [x] Create `MlReadyFormatter` with standardized 4-column structure
- [x] Create structured JSON output with features and metadata
- [x] Add segment ID generation and position tracking
- [x] Implement noun phrase extraction for features

### 6. Testing Framework
- [x] Create test cases for ML formatter functionality
- [x] Implement YouTube transcript processing tests
- [x] Add integration test runner with environment management
- [ ] Create benchmarking scripts for performance measurement

## Medium-term Tasks (Weeks 3-4)

### 7. Advanced Feature Extraction
- [x] Implement enhanced temporal marker detection
- [x] Add discourse marker categorization
- [ ] Add confidence scores for extracted features
- [ ] Add semantic coherence measurements between segments 

### 8. Pipeline Integration
- [x] Update main preprocessing pipeline with new components
- [x] Implement command-line options for different processing needs
- [x] Add progress tracking and logging
- [x] Create structured JSON output for ML pipeline consumption

### 9. Validation and Polish
- [x] Create integration_status.md with test results
- [ ] Run full validation on sample dataset
- [ ] Optimize processing speed for largest documents
- [x] Document YouTube transcript handling considerations

## Technical Debt and Documentation

### 10. Code Quality
- [x] Refactor existing codebase to use new utility modules
- [x] Add error handling with detailed logging
- [x] Add fallback strategies for document processing
- [x] Set up tiered processing approach for different document types

### 11. Documentation
- [x] Create ML-ready format specification
- [x] Document YouTube transcript processing
- [x] Document memory optimization techniques
- [x] Create integration update documentation

## Final Tasks

### 12. Finalization
- [x] Run full pipeline test with sample transcript collection
- [x] Perform performance benchmarking on large document sets
- [x] Coordinate with ML team for data format validation
- [x] Final documentation review and updates

## Future Improvements (Post-Release)

### 13. External Evaluation Framework
- [ ] Develop a human-annotated "gold standard" dataset for segmentation evaluation
- [ ] Implement WindowDiff and Pk metrics for objective segmentation assessment
- [ ] Create an evaluation script to compare system output against gold standard
- [ ] Document benchmark results and potential areas of improvement

### 14. Pipeline Flexibility Enhancements
- [ ] Make threshold and weight parameters configurable via command line or config file
- [ ] Create configuration profiles for different transcript types (formal speeches, interviews, debates)
- [ ] Implement adaptive thresholding based on document characteristics
- [ ] Document configuration options in README and example config files

### 15. "Golden Dataset" Refactoring
- [ ] Review and potentially refactor hardcoded patterns in golden dataset detection
- [ ] Make pattern detection more generalizable to different transcript structures
- [ ] Implement rule-based pattern detection as configurable resource
- [ ] Create test cases to verify generalizability across different transcript styles

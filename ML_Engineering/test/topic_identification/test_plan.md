# Test Plan for Topic Identification Module

## Overview
This test plan outlines the approach for testing the Topic Identification Module, specifically focusing on the keyword extraction functionality. The module is designed to identify key topics in French sustainability texts using different algorithms, including TextRank, TF-IDF, and the newly implemented YAKE! (Yet Another Keyword Extractor) algorithm.

## Test Objectives
1. Verify that the KeywordExtractor class correctly implements all the specified keyword extraction algorithms 
2. Ensure that the YAKE! method correctly extracts relevant keywords from French sustainability texts
3. Confirm that the extracted keywords are meaningful and relevant to the sustainability domain
4. Validate that the system can handle various text formats and content types

## Test Environment
- Python 3.x
- Required libraries: spaCy, networkx, scikit-learn, numpy
- French language model: fr_core_news_lg (spaCy)

## Test Cases

### TC-TI-001: Basic Functionality Testing
- **Purpose**: To verify that the KeywordExtractor class can be instantiated with different methods
- **Test Steps**:
  1. Create KeywordExtractor instances with different methods (textrank, tfidf, yake)
  2. Verify that the instances are created successfully
- **Expected Result**: All instances are created without errors
- **Specific Conditions**: None

### TC-TI-002: TextRank Method Testing
- **Purpose**: To validate the TextRank algorithm implementation
- **Test Steps**:
  1. Initialize KeywordExtractor with method="textrank"
  2. Apply to sample French sustainability texts
  3. Verify keyword extraction produces reasonable results
- **Expected Result**: 
  - Keywords are extracted successfully
  - Extracted keywords are relevant to the input text
  - Scores are assigned to each keyword
- **Specific Conditions**: None

### TC-TI-003: YAKE! Method Testing
- **Purpose**: To validate the YAKE! algorithm implementation
- **Test Steps**:
  1. Initialize KeywordExtractor with method="yake"
  2. Apply to sample French sustainability texts
  3. Verify keyword extraction produces reasonable results
- **Expected Result**: 
  - Keywords are extracted successfully
  - Extracted keywords include multi-word phrases relevant to the input text
  - Scores are assigned to each keyword (higher scores for more relevant keywords after inversion)
- **Specific Conditions**: 
  - YAKE! should work without requiring a corpus
  - Should be able to extract multi-word phrases (n-grams)

### TC-TI-004: Sustainability Terms Boosting
- **Purpose**: To verify that sustainability domain terms are properly boosted
- **Test Steps**:
  1. Create a simple file with sustainability terms
  2. Initialize KeywordExtractor with the sustainability_terms_path
  3. Extract keywords from a text containing those terms
  4. Verify that the sustainability terms have boosted scores
- **Expected Result**: Keywords found in the sustainability terms file have their scores boosted by a factor of 1.5
- **Specific Conditions**: A sustainability terms file must be available

### TC-TI-005: Processing Multiple Segments
- **Purpose**: To verify the module can process multiple text segments
- **Test Steps**:
  1. Prepare a list of text segments
  2. Call the process_segments method
  3. Verify that topics are extracted for all segments
- **Expected Result**: All segments have topics assigned
- **Specific Conditions**: None

### TC-TI-006: Error Handling
- **Purpose**: To verify the module gracefully handles various error conditions
- **Test Steps**:
  1. Test with empty text
  2. Test with very short text
  3. Test with malformed input
- **Expected Result**: The module returns empty results without crashing
- **Specific Conditions**: None

## Success Criteria
1. All test cases pass successfully
2. The YAKE! algorithm extracts relevant keywords from French sustainability texts
3. The system can handle various input scenarios without errors
4. The extracted keywords align with human judgment of important terms in the texts

## Additional Notes
- Manual review of extracted keywords will be necessary to assess quality
- Performance tuning may be required based on the size of input texts
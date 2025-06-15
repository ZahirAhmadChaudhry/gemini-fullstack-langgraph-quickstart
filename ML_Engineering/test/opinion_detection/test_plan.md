# Test Plan for Opinion Detection Module

## Overview
This test plan outlines the approach for testing the Opinion Detection Module, specifically focusing on sentiment analysis functionality. The module is designed to analyze sentiment in French sustainability texts using lexicon-based and transformer-based approaches, with special handling for French language features like negation and contrastive markers.

## Test Objectives
1. Verify that the SentimentAnalyzer class correctly implements all the specified sentiment analysis methods
2. Ensure that the lexicon-based sentiment analysis correctly handles French negation patterns
3. Confirm that contrastive marker detection and handling properly adjusts sentiment scores
4. Validate the transformer-based sentiment analysis implementation and its fallback mechanism
5. Test that the methods perform reasonably well on French sustainability-related text

## Test Environment
- Python 3.10.16
- Required libraries: spaCy, transformers (Hugging Face), PyTorch
- French language model: fr_core_news_lg (spaCy)
- Multilingual BERT model: nlptown/bert-base-multilingual-uncased-sentiment

## Test Cases

### TC-OD-001: Basic Functionality Testing
- **Purpose**: To verify that the SentimentAnalyzer class can be instantiated with different methods
- **Test Steps**:
  1. Create SentimentAnalyzer instances with different methods (lexicon_based, transformer_based)
  2. Verify that the instances are created successfully
- **Expected Result**: All instances are created without errors
- **Specific Conditions**: None

### TC-OD-002: Lexicon-Based Sentiment Analysis
- **Purpose**: To validate the lexicon-based sentiment analysis implementation
- **Test Steps**:
  1. Initialize SentimentAnalyzer with method="lexicon_based"
  2. Apply to sample French sustainability texts with known sentiment
  3. Verify sentiment scores and labels are reasonable
- **Expected Result**: 
  - Sentiment scores and labels are assigned correctly
  - Positive text receives positive scores, negative text receives negative scores
  - Scores are normalized appropriately
- **Specific Conditions**: None

### TC-OD-003: French Negation Handling
- **Purpose**: To verify the module correctly handles French negation patterns
- **Test Steps**:
  1. Create test cases with French negation patterns (ne...pas, jamais, etc.)
  2. Apply lexicon-based sentiment analysis with negation_handling=True
  3. Compare with results when negation_handling=False
- **Expected Result**: 
  - Sentiment polarity is inverted for negated sentiment-bearing words
  - Overall sentiment scores reflect the negation effect
- **Specific Conditions**: 
  - Test with various French negation constructions (ne...pas, ne...jamais, etc.)

### TC-OD-004: Contrastive Marker Detection and Handling
- **Purpose**: To verify the module correctly detects and handles contrastive markers
- **Test Steps**:
  1. Create test cases with contrastive markers (mais, cependant, toutefois, etc.)
  2. Apply sentiment analysis to these test cases
  3. Examine the impact on sentiment scores
- **Expected Result**: 
  - Contrastive markers are detected correctly
  - Sentiment scores are adjusted to give more weight to the clause after the marker
- **Specific Conditions**: 
  - Test with various contrastive markers in different positions

### TC-OD-005: Transformer-Based Sentiment Analysis
- **Purpose**: To validate the transformer-based sentiment analysis implementation
- **Test Steps**:
  1. Initialize SentimentAnalyzer with method="transformer_based"
  2. Apply to sample French sustainability texts
  3. Verify sentiment scores and labels are reasonable
- **Expected Result**: 
  - Sentiment scores and labels are assigned correctly
  - The transformer model successfully captures sentiment in context
  - The star rating system (1-5) is correctly converted to standard sentiment scores (-1 to 1) and labels
- **Specific Conditions**: 
  - Test with the multilingual BERT model that's compatible with PyTorch
  - Verify that star ratings are appropriately mapped to sentiment labels

### TC-OD-006: Sustainability Domain Adaptation
- **Purpose**: To evaluate how well the methods handle sustainability-specific language
- **Test Steps**:
  1. Apply sentiment analysis methods to sustainability-specific texts
  2. Analyze performance on key sustainability terms and concepts
- **Expected Result**: 
  - The methods should handle sustainability terminology reasonably well
- **Specific Conditions**: 
  - Test with texts containing sustainability-specific terminology

### TC-OD-007: Error Handling and Edge Cases
- **Purpose**: To verify the module gracefully handles various error conditions
- **Test Steps**:
  1. Test with TensorFlow models when only PyTorch is available
  2. Test transformer-based analysis when the transformers library is not available
  3. Test with empty text and other edge cases
- **Expected Result**: 
  - The module returns reasonable default values without crashing
  - Appropriate fallback mechanisms are activated when needed
- **Specific Conditions**: None

## Test Data
- Sample French sustainability texts with known sentiment (positive, negative, mixed)
- Sample texts with French negation patterns
- Sample texts with contrastive markers
- French sentiment lexicon with positive and negative terms

## Test Execution
1. Run the test script `test_sentiment_analysis.py` which contains implementations of all test cases
2. Examine the output to verify that the sentiment analysis results match expectations
3. Document any issues encountered during testing in the `test_issues.md` file
4. Record the test results in the `test_report.md` file

## Success Criteria
1. All test cases pass successfully
2. The lexicon-based method correctly handles French negation and contrastive markers
3. The transformer-based method provides accurate sentiment analysis (or falls back gracefully)
4. Both methods handle various input scenarios without errors
5. The sentiment analysis results align with expected sentiment in the test texts

## Additional Notes
- The transformer-based approach will typically provide more nuanced sentiment analysis due to its context awareness
- Performance of lexicon-based methods depends heavily on the quality and coverage of the sentiment lexicon
- The multilingual BERT model was fine-tuned on product reviews, which may affect its performance on sustainability texts
- The star rating system (1-5) of the multilingual BERT model requires conversion to standard sentiment scores and labels
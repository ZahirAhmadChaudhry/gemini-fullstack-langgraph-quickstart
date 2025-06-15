# Test Issues for Opinion Detection Module

This document tracks issues encountered during testing of the Opinion Detection Module.

## Open Issues

No open issues at this time.

## In Progress Issues

No issues currently in progress.

## Resolved Issues

### ISSUE-OD-001: Transformer Fallback Didn't Use Loaded Lexicon
- **Status**: Resolved
- **Date Identified**: 2025-04-22
- **Identified By**: GitHub Copilot
- **Description**: When the transformer-based sentiment analysis fell back to lexicon-based analysis (due to missing transformers library), it returned neutral sentiment for all texts even when the lexicon contained sentiment-bearing words.
- **Steps to Reproduce**: 
  1. Run the test script with transformers library unavailable
  2. Observe that transformer-based analysis returns neutral sentiment for all texts
  3. Compare with lexicon-based analysis results for the same texts, which correctly identify sentiment
- **Expected Behavior**: The fallback should use the loaded lexicon to provide meaningful sentiment analysis
- **Actual Behavior**: Returned neutral sentiment (score: 0.0) for all texts
- **Resolution**: Modified the test script to explicitly pass the lexicon path to the SentimentAnalyzer constructor when creating an instance for transformer-based analysis. This ensures that when the transformers library isn't available, the fallback mechanism has access to the lexicon and can provide meaningful sentiment analysis.
- **Resolution Date**: 2025-04-22

### ISSUE-OD-002: Incompatible TensorFlow Model with PyTorch Backend
- **Status**: Resolved
- **Date Identified**: 2025-04-22
- **Identified By**: GitHub Copilot
- **Description**: The default transformer model "tblard/tf-allocine" was a TensorFlow model, but only PyTorch was installed in the environment, causing errors when trying to load the model.
- **Steps to Reproduce**: 
  1. Install PyTorch but not TensorFlow
  2. Try to use the "tblard/tf-allocine" model for sentiment analysis
  3. Observe error message about TensorFlow being required
- **Expected Behavior**: The system should use a compatible model or provide a clear error message
- **Actual Behavior**: Error message: "TFCamembertForSequenceClassification requires the TensorFlow library but it was not found in your environment."
- **Resolution**: Changed the default transformer model to "nlptown/bert-base-multilingual-uncased-sentiment", which is compatible with PyTorch. Updated both the SentimentAnalyzer class and the test script to use this model.
- **Resolution Date**: 2025-04-22

### ISSUE-OD-003: Star Rating Conversion for Transformer Model Output
- **Status**: Resolved
- **Date Identified**: 2025-04-22
- **Identified By**: GitHub Copilot
- **Description**: The multilingual BERT model outputs star ratings (1-5) instead of standard sentiment labels, requiring conversion to our standard format.
- **Steps to Reproduce**: 
  1. Use the multilingual BERT model for sentiment analysis
  2. Observe that it outputs star ratings like "5 stars", "4 stars", etc.
  3. Notice these don't match our standard positive/negative/neutral labels
- **Expected Behavior**: Consistent sentiment labels across methods
- **Actual Behavior**: Different label formats between lexicon-based and transformer-based methods
- **Resolution**: Implemented a conversion system in the transformer_based_sentiment method to map star ratings to standard sentiment scores and labels. Mapped 1-2 stars to negative, 3 stars to neutral, and 4-5 stars to positive. Also implemented a weighted score calculation to convert star ratings to a normalized score between -1 and 1.
- **Resolution Date**: 2025-04-22

## Issue Template

When adding new issues, please use the following format:

```
### ISSUE-ID: Brief Title
- **Status**: [Open/In Progress/Resolved]
- **Date Identified**: YYYY-MM-DD
- **Identified By**: [Name]
- **Description**: Detailed description of the issue
- **Steps to Reproduce**: 
  1. Step 1
  2. Step 2
  3. ...
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Resolution**: (Fill this in when resolved) How the issue was fixed
- **Resolution Date**: (Fill this in when resolved) YYYY-MM-DD
```
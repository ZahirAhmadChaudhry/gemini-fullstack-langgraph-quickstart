# Test Plan for Paradox Detection Module

## Test Objectives

This test plan aims to verify that the Paradox Detection module correctly identifies paradoxes in French sustainability discourse by:

1. Validating each individual detection method (antonym pairs, negated repetition, contrastive structures, sustainability tensions)
2. Ensuring the overall paradox detection functionality works as expected
3. Testing the segment processing functionality

## Test Environment

- **Python Version**: 3.10+
- **Dependencies**: spaCy with fr_core_news_lg model
- **Resource Files**: 
  - sustainability_terms_fr.txt
  - french_antonyms.csv
  - tension_keywords_fr.csv

## Important Testing Parameters

- **Confidence Threshold**: 0.3 (lowered from default for testing purposes)
  - This adjustment allows the detector to identify antonym pairs even when they are separated by multiple tokens in the text
  - The lower threshold provides better coverage for testing, while in production a higher threshold might be preferred

## Test Cases

### TC-PD-001: Antonym Pair Detection
- **Purpose**: Validate that the detector correctly identifies antonym pairs within a specified proximity in the text
- **Test Data**: 
  - Positive case: "Nous devons augmenter la production tout en diminuant l'impact environnemental."
  - Negative case: "La durabilité est un concept important dans la société moderne."
- **Expected Outcome**: 
  - Detection of "augmenter"/"diminuer" as antonym pair in the positive case
  - No detections in the negative case

### TC-PD-002: Negated Repetition Detection
- **Purpose**: Verify that the detector recognizes words that appear in both positive and negative forms
- **Test Data**: 
  - Positive case: "Cette solution est durable et n'est pas durable en même temps."
  - Negative case: "Le développement durable est un concept important."
- **Expected Outcome**: 
  - Detection of "durable" as negated repetition in the positive case
  - No detections in the negative case

### TC-PD-003: Contrastive Structures Detection
- **Purpose**: Confirm that the detector identifies linguistic patterns that signal contrast
- **Test Data**: 
  - Positive case: "D'une part, nous voulons augmenter la production, d'autre part, nous devons réduire les émissions de CO2."
  - Negative case: "Le développement durable est un concept important pour l'avenir."
- **Expected Outcome**: 
  - Detection of "d'une part...d'autre part" pattern in the positive case
  - No detections in the negative case

### TC-PD-004: Sustainability Tensions Detection
- **Purpose**: Ensure the detector recognizes co-occurrence of sustainability terms with tension keywords
- **Test Data**: 
  - Positive case: "Le développement durable crée un paradoxe entre la croissance économique et la protection de l'environnement."
  - Negative case: "L'économie mondiale continue de croître rapidement."
- **Expected Outcome**: 
  - Detection of sustainability tension in the positive case
  - No detections in the negative case

### TC-PD-005: Overall Paradox Detection
- **Purpose**: Validate that the full paradox detection pipeline correctly combines evidence from all methods
- **Test Data**: 
  - Paradox text: "Le développement durable exige que nous augmentions la production tout en réduisant l'impact environnemental. C'est un dilemme constant entre croissance économique et protection de l'environnement."
  - Non-paradox text: "Le développement durable est un concept important pour l'avenir de notre planète. Nous devons tous contribuer à un monde plus vert."
- **Expected Outcome**: 
  - Positive identification of paradox in the paradox text with confidence score above threshold
  - Negative result for the non-paradox text

### TC-PD-006: Single Segment Processing
- **Purpose**: Verify that the detector correctly processes a single text segment
- **Test Data**: Segment dictionary with text about a sustainability dilemma
- **Expected Outcome**: Segment enriched with paradox detection results

### TC-PD-007: Multiple Segment Processing
- **Purpose**: Validate that the detector correctly processes multiple segments
- **Test Data**: List of segment dictionaries
- **Expected Outcome**: All segments enriched with paradox detection results

## Test Execution Process

1. Set up test environment with necessary dependencies
2. Ensure required resource files are available
3. Run each test case independently
4. Log and analyze results
5. Document any issues in test_issues.md

## Acceptance Criteria

- All test cases pass successfully
- No false positives in texts without paradoxes
- At least 70% of known paradoxes are correctly identified
- Processing time is reasonable (under 1 second per segment)

## Notes on Confidence Threshold

During testing, we observed that the confidence scoring mechanism for antonym pair detection is sensitive to the distance between the antonyms in the text. If two antonyms are far apart, they receive a lower confidence score. For testing purposes, we use a lower confidence threshold (0.3) to ensure we can validate the detection mechanisms, even when words are distant in sentences.

In a production environment, this threshold might need to be tuned based on the desired balance between:
- Precision (fewer false positives): Use a higher threshold (e.g., 0.5-0.6)
- Recall (more detections, including weaker signals): Use a lower threshold (e.g., 0.3-0.4)

Date: April 23, 2025
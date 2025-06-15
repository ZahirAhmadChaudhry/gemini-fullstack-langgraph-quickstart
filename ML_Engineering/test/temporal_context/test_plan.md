# Test Plan for Temporal Context Analysis Module

## Test Objectives

This test plan aims to verify that the Temporal Context Analysis module correctly distinguishes between present (2023) and future (2050) contexts in French sustainability discourse by:

1. Validating the detection of temporal markers (present and future time references)
2. Ensuring the detection of verb tenses functions correctly
3. Validating the disambiguation of present tense used for future reference
4. Testing the overall temporal context analysis functionality
5. Verifying the segment processing capabilities

## Test Environment

- **Python Version**: 3.10+
- **Dependencies**: spaCy with fr_core_news_lg model
- **Resource Files**: 
  - present_markers_fr.csv
  - future_markers_fr.csv

## Test Cases

### TC-TC-001: Temporal Markers Detection
- **Purpose**: Validate that the analyzer correctly identifies temporal markers in text
- **Test Data**: 
  - Present markers text: "Aujourd'hui en 2023, le développement durable est un concept important dans la société moderne."
  - Future markers text: "En 2050, le développement durable sera au cœur des politiques environnementales."
  - Mixed markers text: "Aujourd'hui nous prenons des mesures pour que demain, en 2050, notre planète soit préservée."
- **Expected Outcome**: 
  - Detection of present markers in the present markers text
  - Detection of future markers in the future markers text
  - Detection of both present and future markers in the mixed text

### TC-TC-002: Verb Tense Detection
- **Purpose**: Verify that the analyzer recognizes different verb tenses in French text
- **Test Data**: 
  - Present tense text: "Nous travaillons sur le développement durable. Les entreprises adoptent des pratiques écologiques."
  - Future tense text: "Nous adopterons des politiques environnementales plus strictes. Les entreprises devront s'adapter."
- **Expected Outcome**: 
  - Detection of present tense verbs in the present tense text
  - Detection of future tense verbs (futur simple, futur proche, etc.) in the future tense text

### TC-TC-003: Present Tense Disambiguation
- **Purpose**: Confirm that the analyzer can identify when present tense is used to refer to future events
- **Test Data**: "Demain, nous partons en mission pour sauver la planète."
- **Expected Outcome**: Detection of present tense used for future reference

### TC-TC-004: Overall Temporal Context Analysis
- **Purpose**: Validate that the full temporal context analysis correctly identifies the predominant time reference
- **Test Data**: 
  - Present context text: "Aujourd'hui en 2023, nous constatons une prise de conscience accrue sur les enjeux environnementaux. Les entreprises adoptent progressivement des pratiques durables, mais le chemin est encore long."
  - Future context text: "En 2050, nous aurons transformé notre économie pour qu'elle soit entièrement circulaire. Les entreprises adopteront des pratiques zéro déchet, et les énergies renouvelables domineront le marché."
  - Ambiguous text: "Le développement durable représente un défi majeur pour notre société. Il faut repenser nos modèles économiques."
- **Expected Outcome**: 
  - Identification of present context for the present text with confidence score
  - Identification of future context for the future text with confidence score
  - Appropriate handling of ambiguous text

### TC-TC-005: Single Segment Processing
- **Purpose**: Verify that the analyzer correctly processes a single text segment
- **Test Data**: 
  - Present segment: Text about current environmental challenges
  - Future segment: Text about future sustainability developments
- **Expected Outcome**: Segments enriched with appropriate temporal context information

### TC-TC-006: Multiple Segment Processing
- **Purpose**: Validate that the analyzer correctly processes multiple text segments
- **Test Data**: List of segments with present, future, and ambiguous temporal contexts
- **Expected Outcome**: All segments enriched with appropriate temporal context information

## Test Execution Process

1. Set up test environment with necessary dependencies
2. Ensure required resource files are available
3. Run each test case independently
4. Log and analyze results
5. Document any issues in test_issues.md

## Acceptance Criteria

- All test cases pass successfully
- Accuracy in temporal context distinction:
  - At least 90% accuracy for texts with explicit temporal markers
  - At least 75% accuracy for texts with implicit temporal references
- Processing time is reasonable (under 1 second per segment)

## Special Testing Considerations

1. **French Language Specificity**: Testing must account for the specific ways temporal context is expressed in French, including:
   - Use of present tense to refer to future events
   - Various forms of future tense (futur simple, futur proche, conditionnel)
   - French-specific temporal markers

2. **Confidence Scoring**: The confidence score mechanism should be tested to ensure it accurately reflects the certainty of temporal context determination.

3. **Ambiguous Cases**: Special attention should be paid to texts with mixed or ambiguous temporal references to ensure the system makes reasonable determinations.

Date: April 23, 2025
# ML Team Coordination: Data Format Validation

## Overview

This document outlines the coordination process between the Data Engineering and ML teams to validate the standardized ML-ready data format. It provides guidelines for reviewing and testing the preprocessing pipeline output to ensure it meets the ML pipeline's requirements.

## Data Format Specification

The preprocessing pipeline produces JSON files with the following structure:

```json
{
  "source_file": "string",
  "processed_timestamp": "ISO-8601 timestamp",
  "segments": [
    {
      "id": "string",
      "text": "string",
      "features": {
        "temporal_context": "2023|2050|unknown",
        "discourse_markers": ["array of strings"],
        "sentence_count": integer,
        "word_count": integer,
        "noun_phrases": ["array of strings"]
      },
      "metadata": {
        "source": "string",
        "segment_lines": integer,
        "position": {
          "start": integer,
          "end": integer
        }
      }
    }
  ]
}
```

## Validation Checklist

### Basic Format Validation

- [ ] All required fields are present in the output file
- [ ] Data types match the expected types in the schema
- [ ] All segment IDs follow the expected format (`{filename}_seg_{number}`)
- [ ] No invalid or null values in required fields
- [ ] Temporal context is limited to allowed values

### Feature Validation

- [ ] `temporal_context` value is one of: "2023", "2050", or "unknown"
- [ ] `discourse_markers` contains valid discourse markers
- [ ] `noun_phrases` extraction properly identifies French noun phrases
- [ ] `sentence_count` matches the actual number of sentences in the segment text
- [ ] `word_count` matches the actual number of words in the segment text

### Content Validation

- [ ] Segments are properly extracted from documents
- [ ] French diacritics are preserved correctly
- [ ] YouTube transcripts with missing punctuation are properly segmented
- [ ] Unicode normalization is applied consistently

## ML Pipeline Integration Points

### Data Loading

Example Python code for loading the ML-ready data:

```python
import json
import pandas as pd

def load_transcript_data(file_path):
    """Load preprocessed transcript data into pandas DataFrame."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract segments into DataFrame rows
    rows = []
    for segment in data["segments"]:
        row = {
            "segment_id": segment["id"],
            "text": segment["text"],
            "temporal_context": segment["features"]["temporal_context"],
            "discourse_markers": segment["features"]["discourse_markers"],
            "sentence_count": segment["features"]["sentence_count"],
            "word_count": segment["features"]["word_count"],
            "source_file": segment["metadata"]["source"],
            "position_start": segment["metadata"]["position"]["start"],
            "position_end": segment["metadata"]["position"]["end"]
        }
        
        # Add noun phrases if available
        if "noun_phrases" in segment["features"]:
            row["noun_phrases"] = segment["features"]["noun_phrases"]
        
        rows.append(row)
    
    return pd.DataFrame(rows)
```

### Feature Processing

Guidelines for processing the features in the ML pipeline:

1. **Temporal Context**
   - Use as a categorical feature (one-hot encode)
   - Can be used to filter segments for time-specific analysis

2. **Discourse Markers**
   - Convert to multi-hot encoding
   - Use as features for tension detection

3. **Noun Phrases**
   - Can be used for concept extraction
   - Should be vectorized using appropriate NLP techniques

## Validation Process

1. **Sample Data Review**
   - ML team reviews a set of sample preprocessed files
   - Verifies all required fields are present and correctly formatted

2. **Integration Testing**
   - Run ML data loading code on sample preprocessed files
   - Ensure features can be extracted and processed correctly 

3. **Model Compatibility Testing**
   - Run initial model training on preprocessed data
   - Validate that models can effectively utilize the features

## Feedback Mechanism

When issues are found in the preprocessed data:

1. Create a ticket in the issue tracker with detailed information:
   - File that exhibited the issue
   - Exact nature of the issue
   - Expected vs. actual output

2. Include a minimal reproducible example if possible

3. Tag both ML and Data Engineering team leads

## Coordination Schedule

- Initial Data Format Review: June 15, 2025
- Integration Testing: June 18, 2025
- Model Compatibility Testing: June 22, 2025
- Final Format Validation: June 25, 2025

## ML Team Contacts

- ML Team Lead: [Name]
- ML Engineer (Feature Processing): [Name]
- ML Engineer (Model Development): [Name]

## Data Engineering Team Contacts

- Data Engineering Lead: [Name]
- Preprocessing Pipeline Developer: [Name]

## Next Steps

1. Share sample preprocessed files with ML team
2. Schedule initial review meeting
3. Document feedback and required changes
4. Implement and validate any necessary adjustments

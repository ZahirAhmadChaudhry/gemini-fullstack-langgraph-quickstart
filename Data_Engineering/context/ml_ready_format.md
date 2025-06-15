# ML-Ready Data Format Specification

## Overview

This document provides the detailed specification for the ML-ready data format produced by the French Transcript Preprocessing Pipeline. This format is specifically designed to facilitate machine learning tasks by providing structured, annotated transcript segments with associated features and metadata.

## File Structure

Each preprocessed transcript is saved as a JSON file with the following high-level structure:

```json
{
  "source_file": "string",
  "processed_timestamp": "ISO-8601 timestamp",
  "segments": [
    {
      "id": "string",
      "text": "string",
      "features": {
        // Feature dictionary
      },
      "metadata": {
        // Metadata dictionary
      }
    },
    // Additional segments...
  ]
}
```

## Field Descriptions

### Top-Level Fields

| Field | Type | Description |
|-------|------|-------------|
| `source_file` | string | Original source filename |
| `processed_timestamp` | string | ISO-8601 timestamp of when the file was processed |
| `segments` | array | Array of segment objects |

### Segment Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier for the segment (format: `{filename}_seg_{number}`) |
| `text` | string | Full text content of the segment |
| `features` | object | Dictionary of extracted linguistic features |
| `metadata` | object | Dictionary of metadata about the segment |

### Features Object

| Field | Type | Description |
|-------|------|-------------|
| `temporal_context` | string | One of: "2023" (present), "2050" (future), "unknown" |
| `discourse_markers` | array | List of discourse markers found in the segment |
| `sentence_count` | integer | Number of sentences in the segment |
| `word_count` | integer | Number of words in the segment |
| `noun_phrases` | array | List of extracted noun phrases (if available) |

### Metadata Object

| Field | Type | Description |
|-------|------|-------------|
| `source` | string | Source filename |
| `segment_lines` | integer | Number of original lines in this segment |
| `position` | object | Position information with `start` and `end` indices |

## Example

```json
{
  "source_file": "climate_interview.txt",
  "processed_timestamp": "2025-06-11T08:30:42.123Z",
  "segments": [
    {
      "id": "climate_interview_seg_001",
      "text": "Aujourd'hui, nous devons agir pour le climat. Les températures augmentent régulièrement et les événements météorologiques extrêmes deviennent plus fréquents.",
      "features": {
        "temporal_context": "2023",
        "discourse_markers": ["temporal"],
        "sentence_count": 2,
        "word_count": 21,
        "noun_phrases": [
          "le climat",
          "Les températures",
          "les événements météorologiques extrêmes"
        ]
      },
      "metadata": {
        "source": "climate_interview.txt",
        "segment_lines": 2,
        "position": {
          "start": 0,
          "end": 2
        }
      }
    },
    {
      "id": "climate_interview_seg_002",
      "text": "En 2050, si nous ne faisons rien, la température moyenne aura augmenté de plusieurs degrés. Les conséquences seront désastreuses pour la biodiversité.",
      "features": {
        "temporal_context": "2050",
        "discourse_markers": ["conditional"],
        "sentence_count": 2,
        "word_count": 23,
        "noun_phrases": [
          "la température moyenne",
          "plusieurs degrés",
          "Les conséquences",
          "la biodiversité"
        ]
      },
      "metadata": {
        "source": "climate_interview.txt",
        "segment_lines": 2,
        "position": {
          "start": 3,
          "end": 5
        }
      }
    }
  ]
}
```

## Usage in ML Pipelines

### Loading the Data

```python
import json
import pandas as pd

def load_ml_ready_data(file_path):
    """Load ML-ready data and convert to a pandas DataFrame."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create DataFrame from segments
    segments = []
    for segment in data["segments"]:
        # Flatten the structure for easier processing
        segment_dict = {
            "id": segment["id"],
            "text": segment["text"],
            "temporal_context": segment["features"]["temporal_context"],
            "discourse_markers": "|".join(segment["features"]["discourse_markers"]),
            "sentence_count": segment["features"]["sentence_count"],
            "word_count": segment["features"]["word_count"],
            "source_file": segment["metadata"]["source"],
            "position_start": segment["metadata"]["position"]["start"],
            "position_end": segment["metadata"]["position"]["position_end"]
        }
        segments.append(segment_dict)
    
    return pd.DataFrame(segments)
```

### Feature Engineering Example

```python
from sklearn.feature_extraction.text import TfidfVectorizer

def prepare_features(df):
    """Prepare features for ML model training."""
    # Create text features using TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000)
    text_features = vectorizer.fit_transform(df["text"])
    
    # Create additional features
    df["is_present"] = df["temporal_context"] == "2023"
    df["is_future"] = df["temporal_context"] == "2050"
    df["has_discourse_markers"] = df["discourse_markers"] != ""
    
    # One-hot encode discourse markers
    discourse_markers = pd.get_dummies(
        df["discourse_markers"].str.split("|").explode(),
        prefix="marker"
    )
    
    return text_features, df[["is_present", "is_future", "has_discourse_markers"]], discourse_markers
```

## Implementation Notes

- The ML-ready format is generated by the `MlReadyFormatter` class in `utils/ml_formatter.py`
- Noun phrases are extracted using the spaCy French language model
- Temporal context is determined through a combination of rule-based detection and linguistic analysis
- Discourse markers are categorized based on their functional role in the text

## File Location

ML-ready files are saved in the `preprocessed_data/ml_ready_data/` directory with the naming convention `{original_filename}_ml_ready.json`.

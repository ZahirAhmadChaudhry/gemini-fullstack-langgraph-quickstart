# Data Formats Documentation

This document describes the input and output data formats for the baseline NLP system.

## Input Data Format

The system expects preprocessed data in JSON format with the following structure:

```json
{
  "filename": "Table_X_preprocessed.json",
  "segments": [
    {
      "text": ["String of text from segment 1", "Optional additional line"],
      "has_discourse_marker": true,
      "discourse_marker_type": "contrast",
      "temporal_markers": {
        "2023_reference": true,
        "2050_reference": false
      },
      "source_file": "Table_X.docx",
      "segment_id": "Table_X.docx_123"
    },
    {
      "text": ["String of text from segment 2"],
      "has_discourse_marker": false,
      "discourse_marker_type": null,
      "temporal_markers": {
        "2023_reference": false,
        "2050_reference": false
      },
      "source_file": "Table_X.docx",
      "segment_id": "Table_X.docx_124"
    }
  ]
}
```

### Key Input Fields

| Field | Type | Description |
|-------|------|-------------|
| `filename` | string | Name of the preprocessed file |
| `segments` | array | Array of text segment objects |
| `segments[].text` | array | Array of strings representing lines of text in the segment |
| `segments[].has_discourse_marker` | boolean | Whether segment contains a discourse marker |
| `segments[].discourse_marker_type` | string/null | Type of discourse marker if present |
| `segments[].temporal_markers` | object | Explicit temporal references |
| `segments[].source_file` | string | Original source file name |
| `segments[].segment_id` | string | Unique identifier for the segment |

## Output Data Format

The system produces labeled output in JSON format with the following structure:

```json
{
  "filename": "Table_X_labeled.json",
  "segments": [
    {
      "text": ["String of text from segment 1", "Optional additional line"],
      "has_discourse_marker": true,
      "discourse_marker_type": "contrast",
      "temporal_markers": {
        "2023_reference": true,
        "2050_reference": false
      },
      "source_file": "Table_X.docx",
      "segment_id": "Table_X.docx_123",
      "topics": [
        {
          "term": "développement durable",
          "score": 0.178,
          "is_sustainability_term": true
        },
        {
          "term": "économie",
          "score": 0.145,
          "is_sustainability_term": false
        }
      ],
      "sentiment": {
        "score": -0.25,
        "raw_score": -0.15,
        "magnitude": 0.4,
        "label": "negative",
        "details": {
          "negations": 1,
          "intensifiers": 0,
          "contrastive_markers": 1
        }
      },
      "paradox": {
        "is_paradox": true,
        "confidence": 0.75,
        "detections": [
          {
            "method": "antonym_pair",
            "evidence": {
              "term1": "augmentation",
              "term2": "diminution",
              "distance": 8
            }
          }
        ]
      },
      "temporal_context": {
        "context": "present",
        "confidence": 0.85,
        "evidence": {
          "markers": {
            "present": ["aujourd'hui"],
            "future": []
          },
          "tenses": {
            "present": 5,
            "futur_simple": 0,
            "conditionnel": 0,
            "present_for_future": 1
          }
        }
      }
    }
  ]
}
```

### Key Output Fields

| Field | Type | Description |
|-------|------|-------------|
| All input fields | various | All fields from the input are preserved |
| `segments[].topics` | array | Array of topic objects |
| `segments[].topics[].term` | string | Extracted topic term |
| `segments[].topics[].score` | float | TextRank score (0-1) |
| `segments[].topics[].is_sustainability_term` | boolean | Whether term is in sustainability lexicon |
| `segments[].sentiment` | object | Sentiment analysis results |
| `segments[].sentiment.score` | float | Normalized sentiment score (-1 to 1) |
| `segments[].sentiment.raw_score` | float | Raw sentiment score from lexicon |
| `segments[].sentiment.magnitude` | float | Sentiment strength/intensity |
| `segments[].sentiment.label` | string | "positive", "negative", or "neutral" |
| `segments[].sentiment.details` | object | Supporting details about sentiment analysis |
| `segments[].paradox` | object | Paradox detection results |
| `segments[].paradox.is_paradox` | boolean | Whether paradox was detected |
| `segments[].paradox.confidence` | float | Confidence score (0-1) |
| `segments[].paradox.detections` | array | Array of specific paradox detections |
| `segments[].temporal_context` | object | Temporal context analysis results |
| `segments[].temporal_context.context` | string | "present" (2023) or "future" (2050) |
| `segments[].temporal_context.confidence` | float | Confidence score (0-1) |
| `segments[].temporal_context.evidence` | object | Supporting evidence for classification |

## File Naming Conventions

### Input Files
- Location: `preprocessed_data_by_Data_Engineer/`
- Pattern: `Table_[A-H]_preprocessed.json`

### Output Files
- Location: `labeled_output/`
- Pattern: `Table_[A-H]_labeled.json`

## Data Volume

- Input: 8 JSON files corresponding to Tables A through H
- Each file contains varying numbers of segments (typically 100-300 per table)
- Total segments across all tables: Approximately 1,500-2,000
- Average segment size: 2-3 sentences
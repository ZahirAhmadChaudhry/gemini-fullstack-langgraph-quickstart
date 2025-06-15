# Baseline NLP System Architecture Overview

## System Purpose

This baseline NLP system analyzes French sustainability discourse from academic roundtable discussions. It processes text segments to identify topics, detect sentiment, recognize paradoxes, and distinguish temporal contexts (present vs. future) in sustainability discussions.

## High-Level Architecture

The system follows a modular pipeline architecture:

```
┌─────────────────┐     ┌────────────────────────────────────────────────┐     ┌────────────────┐
│  Preprocessed   │     │                 NLP Pipeline                   │     │                │
│   JSON Input    │────▶│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────┐ │────▶│  Labeled JSON  │
│    (8 tables)   │     │  │ Topic   │  │ Opinion │  │ Paradox │  │Temp.│ │     │    Output     │
│                 │     │  │ Ident.  │──▶│ Detect │──▶│ Detect │──▶│Cont.│ │     │               │
└─────────────────┘     │  └─────────┘  └─────────┘  └─────────┘  └─────┘ │     └────────────────┘
                        └────────────────────────────────────────────────┘
```

## Core Components

1. **Topic Identification**: Uses the TextRank algorithm to extract and rank relevant topics in each text segment
2. **Opinion Detection**: Employs a lexicon-based approach with French adaptations to analyze sentiment
3. **Paradox Detection**: Applies rule-based methods to detect linguistic paradoxes in sustainability discourse
4. **Temporal Context Distinction**: Implements rule-based classification to distinguish between present (2023) and future (2050) contexts

## Data Flow

1. **Input**: JSON files containing preprocessed text segments from 8 tables of French sustainability discourse
2. **Processing**: Each text segment passes through all four NLP components sequentially
3. **Output**: JSON files with the original text segments plus topic labels, sentiment scores, paradox flags, and temporal context classifications

## Technical Stack

- **Language**: Python 3.10+
- **Primary Libraries**: 
  - spaCy (with fr_core_news_lg model)
  - NLTK
  - scikit-learn
  - Custom rule-based implementations
- **Input/Output Format**: JSON

## System Boundaries

This system is designed as a standalone analysis pipeline. It:
- Does NOT include a web interface or API
- Does NOT have continuous learning capabilities
- Requires preprocessed, formatted JSON input
- Produces labeled JSON output for further analysis or visualization

See the component-specific documentation for detailed information about each module's implementation, performance considerations, and resource requirements.
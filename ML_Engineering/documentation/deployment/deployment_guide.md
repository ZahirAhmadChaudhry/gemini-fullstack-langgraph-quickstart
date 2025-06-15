# Deployment Guide

This guide provides instructions for the MLOps Engineer to deploy and maintain the baseline NLP system for French sustainability discourse analysis.

## System Requirements

- **Python Version**: 3.10+
- **Memory**: At least 2GB RAM (4GB+ recommended)
- **Storage**: At least 1GB for codebase, dependencies, and models
- **Operating System**: Platform-independent (tested on Windows, Linux, and macOS)

## Dependencies

The following key dependencies are required:

- spaCy 3.5+ with fr_core_news_lg model (~500MB)
- NLTK 3.8+
- scikit-learn 1.2+
- networkx 2.8+
- pandas 1.5+

A complete list is available in the `requirements.txt` file.

## Installation Steps

1. **Create a Virtual Environment**:

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Using conda
conda create -n nlp_baseline python=3.10
conda activate nlp_baseline
```

2. **Install Dependencies**:

```bash
# Using pip
pip install -r requirements.txt

# Using pip with uv (faster installation)
uv pip install -r requirements.txt
```

3. **Download Required Models**:

```bash
# Download spaCy's French language model
python -m spacy download fr_core_news_lg

# Download NLTK resources (if needed)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Running the Pipeline

The main entry point for the system is `run_nlp_pipeline.py` in the root of the ML_Engineering directory.

### Basic Usage

```bash
python run_nlp_pipeline.py
```

This will:
1. Read all preprocessed JSON files from `preprocessed_data_by_Data_Engineer/`
2. Process each file through the NLP pipeline
3. Write labeled output files to `labeled_output/`

### Advanced Options

The script supports several command-line arguments:

```bash
# Process a specific input file
python run_nlp_pipeline.py --input_file="preprocessed_data_by_Data_Engineer/Table_A_preprocessed.json"

# Specify output directory
python run_nlp_pipeline.py --output_dir="custom_output"

# Control logging verbosity
python run_nlp_pipeline.py --log_level=DEBUG

# Use a specific configuration file
python run_nlp_pipeline.py --config="configs/custom_config.json"
```

## Configuration

The system uses a configuration system to control component-specific parameters. The default configuration is in `baseline_nlp/config.py`.

To customize configuration, create a JSON file with the following structure:

```json
{
  "topic_identification": {
    "algorithm": "textrank",
    "window_size": 5,
    "top_n": 5,
    "iterations": 100
  },
  "opinion_detection": {
    "use_linguistic_rules": true,
    "threshold": 0.1
  },
  "paradox_detection": {
    "proximity_threshold": 15,
    "confidence_threshold": 0.5
  },
  "temporal_context": {
    "confidence_threshold": 0.6,
    "default_context": "present"
  }
}
```

## Monitoring and Logging

The system uses Python's logging module. Logs are written to:
1. Console (stderr) for basic information
2. `logs/pipeline.log` for detailed operation logs

Log levels can be configured via command-line or in the config file.

## Resource Management

### Memory Considerations

The spaCy language model is the most memory-intensive component. To optimize:

- Load the model once at the beginning of processing
- Share the model across components
- Process data in batches if memory is limited

### Processing Time

For the complete dataset (8 tables, ~2000 segments):
- Expected processing time: 5-15 minutes depending on hardware
- Most time-intensive components:
  - Topic identification (TextRank algorithm)
  - spaCy language model initialization

## Error Handling

The pipeline implements the following error-handling strategies:

- Graceful degradation for component failures
- Exception capture and logging
- Continuation with partial results when possible

Errors are logged to `logs/errors.log` with full stack traces.

## Scaling Considerations

For larger datasets:

- Implement parallel processing (code scaffold in `baseline_nlp/utils/parallel.py`)
- Consider distributing workload across multiple processes
- Add checkpointing to save progress during long runs

## Updating Components

Each component is designed to be modular:

- To update lexicons or rule-based resources, edit files in the respective `resources` directory
- To modify algorithm parameters, update the configuration file
- To replace a component entirely, implement the same interface as defined in the component's `__init__.py`

## Performance Evaluation

A basic evaluation script is provided in `evaluation/evaluate_performance.py`:

```bash
# Run evaluation against expert-labeled gold standard
python evaluation/evaluate_performance.py --gold_data="evaluation/expert_sample.json"
```

The script generates precision, recall, and F1 scores for paradox detection and accuracy metrics for other components.
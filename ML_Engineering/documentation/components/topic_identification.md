# Topic Identification Component

## Overview

The topic identification component extracts and ranks key terms from French sustainability discourse text segments. It uses the TextRank algorithm, an adaptation of Google's PageRank, to identify the most important terms in each text segment without requiring training data.

## Algorithm

The implementation follows the TextRank approach with adaptations for French sustainability discourse:

1. **Preprocessing**:
   - Text normalization (lowercasing, accent normalization)
   - Tokenization using spaCy's French language model
   - Part-of-speech filtering (retaining only nouns, adjectives, verbs)
   - Stopword removal using a custom French stopword list

2. **Graph Construction**:
   - Create a graph where nodes are candidate words
   - Create edges between words that co-occur within a window (default: 5 words)
   - Weight edges based on co-occurrence frequency

3. **Ranking**:
   - Apply the TextRank algorithm to score terms based on their importance in the graph structure
   - Iterate until convergence or a maximum number of iterations

4. **Term Selection**:
   - Select top N terms (default: 5) with highest TextRank scores
   - Flag terms that match the sustainability lexicon

## Dependencies

- **Primary Library**: spaCy with fr_core_news_lg model
- **Secondary Libraries**: networkx (for graph operations), scikit-learn

## Resources

- **Sustainability Term Lexicon**: A curated list of French sustainability terms located at `baseline_nlp/topic_identification/resources/sustainability_terms_fr.txt`
- **Custom Stopwords**: Extended French stopwords list located at `baseline_nlp/topic_identification/resources/french_stopwords.txt`

## Input/Output

### Input
- Text segment (string or list of strings)
- Optional parameters:
  - window_size: Co-occurrence window size (default: 5)
  - top_n: Number of terms to return (default: 5)
  - iterations: Maximum TextRank iterations (default: 100)

### Output
- List of topic dictionaries with:
  - `term`: The extracted term (string)
  - `score`: TextRank score (float)
  - `is_sustainability_term`: Boolean flag indicating if term is in sustainability lexicon

## Example Output

```json
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
  },
  {
    "term": "transition",
    "score": 0.116,
    "is_sustainability_term": true
  }
]
```

## Performance Considerations

- **Computational Complexity**: O(E×I×N) where E=edges, I=iterations, N=nodes
- **Memory Usage**: Proportional to vocabulary size (nodes) and co-occurrence patterns (edges)
- **Runtime**: Typically fast for individual segments but scales with text length
- **spaCy Model Size**: fr_core_news_lg model requires ~500MB memory

## Limitations

- Reliance on co-occurrence patterns may miss some semantically related terms
- Performs better on longer text segments with sufficient context
- May not capture multi-word expressions efficiently without additional preprocessing
- Knowledge-based only (no semantic understanding)

## Future Improvements

- Incorporate word embeddings for semantic similarity
- Implement phrase detection for multi-word expressions
- Add domain adaptation techniques for sustainability discourse
- Implement hyperparameter tuning based on corpus statistics
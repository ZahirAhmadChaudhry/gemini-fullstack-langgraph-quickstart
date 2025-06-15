Looking at your current pipeline and the target output from the data analysis document, there's a significant gap between what you have now and what you need. Let me break this down and discuss the options with you.

## Current State vs Target State

**What you have now:**
- Basic preprocessing (encoding, tokenization, segmentation)
- Temporal marker detection (2023/2050)
- Text segments of 2-10 lines
- Discourse marker identification

**What you need (based on data_labelled.xlsx):**
- 12 columns of hierarchical categorization
- Tension/paradox extraction ("A vs B" format)
- Theme classification (Legitimacy vs Performance)
- Second-order concept grouping
- Synthesis generation
- Imaginaire analysis (IFa/IFr classification)

## Key Decision Points

Let me present scenarios for the main components we need to add:

### 1. **Tension/Paradox Detection**

This is the core of your target output - identifying "A vs B" tensions in the text.

**Option A: Rule-based with discourse markers (Fastest)**
- Use your existing discourse markers (mais, cependant, versus)
- Extract noun phrases around these markers
- Simple pattern: [NP1] + [marker] + [NP2] = "NP1 vs NP2"
- **Pros:** Quick to implement, leverages existing work
- **Cons:** Will miss implicit tensions, limited accuracy

**Option B: Dependency parsing approach**
- Use spaCy to find opposing concepts through dependency trees
- Look for contrastive relationships
- **Pros:** More accurate, catches more tensions
- **Cons:** Requires more development time

**My recommendation:** Start with Option A for baseline, plan to enhance with B later.

### 2. **Theme Classification (Legitimacy vs Performance)**

**Option A: Keyword-based classification**
```python
legitimacy_keywords = ["transparence", "éthique", "responsabilité", "sociétal", "environnement"]
performance_keywords = ["rentabilité", "profit", "croissance", "efficacité", "productivité"]
```
- **Pros:** Very fast to implement
- **Cons:** ~70-80% accuracy

**Option B: Train a simple classifier**
- Use your labeled data to train SVM or Random Forest
- **Pros:** Better accuracy
- **Cons:** Requires labeled training data, more time

**My recommendation:** Option A for now - it's good enough for baseline.

### 3. **Second-Order Concept Mapping**

**Option A: Fixed taxonomy mapping**
```python
concept_map = {
    "accumulation": "Accumulation/Partage",
    "richesse": "Accumulation/Partage", 
    "croissance": "croissance/soutenabilité",
    "durabilité": "croissance/soutenabilité"
}
```
- **Pros:** Predictable, fast
- **Cons:** Limited to predefined concepts

**Option B: Clustering approach**
- Use embeddings to cluster similar tensions
- **Pros:** Can discover new concepts
- **Cons:** Requires tuning, less predictable

**My recommendation:** Option A - you already know the ~15 concepts from analysis.

### 4. **Synthesis Generation**

**Option A: Template-based**
```python
template = "Tension entre {concept1} et {concept2}"
```
- **Pros:** Consistent format, fast
- **Cons:** Generic summaries

**Option B: Extractive summarization**
- Pick the most relevant sentence from segment
- **Pros:** More natural
- **Cons:** May not always capture the tension clearly

**My recommendation:** Option A for speed and consistency.

## Proposed Implementation Plan

Here's my suggested refactored plan for the fastest baseline:

### Phase 1: Enhance Current Pipeline (1-2 days)
1. Add tension extraction to `_segment_text()`:
   ```python
   def extract_tensions(self, segment_text):
       # Look for contrastive markers
       if any(marker in segment_text for marker in ["mais", "cependant", "versus"]):
           # Extract surrounding noun phrases
           # Return as "A vs B" format
   ```

2. Enhance temporal detection to output clean "2023" or "2050" labels

3. Add session tracking from filename/metadata

### Phase 2: Add Classification Layers (2-3 days)
1. Implement keyword-based theme classifier
2. Create fixed concept mapping dictionary
3. Add imaginaire classifier (sentiment-based for IFa/IFr)

### Phase 3: Output Generation (1 day)
1. Map segments to 12-column format
2. Generate CSV output
3. Add confidence scores for manual review

## Refactored Project Structure

```
preprocessing_pipeline/
├── core/
│   ├── preprocessor.py (existing, enhanced)
│   ├── tension_extractor.py (new)
│   ├── theme_classifier.py (new)
│   └── concept_mapper.py (new)
├── utils/
│   ├── synthesis_generator.py (new)
│   └── output_formatter.py (new)
├── config/
│   ├── keywords.yaml (legitimacy/performance terms)
│   └── concept_taxonomy.yaml (second-order mappings)
└── run_pipeline.py (orchestrator)
```

## Critical Questions for You:

1. **Do you have any labeled examples** from the Excel file that we can use to validate our rules? Even 20-30 examples would help tremendously.

2. **For the "Code spé" column** - do you have the full list of these codes? The analysis mentions ~50 unique codes but doesn't list them all.

3. **For C/S distinction** (Constat vs Stereotype) - this seems highly subjective. Should we skip this for baseline and mark everything as "C"?

4. **Output format** - Do you need exactly the same column names in French, or can we use English equivalents for development?

## Next Steps

Given your timeline constraints, I suggest:

1. **Week 1**: Implement basic tension extraction and theme classification
2. **Week 2**: Add concept mapping and output formatting
3. **Week 3**: Testing and refinement with real data

This approach prioritizes getting a working baseline quickly. We can always iterate and improve accuracy later.

What do you think about these options? Which approaches align best with your timeline and resources?
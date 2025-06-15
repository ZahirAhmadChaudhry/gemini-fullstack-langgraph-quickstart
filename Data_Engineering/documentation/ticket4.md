# French Temporal Marker Detection: Distinguishing Between Present and Future Contexts

The automated detection of temporal markers in French texts requires sophisticated natural language processing techniques to accurately identify expressions that signal different time frames, such as distinguishing between references to 2023 (present/recent past) versus 2050 (future). This report presents a comprehensive analysis of temporal markers in French and methodologies for their detection.

## Typology of French Temporal Markers

### Explicit Date References

Explicit date references provide the most straightforward signals of temporal context and can be reliably detected using pattern matching approaches.

- **Present/Recent Past Markers**: "en 2023", "actuellement", "cette année", "au cours de l'année", "de nos jours"
- **Future Markers**: "en 2050", "d'ici 2050", "d'ici la fin du siècle", "dans les années 2040", "à l'horizon 2050"

These explicit references can be detected using regular expressions that identify year patterns combined with appropriate prepositions[1][4].

### Verb Tense Indicators

French employs distinct verb tenses to distinguish between present and future contexts:

- **Le Présent**: Used for current situations, facts, repeated actions, and notably, for planned future events when accompanied by future time indicators[7].
  - Example: "Je travaille à Paris" (I work in Paris) - present context
  - Example: "Vendredi prochain à 14 heures, je participe à un concours" (Next Friday at 2pm, I'm participating in a competition) - planned future[7]

- **Le Futur Simple**: Indicates actions that will take place in the more distant future, constructed by adding endings (-ai, -as, -a, -ons, -ez, -ont) to the infinitive form of regular verbs[3].
  - Example: "Je parlerai français en 2050" (I will speak French in 2050)

- **Le Futur Proche**: Indicates imminent future actions, formed with the present tense of "aller" + infinitive[3].
  - Example: "Je vais déménager le mois prochain" (I am going to move next month)

### Specialized Temporal Prepositions and Adverbs

French uses specific prepositions and adverbial expressions to mark temporal relations:

- **"depuis"**: Indicates continuity from past to present[5].
  - Example: "J'habite ici depuis 2023" (I have been living here since 2023)

- **"pendant"**: Marks a specific, limited duration in the past[5].
  - Example: "J'ai travaillé là pendant cinq ans" (I worked there for five years)

- **"il y a"**: Refers to a specific moment in the past[5].
  - Example: "Il a fait froid il y a deux jours" (It was cold two days ago)

- **"pour"**: Expresses a planned future duration[5].
  - Example: "Je serai absent pour deux semaines" (I will be absent for two weeks)

- **"dans"**: Indicates a future moment[5].
  - Example: "Les changements seront visibles dans vingt ans" (The changes will be visible in twenty years)

## NLP Detection Techniques

### Regular Expression Patterns

Regular expressions provide an efficient method for identifying explicit temporal markers, especially those containing dates or specific temporal references[1][4].

```python
# Regex patterns for date references
present_date_pattern = r'\ben 20(1|2)\d\b'  # matches years 2010-2029
future_date_pattern = r'\b(en|d\'ici|à l\'horizon) 20[5-9]\d\b'  # matches years 2050-2099
```

### Part-of-Speech Tagging with spaCy

spaCy's French models can be utilized to perform part-of-speech tagging, which is crucial for identifying verb tenses[2][6].

```python
import spacy
nlp = spacy.load("fr_core_news_sm")

# Add Lefff lemmatizer for improved French processing
from spacy_lefff import LefffLemmatizer
lefff = LefffLemmatizer()
nlp.add_pipe(lefff, after='tagger')

def detect_verb_tense(text):
    doc = nlp(text)
    for token in doc:
        if token.pos_ == "VERB":
            # Analyze verb forms to determine tense
            if token.tag_ in ["VBP", "VBZ"]:  # Present tense markers
                print(f"Present tense: {token.text}")
            elif "FUT" in token.morph.get("Tense", []):  # Future tense markers
                print(f"Future tense: {token.text}")
```

### Context Scanning Strategy (CSS)

As described in the research, a context scanning strategy can be employed to identify temporal expressions in French text[1][4]:

1. **Marker Detection**: Identify triggering markers (stand-alone or parsing-triggering)
2. **Context Analysis**: Analyze left and right contexts of these markers
3. **Expression Boundary Detection**: Determine the full extent of the temporal expression

The search results classify markers into several categories:
- **Stand-alone markers** (→): Constitute a temporal expression by themselves
- **Right-extending markers** (→→): Temporal expressions extending only to the right
- **Left-right extending markers** (←→): Expressions extending both left and right[1][4]

### Chart Parsing

Chart parsing techniques can be employed to analyze complex temporal expressions by:

1. Scanning tagged contexts surrounding potential temporal markers
2. Applying recursive rules to determine the boundaries of temporal expressions
3. Integrating semantic constraints to improve detection precision[1][4]

## Detection Rules and Implementation

### Rules for Date Reference Detection

```python
# Pseudocode for date reference detection
def detect_date_references(text):
    # Current/recent past references (2020s)
    present_refs = re.findall(r'\b(en|durant|pendant) (20(1|2)\d)\b', text)
    
    # Future references (2030s and beyond)
    future_refs = re.findall(r'\b(en|d\'ici|à l\'horizon|dans les années) (20[3-9]\d)\b', text)
    
    return present_refs, future_refs
```

### Rules for Verb Tense Analysis

```python
# Pseudocode for verb tense analysis
def analyze_verb_tenses(doc):
    present_contexts = []
    future_contexts = []
    
    for sent in doc.sents:
        has_future_marker = any(token.lemma_ == "aller" and token.pos_ == "AUX" for token in sent)
        has_future_tense = any("FUT" in token.morph.get("Tense", []) for token in sent)
        
        if has_future_marker or has_future_tense:
            future_contexts.append(sent.text)
        else:
            # Check for present tense with future time indicators
            has_future_indicator = any(token.text in ["demain", "prochainement", "bientôt"] for token in sent)
            if has_future_indicator:
                future_contexts.append(sent.text)
            else:
                present_contexts.append(sent.text)
    
    return present_contexts, future_contexts
```

### Rules for Temporal Prepositions

```python
# Pseudocode for temporal preposition analysis
def analyze_temporal_prepositions(doc):
    temporal_markers = {
        "present_continuing": [],  # depuis + duration
        "past_specific": [],       # il y a + duration
        "future_imminent": [],     # dans + short duration
        "future_distant": []       # d'ici + long duration
    }
    
    for sent in doc.sents:
        for token in sent:
            if token.text == "depuis":
                temporal_markers["present_continuing"].append(sent.text)
            elif token.text == "il" and token.i < len(doc) - 2 and doc[token.i+1].text == "y" and doc[token.i+2].text == "a":
                temporal_markers["past_specific"].append(sent.text)
            elif token.text == "dans":
                # Check if followed by short or long duration
                # For simplicity, we're assuming detection of duration would be implemented
                temporal_markers["future_imminent"].append(sent.text)
            elif token.text == "d'ici" or token.text == "d'" and token.i < len(doc) - 1 and doc[token.i+1].text == "ici":
                temporal_markers["future_distant"].append(sent.text)
    
    return temporal_markers
```

## Comprehensive Detection Strategy

A robust approach to detecting temporal contexts in French text would combine:

1. **Initial tokenization** using spaCy's French pipeline[8]
2. **Pattern-based detection** of explicit date references
3. **Morphological analysis** to identify verb tenses
4. **Context scanning** to detect complex temporal expressions
5. **Rule-based classification** of expressions as present (2023) or future (2050) contexts

For sentences with potentially ambiguous temporal references, a confidence score could be calculated based on the presence of multiple consistent markers.

## Detection Challenges and Considerations

Several challenges should be noted when implementing temporal marker detection:

1. **Ambiguity**: Present tense in French can sometimes refer to future events[7]
2. **Implicit references**: Temporal context may be implied rather than explicitly stated
3. **Domain-specific language**: Climate change discussions may use specialized temporal terminology
4. **Anaphoric references**: Pronouns referring to previously mentioned dates require coreference resolution

## Conclusion

Effective detection of temporal markers distinguishing between 2023 and 2050 contexts in French texts requires a multi-faceted approach combining pattern matching, morphological analysis, and contextual understanding. By implementing the detection rules outlined in this report and leveraging NLP libraries like spaCy with French-specific extensions such as Lefff[2], developers can build robust systems capable of accurately classifying temporal contexts in French texts.

By combining explicit date references, verb tense analysis, and specialized temporal marker detection, systems can reliably differentiate between present and future contexts, enabling more precise analysis of time-sensitive content.

Citations:
[1] https://www.atala.org/sites/default/files/actes_taln/AC_0132.pdf
[2] https://github.com/sammous/spacy-lefff
[3] https://www.frenchtoday.com/blog/french-verb-conjugation/2-french-future-tenses-futur-proche-simple/
[4] https://aclanthology.org/2001.jeptalnrecital-long.29.pdf
[5] https://www.startfrenchnow.com/blog/post/maitrisez-les-marqueurs-temporels-en-francais-il-y-a-depuis-pendant-dans-pour
[6] https://spacy.io/usage/linguistic-features
[7] https://francais.lingolia.com/en/grammar/tenses/le-present
[8] https://stackoverflow.com/questions/75134878/python-nltk-and-spacy-dont-get-same-result-when-tokenize-sentence-in-french
[9] https://www.lajavaness.com/post/detection-and-normalization-of-temporal-expressions-in-french-text-part-1-build-a-dataset-1?lang=en
[10] https://aclanthology.org/W01-1314.pdf
[11] https://www.lawlessfrench.com/grammar/verb-tense/
[12] https://dl.acm.org/doi/pdf/10.5555/1698381.1698388
[13] https://acupoffrench.com/french-vocabulary/time-indicators/
[14] https://www.ekino.fr/publications/handson-de-quelques-taches-courantes-en-nlp/
[15] https://www.fromecollege.org/assets/Sixth-Form/Transition-Work-2024/TENSES-SUMMER-WORK-French-Tenses-Booklet-All-Tenses.pdf
[16] https://www.startfrenchnow.com/tr/blog/post/maitrisez-les-marqueurs-temporels-en-francais-il-y-a-depuis-pendant-dans-pour
[17] https://stackoverflow.com/questions/1922097/regular-expression-for-french-characters
[18] https://spacy.io/models/fr
[19] https://www.reddit.com/r/learnfrench/comments/ilwmsq/i_made_a_overview_of_all_french_verb_tenses_i/
[20] https://coralogix.com/blog/regex-101/
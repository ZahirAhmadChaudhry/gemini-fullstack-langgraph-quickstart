# French Discourse Markers Research

## Categories of Discourse Markers

### 1. Sequential/Temporal Markers
- D'abord (First)
- Puis (Then)
- Ensuite (Next)
- Enfin (Finally)
- Premièrement, Deuxièmement, etc. (Firstly, Secondly, etc.)
- En premier lieu, En second lieu (In the first place, In the second place)
- Pour commencer (To begin with)
- Pour finir (To end with)
- Après (After)
- Plus tard (Later)

### 2. Causal/Explanatory Markers
- Car (Because/For)
- Parce que (Because)
- Puisque (Since)
- En effet (Indeed/In fact)
- C'est pourquoi (That's why)
- Donc (Therefore)
- Par conséquent (Consequently)
- Ainsi (Thus)
- De ce fait (As a result)
- Alors (So/Then)

### 3. Contrastive/Opposition Markers
- Mais (But)
- Cependant (However)
- Toutefois (Nevertheless)
- Par contre (On the other hand)
- En revanche (On the other hand)
- Néanmoins (Nevertheless)
- Pourtant (Yet)
- Malgré (Despite)
- Bien que (Although)
- Au contraire (On the contrary)

### 4. Additive/Elaborative Markers
- De plus (Moreover)
- En outre (Furthermore)
- Par ailleurs (Moreover/Besides)
- Également (Also)
- D'ailleurs (Besides/Moreover)
- En fait (In fact)
- C'est-à-dire (That is to say)
- Notamment (Particularly/Notably)
- En d'autres termes (In other words)
- À savoir (Namely)

### 5. Conclusive Markers
- En conclusion (In conclusion)
- Pour conclure (To conclude)
- En résumé (In summary)
- En bref (In brief)
- Finalement (Finally)
- En définitive (Ultimately)
- En fin de compte (In the end)
- Pour terminer (To end)
- En somme (In sum)
- Bref (In short)

### 6. Reformulation/Clarification Markers
- Autrement dit (In other words)
- C'est-à-dire (That is)
- En d'autres termes (In other terms)
- Pour préciser (To be precise)
- Plus précisément (More precisely)
- À vrai dire (To tell the truth)
- En fait (Actually)
- Au fond (Basically)
- En réalité (In reality)
- Si vous voulez (If you will)

## Usage Patterns

1. **Sentence Position:**
   - Most discourse markers appear at the beginning of sentences
   - Some can appear mid-sentence (e.g., "en fait", "d'ailleurs")
   - Conclusive markers typically start new paragraphs

2. **Punctuation:**
   - Often followed by a comma when at the start of a sentence
   - May be preceded by a semicolon when joining independent clauses
   - Can be surrounded by commas in mid-sentence positions

3. **Combinations:**
   - "Tout d'abord... ensuite... enfin"
   - "D'une part... d'autre part"
   - "Non seulement... mais aussi"

## Implementation Considerations

1. **Priority Markers:**
   These markers most strongly indicate thematic transitions and should trigger segmentation:
   - Temporal sequence markers (d'abord, ensuite, enfin)
   - Strong contrast markers (cependant, toutefois, néanmoins)
   - Topic shift markers (par ailleurs, en ce qui concerne)
   - Conclusive markers (en conclusion, pour conclure)

2. **Context-Dependent Markers:**
   These markers may indicate transitions but require additional context:
   - Causal markers (car, parce que, puisque)
   - Additive markers (de plus, également)
   - Reformulation markers (c'est-à-dire, autrement dit)

3. **Pattern Recognition:**
   - Check for marker combinations
   - Consider capitalization and punctuation
   - Account for variations in spelling and spacing

## Testing Strategy

1. **Coverage Testing:**
   - Verify detection of all marker categories
   - Test variations in capitalization and spacing
   - Validate handling of combined markers

2. **Precision Testing:**
   - Check for false positives with similar words
   - Verify correct handling of markers in different sentence positions
   - Test boundary conditions with minimum/maximum segment sizes

3. **Context Testing:**
   - Verify preservation of semantic relationships
   - Test handling of nested discourse structures
   - Validate multi-sentence logical unit preservation

## References

1. Scholarly works on French discourse analysis
2. French academic writing guides
3. Current implementation in preprocessing pipeline
4. Linguistic research on text segmentation
# YouTube Transcript Processing Guide

## Overview

This document explains how the French Transcript Preprocessing Pipeline handles YouTube transcripts, which often pose unique challenges due to their format and lack of proper punctuation.

## Challenges with YouTube Transcripts

YouTube auto-generated transcripts typically present the following challenges:

1. **Missing or Inconsistent Punctuation**: Auto-generated transcripts often lack proper sentence boundaries
2. **Timestamp Interruptions**: Text is frequently interrupted by timestamp markers
3. **Speaker Identification Issues**: Speaker changes may be unmarked or inconsistently marked
4. **Incomplete Sentences**: Sentences may be cut off or fragmented

## Detection Strategy

The pipeline automatically detects YouTube transcripts using these indicators:

1. **Filename Analysis**:
   - Presence of "youtube", "yt", or "transcript" in the filename

2. **Content Pattern Analysis**:
   - YouTube-specific timestamp formats (e.g., `[0:00]`, `0:00 - 0:15`)
   - YouTube-specific header text ("Generated automatically by YouTube")
   - High ratio of short lines with minimal punctuation

## Processing Approach

### 1. Detection Phase
```python
def _is_youtube_transcript(self, file_path: Path, text_content: str) -> bool:
    # Check filename for YouTube indicators
    filename = file_path.name.lower()
    if "youtube" in filename or "yt" in filename or "transcript" in filename:
        return True
        
    # Check text content for YouTube-specific patterns
    youtube_patterns = [
        r'\[\d{1,2}:\d{2}\]',  # [0:00] timestamp format
        r'\d{1,2}:\d{2} - \d{1,2}:\d{2}',  # 0:00 - 0:15 timestamps
        r'YouTube\s*[Tt]ranscript',
        r'Generated automatically by YouTube'
    ]
    
    for pattern in youtube_patterns:
        if re.search(pattern, text_content):
            return True
    
    # Check for lack of punctuation with many short lines
    lines = text_content.split('\n')
    if len(lines) > 20:
        short_lines = sum(1 for line in lines if len(line.strip()) < 50)
        punctuation_ratio = len(re.findall(r'[.!?]', text_content)) / max(1, len(text_content) / 100)
        
        if short_lines > len(lines) * 0.6 and punctuation_ratio < 0.5:
            return True
    
    return False
```

### 2. Specialized Sentence Tokenization

For YouTube transcripts, the pipeline employs the `ImprovedSentenceTokenizer` which:

1. **Detects sentence boundaries** using:
   - Potential sentence starts (discourse markers, temporal markers)
   - Speaker changes
   - Question words at beginning of phrases
   - Capital letters after spaces that follow lowercase letters

2. **Adds implied periods** where they are likely to belong

3. **Segments long unpunctuated passages** into more reasonable sentence units

### 3. Special Processing Steps

1. **Timestamp Removal**: Systematically removes various timestamp formats
2. **Aggressive Tokenization**: More aggressive sentence boundary detection for unpunctuated text
3. **Enhanced Segmentation**: Uses linguistic cues to divide text where punctuation is missing

## Example

Original YouTube transcript:
```
[0:15] donc aujourd'hui je vais vous parler de l'avenir des énergies 
[0:32] nous allons examiner quelques projets innovants
[0:45] les voitures électriques représentent une solution intéressante 
[1:08] mais il y a encore des défis à surmonter
```

After processing:
```
Donc aujourd'hui je vais vous parler de l'avenir des énergies.
Nous allons examiner quelques projets innovants.
Les voitures électriques représentent une solution intéressante.
Mais il y a encore des défis à surmonter.
```

## Testing YouTube Transcript Processing

To test the YouTube transcript processing functionality:

```powershell
cd E:\2_Baseline_NLP_Project\BaseNLP\Data_Engineering\test
python run_ml_test.py
```

This will run the `test_sentence_tokenizer_with_youtube` test which validates that:
1. The tokenizer correctly identifies sentence boundaries in unpunctuated text
2. The segmentation produces a reasonable number of sentences
3. The system correctly handles both YouTube and standard text formats

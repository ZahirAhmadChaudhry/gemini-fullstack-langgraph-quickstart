"""Test utilities and helper functions."""

import random
import codecs
from pathlib import Path
from typing import List, Dict, Any
from config import TEST_DATA_DIR, ENCODINGS, logger

def generate_test_file_with_encoding(content: str, filename: str, encoding: str) -> Path:
    """Generate a test file with specific encoding."""
    filepath = TEST_DATA_DIR / filename
    try:
        with codecs.open(filepath, 'w', encoding=encoding) as f:
            f.write(content)
        logger.info(f"Created test file {filename} with {encoding} encoding")
        return filepath
    except Exception as e:
        logger.error(f"Error creating test file {filename}: {e}")
        raise

def create_french_text_with_markers(
    num_segments: int = 3,
    include_diacritics: bool = True,
    include_discourse_markers: bool = True,
    include_temporal_markers: bool = True
) -> str:
    """Create French text with various markers for testing."""
    
    discourse_markers = [
        "Premièrement", "Ensuite", "Enfin",
        "Par ailleurs", "Cependant", "En conclusion",
        "Car", "Donc", "En effet"
    ]
    
    temporal_markers_2023 = [
        "en 2023", "actuellement", "maintenant",
        "aujourd'hui", "à présent"
    ]
    
    temporal_markers_2050 = [
        "en 2050", "dans le futur", "à l'avenir",
        "d'ici 2050", "dans les années à venir"
    ]
    
    french_sentences = [
        "Les changements climatiques affectent notre environnement.",
        "Nous devons agir pour protéger la planète.",
        "La technologie évolue rapidement.",
        "Les énergies renouvelables sont importantes.",
        "L'innovation est essentielle pour l'avenir.",
        "La société doit s'adapter aux changements.",
        "Les solutions durables sont nécessaires.",
        "L'éducation joue un rôle crucial.",
        "Les défis sont nombreux mais surmontables.",
        "La collaboration internationale est importante."
    ]
    
    text_segments = []
    for i in range(num_segments):
        segment = []
        
        # Add discourse marker
        if include_discourse_markers and random.random() > 0.3:
            segment.append(random.choice(discourse_markers))
        
        # Add 2-4 sentences
        num_sentences = random.randint(2, 4)
        segment.extend(random.sample(french_sentences, num_sentences))
        
        # Add temporal marker
        if include_temporal_markers and random.random() > 0.5:
            if random.random() > 0.5:
                segment.append(f"Comme nous le voyons {random.choice(temporal_markers_2023)}.")
            else:
                segment.append(f"Nous prévoyons que {random.choice(temporal_markers_2050)}.")
        
        text_segments.append(" ".join(segment))
    
    return "\n\n".join(text_segments)

def create_golden_dataset() -> Dict[str, Any]:
    """Create a golden dataset with known characteristics for testing."""
    text = """Premièrement, les changements climatiques affectent notre environnement en 2023. 
La situation actuelle nécessite une action immédiate.

Ensuite, nous devons considérer les solutions technologiques. 
Les énergies renouvelables sont importantes aujourd'hui. 
Les innovations actuelles ouvrent de nouvelles possibilités.

En conclusion, d'ici 2050, nous aurons transformé notre société. 
Les solutions durables seront essentielles dans le futur. 
La collaboration internationale jouera un rôle crucial."""

    expected_segments = [
        {
            "text": ["Premièrement, les changements climatiques affectent notre environnement en 2023.",
                    "La situation actuelle nécessite une action immédiate."],
            "has_discourse_marker": True,
            "discourse_marker_type": "sequential",
            "temporal_markers": {"2023_reference": True, "2050_reference": False}
        },
        {
            "text": ["Ensuite, nous devons considérer les solutions technologiques.",
                    "Les énergies renouvelables sont importantes aujourd'hui.",
                    "Les innovations actuelles ouvrent de nouvelles possibilités."],
            "has_discourse_marker": True,
            "discourse_marker_type": "sequential",
            "temporal_markers": {"2023_reference": True, "2050_reference": False}
        },
        {
            "text": ["En conclusion, d'ici 2050, nous aurons transformé notre société.",
                    "Les solutions durables seront essentielles dans le futur.",
                    "La collaboration internationale jouera un rôle crucial."],
            "has_discourse_marker": True,
            "discourse_marker_type": "conclusive",
            "temporal_markers": {"2023_reference": False, "2050_reference": True}
        }
    ]
    
    return {
        "text": text,
        "expected_segments": expected_segments,
        "num_segments": 3,
        "has_diacritics": True,
        "has_discourse_markers": True,
        "has_temporal_markers": True
    }

def compare_segments(actual_segments: List[Dict], expected_segments: List[Dict]) -> Dict[str, Any]:
    """Compare actual segments with expected segments and return metrics."""
    total_segments = len(expected_segments)
    matched_segments = 0
    matched_markers = 0
    matched_temporal = 0
    
    for actual, expected in zip(actual_segments, expected_segments):
        # Check text content
        if actual["text"] == expected["text"]:
            matched_segments += 1
        
        # Check discourse markers
        if (actual.get("has_discourse_marker") == expected["has_discourse_marker"] and
            actual.get("discourse_marker_type") == expected["discourse_marker_type"]):
            matched_markers += 1
        
        # Check temporal markers
        if actual.get("temporal_markers") == expected["temporal_markers"]:
            matched_temporal += 1
    
    return {
        "segment_accuracy": matched_segments / total_segments,
        "marker_accuracy": matched_markers / total_segments,
        "temporal_accuracy": matched_temporal / total_segments,
        "total_segments": total_segments,
        "matched_segments": matched_segments,
        "matched_markers": matched_markers,
        "matched_temporal": matched_temporal
    }
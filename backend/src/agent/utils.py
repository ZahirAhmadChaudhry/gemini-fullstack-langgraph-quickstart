from typing import Any, Dict, List
import re
from agent.tools_and_schemas import CONCEPT_CODE_MAPPING, THEME_KEYWORDS


def clean_transcript(text: str) -> str:
    """
    Clean transcript text by removing timestamps and artifacts.
    """
    # Remove common transcript artifacts
    text = re.sub(r'\[Music\]|\[Applause\]|\[Laughter\]', '', text)
    # Remove timestamps (various formats)
    text = re.sub(r'\d{2}:\d{2}:\d{2}', '', text)
    text = re.sub(r'\d{1,2}:\d{2}', '', text)
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def detect_period(text: str) -> str:
    """
    Detect temporal markers (2023 or 2050) in the text.
    """
    if '2050' in text:
        return '2050'
    elif '2023' in text:
        return '2023'
    else:
        return ''


def determine_theme(text: str) -> str:
    """
    Determine theme (Légitimité vs Performance) based on keyword analysis.
    """
    text_lower = text.lower()

    legitimacy_count = sum(1 for keyword in THEME_KEYWORDS["Légitimité"]
                          if keyword.lower() in text_lower)
    performance_count = sum(1 for keyword in THEME_KEYWORDS["Performance"]
                           if keyword.lower() in text_lower)

    if legitimacy_count > performance_count:
        return "Légitimité"
    elif performance_count > legitimacy_count:
        return "Performance"
    else:
        return ""  # Unclear or mixed


def assign_code(reformulated_item: str, second_order_concept: str) -> str:
    """
    Assign specific code based on reformulated item and second-order concept.
    """
    if second_order_concept in CONCEPT_CODE_MAPPING:
        concept_mappings = CONCEPT_CODE_MAPPING[second_order_concept]

        # Try exact match first
        if reformulated_item in concept_mappings:
            return concept_mappings[reformulated_item]

        # Try fuzzy matching based on keywords
        reformulated_lower = reformulated_item.lower()
        for tension_pattern, code in concept_mappings.items():
            pattern_words = tension_pattern.lower().split()
            if any(word in reformulated_lower for word in pattern_words):
                return code

    return "Unknown"


def format_csv(results: List[Dict[str, Any]]) -> str:
    """
    Format analysis results as CSV string.
    """
    if not results:
        return ""

    import csv
    import io

    output = io.StringIO()
    fieldnames = [
        "Concepts de 2nd ordre",
        "Items de 1er ordre reformulé",
        "Items de 1er ordre (intitulé d'origine)",
        "Détails",
        "Synthèse",
        "Période",
        "Thème",
        "Code spé",
        "Constat ou stéréotypes (C ou S) (Imaginaire facilitant IFa ou Imaginaire frein IFr)"
    ]

    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

    return output.getvalue()

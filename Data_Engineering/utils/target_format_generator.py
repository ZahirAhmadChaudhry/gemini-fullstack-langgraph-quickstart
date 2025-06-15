#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Target Format Generator for French Transcripts

This module generates the exact target format that matches the human-annotated
reference data (data.json) structure. It creates ML-ready output that enables
downstream ML pipelines to produce equivalent results.

Target format columns:
- "Concepts de 2nd ordre": High-level conceptual categories
- "Items de 1er ordre reformulé": Refined first-order concepts  
- "Items de 1er ordre (intitulé d'origine)": Original first-order labels
- "Détails": Raw transcript segments
- "Période": Temporal context (2050.0)
- "Thème": Thematic categories (Performance, Légitimité)
- "Code spé": Specialized tension codes
"""

import json
import os
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class TargetFormatGenerator:
    """
    Generates output in the exact target format for ML pipeline compatibility.
    
    This formatter creates structured data that enables any competent ML pipeline
    to generate results matching the sophistication of the human-annotated data.json.
    """
    
    def __init__(self, output_dir: str = "target_format_data"):
        """Initialize the target format generator."""
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize classification mappings based on data.json analysis
        self._init_concept_mappings()
        self._init_theme_mappings()
        self._init_tension_codes()
    
    def _init_concept_mappings(self):
        """Initialize second-order concept mappings."""
        self.concept_mappings = {
            "economic": "MODELES SOCIO-ECONOMIQUES",
            "organizational": "MODELES ORGANISATIONNELS", 
            "environmental": "MODELES ENVIRONNEMENTAUX",
            "technological": "MODELES TECHNOLOGIQUES",
            "social": "MODELES SOCIAUX"
        }
    
    def _init_theme_mappings(self):
        """Initialize theme classification mappings."""
        self.theme_indicators = {
            "Performance": [
                "performance", "efficacité", "croissance", "résultats", "indicateurs",
                "mesure", "évaluation", "productivité", "optimisation", "bénéfices",
                "rentabilité", "compétitivité", "innovation"
            ],
            "Légitimité": [
                "légitimité", "éthique", "responsabilité", "durabilité", "social",
                "environnemental", "équitable", "justice", "transparence", "confiance",
                "acceptabilité", "crédibilité", "authenticité"
            ]
        }
    
    def _init_tension_codes(self):
        """Initialize specialized tension codes based on data.json patterns."""
        self.tension_codes = {
            "accumulation_partage": "10.tensions.alloc.travail.richesse.temps",
            "croissance_decroissance": "10.tensions.diff.croiss.dévelpmt",
            "individuel_collectif": "10.tensions.respons.indiv.coll.etatique.NV",
            "court_terme_long_terme": "10.tensions.court.long.terme",
            "local_global": "10.tensions.local.global",
            "ecologie_prix": "10.tensions.écologie.prix.coûts",
            "financier_extra_financier": "10.tensions.financier.extra-financier.NV",
            "hommes_machines": "10.tensions.hommes versus machines.NV",
            "utilite_envie": "10.tensions.utilité.envie.besoin",
            "dependance_environnement": "10.tensions.dependance.environ.ressources.NV"
        }
    
    def generate_target_format(self, ml_ready_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate target format data from ML-ready preprocessed segments.
        
        Args:
            ml_ready_data: Enhanced ML-ready data from MlReadyFormatter
            
        Returns:
            List of dictionaries in target format structure
        """
        target_format_entries = []
        
        for segment in ml_ready_data.get("segments", []):
            # Extract segment data
            text = segment.get("text", "")
            features = segment.get("features", {})
            ml_features = features.get("ml_features", {})
            
            # Generate target format entry
            entry = {
                "Concepts de 2nd ordre": self._determine_second_order_concept(text, features),
                "Items de 1er ordre reformulé": self._generate_first_order_refined(text, features),
                "Items de 1er ordre (intitulé d'origine)": self._generate_first_order_original(text, features),
                "Détails": text,
                "Période": ml_features.get("temporal_period", 2050.0),
                "Thème": self._determine_theme(text, features),
                "Code spé": self._generate_specialized_code(text, features)
            }
            
            target_format_entries.append(entry)
        
        return target_format_entries
    
    def _determine_second_order_concept(self, text: str, features: Dict[str, Any]) -> str:
        """Determine the second-order concept category."""
        conceptual_markers = features.get("conceptual_markers", [])
        
        if "MODELES_SOCIO_ECONOMIQUES" in conceptual_markers:
            return "MODELES SOCIO-ECONOMIQUES"
        elif "MODELES_ORGANISATIONNELS" in conceptual_markers:
            return "MODELES ORGANISATIONNELS"
        elif "MODELES_ENVIRONNEMENTAUX" in conceptual_markers:
            return "MODELES ENVIRONNEMENTAUX"
        else:
            # Default based on content analysis
            text_lower = text.lower()
            if any(term in text_lower for term in ["économique", "économie", "marché", "capital"]):
                return "MODELES SOCIO-ECONOMIQUES"
            else:
                return "MODELES SOCIO-ECONOMIQUES"  # Default fallback
    
    def _generate_first_order_refined(self, text: str, features: Dict[str, Any]) -> str:
        """Generate refined first-order concept label."""
        tension_patterns = features.get("tension_patterns", {})
        
        # Identify primary tension
        if "accumulation_partage" in tension_patterns:
            return "Accumulation / Partage"
        elif "croissance_decroissance" in tension_patterns:
            return "Croissance / Décroissance"
        elif "individuel_collectif" in tension_patterns:
            return "Individuel / Collectif"
        elif "local_global" in tension_patterns:
            return "Local / Global"
        else:
            # Generate based on thematic content
            thematic = features.get("thematic_indicators", {})
            if thematic.get("performance_indicators", 0) > 0:
                return "Efficacité / Impact"
            elif thematic.get("legitimacy_indicators", 0) > 0:
                return "Responsabilité / Profit"
            else:
                return "Présent / Futur"  # Default fallback
    
    def _generate_first_order_original(self, text: str, features: Dict[str, Any]) -> str:
        """Generate original first-order concept label."""
        # This would typically be more sophisticated, analyzing specific terms
        refined = self._generate_first_order_refined(text, features)
        
        # Convert refined to more specific original format
        mapping = {
            "Accumulation / Partage": "accumulation vs partage",
            "Croissance / Décroissance": "croissance / décroissance", 
            "Individuel / Collectif": "besoins individuels / utilité sociale",
            "Local / Global": "délocalisation vs relocalisation",
            "Efficacité / Impact": "optimisation des bénéfices VS impact social",
            "Responsabilité / Profit": "valeur financière / éthique",
            "Présent / Futur": "modèle présent / modèle futur"
        }
        
        return mapping.get(refined, refined.lower())
    
    def _determine_theme(self, text: str, features: Dict[str, Any]) -> str:
        """Determine the thematic category."""
        ml_features = features.get("ml_features", {})
        performance_score = ml_features.get("performance_score", 0)
        legitimacy_score = ml_features.get("legitimacy_score", 0)
        
        if performance_score > legitimacy_score:
            return "Performance"
        elif legitimacy_score > performance_score:
            return "Légitimité"
        else:
            # Fallback analysis
            text_lower = text.lower()
            performance_count = sum(1 for indicator in self.theme_indicators["Performance"] 
                                  if indicator in text_lower)
            legitimacy_count = sum(1 for indicator in self.theme_indicators["Légitimité"] 
                                 if indicator in text_lower)
            
            return "Performance" if performance_count >= legitimacy_count else "Légitimité"
    
    def _generate_specialized_code(self, text: str, features: Dict[str, Any]) -> str:
        """Generate specialized tension code."""
        tension_indicators = features.get("ml_features", {}).get("tension_indicators", [])
        
        # Map detected tensions to specialized codes
        for tension in tension_indicators:
            if tension in self.tension_codes:
                return self.tension_codes[tension]
        
        # Fallback based on theme and content
        theme = self._determine_theme(text, features)
        if theme == "Performance":
            return "10.tensions.diff.croiss.dévelpmt"
        else:
            return "10.tensions.respons.indiv.coll.etatique.NV"
    
    def save_target_format(self, target_data: List[Dict[str, Any]], 
                          source_filename: str) -> str:
        """Save target format data to JSON file."""
        base_name = os.path.splitext(os.path.basename(source_filename))[0]
        output_filename = f"{base_name}_target_format.json"
        output_path = self.output_dir / output_filename
        
        # Wrap in the same structure as data.json
        output_data = {
            "source_file": source_filename,
            "generated_timestamp": datetime.now().isoformat(),
            "format_version": "data_json_compatible",
            "entries": target_data
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved target format data to {output_path}")
        return str(output_path)

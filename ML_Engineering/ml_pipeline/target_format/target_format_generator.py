"""
Target Format Generator for ML Pipeline.

This module generates output in the exact data.json format required by the business,
leveraging enhanced features from the data engineering pipeline.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class TargetFormatGenerator:
    """
    Generates output in the exact data.json format using enhanced preprocessing features.
    
    Maps ML-ready features to the 7-column structure:
    - "Concepts de 2nd ordre"
    - "Items de 1er ordre reformulé" 
    - "Items de 1er ordre (intitulé d'origine)"
    - "Détails"
    - "Période"
    - "Thème"
    - "Code spé"
    """
    
    def __init__(self):
        """Initialize the target format generator."""
        self.tension_mappings = self._initialize_tension_mappings()
        self.concept_mappings = self._initialize_concept_mappings()
        self.specialized_codes = self._initialize_specialized_codes()
        
    def _initialize_tension_mappings(self) -> Dict[str, Dict[str, str]]:
        """Initialize tension pattern mappings based on data.json analysis."""
        return {
            "accumulation_partage": {
                "reformulated": "Accumulation / Partage",
                "original": "accumulation vs partage"
            },
            "croissance_decroissance": {
                "reformulated": "Croissance / Décroissance", 
                "original": "croissance / décroissance"
            },
            "individuel_collectif": {
                "reformulated": "Individuel / Collectif",
                "original": "besoins individuels / utilité sociale"
            },
            "local_global": {
                "reformulated": "Local / Global",
                "original": "délocalisation vs relocalisation"
            },
            "court_terme_long_terme": {
                "reformulated": "Court terme / Long terme",
                "original": "court terme vs long terme"
            }
        }
    
    def _initialize_concept_mappings(self) -> Dict[str, str]:
        """Initialize conceptual marker mappings."""
        return {
            "MODELES_SOCIO_ECONOMIQUES": "MODELES SOCIO-ECONOMIQUES",
            "MODELES_ORGANISATIONNELS": "MODELES ORGANISATIONNELS", 
            "MODELES_ENVIRONNEMENTAUX": "MODELES ENVIRONNEMENTAUX"
        }
    
    def _initialize_specialized_codes(self) -> Dict[str, str]:
        """Initialize specialized code mappings based on data.json patterns."""
        return {
            "accumulation_partage": "10.tensions.alloc.travail.richesse.temps",
            "croissance_decroissance": "10.tensions.diff.croiss.dévelpmt",
            "individuel_collectif": "10.tensions.respons.indiv.coll.etatique.NV",
            "local_global": "10.tensions.local.global",
            "court_terme_long_terme": "10.tensions.court.long.terme",
            "default_performance": "10.tensions.diff.croiss.dévelpmt",
            "default_legitimacy": "10.tensions.respons.indiv.coll.etatique.NV"
        }
    
    def generate_target_format(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate target format output from ML-ready segments.
        
        Args:
            segments: List of segments with enhanced features
            
        Returns:
            Dictionary in exact data.json format
        """
        if not segments:
            logger.warning("No segments provided for target format generation")
            return {"entries": []}
        
        logger.info(f"Generating target format for {len(segments)} segments")
        
        entries = []
        for segment in segments:
            try:
                entry = self._process_segment(segment)
                if entry:
                    entries.append(entry)
            except Exception as e:
                logger.error(f"Error processing segment {segment.get('id', 'unknown')}: {e}")
                continue
        
        result = {
            "entries": entries,
            "metadata": {
                "generated_timestamp": datetime.now().isoformat(),
                "total_segments": len(segments),
                "processed_entries": len(entries),
                "format_version": "data_json_compatible"
            }
        }
        
        logger.info(f"Generated {len(entries)} target format entries")
        return result
    
    def _process_segment(self, segment: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Process a single segment into target format entry.
        
        Args:
            segment: Segment with enhanced features
            
        Returns:
            Target format entry or None if processing fails
        """
        features = segment.get("features", {})
        ml_features = features.get("ml_features", {})
        
        # Extract basic information
        text = segment.get("text", "")
        if not text:
            logger.warning(f"Segment {segment.get('id')} has no text")
            return None
        
        # Determine temporal period
        periode = self._determine_periode(ml_features)
        
        # Determine theme
        theme = self._determine_theme(ml_features)
        
        # Determine primary tension and concepts
        primary_tension = self._determine_primary_tension(features)
        concept_ordre = self._determine_concept_ordre(features, primary_tension)
        
        # Generate first-order items
        items_reformule, items_origine = self._generate_first_order_items(primary_tension)
        
        # Generate specialized code
        code_spe = self._generate_specialized_code(primary_tension, theme)
        
        return {
            "Concepts de 2nd ordre": concept_ordre,
            "Items de 1er ordre reformulé": items_reformule,
            "Items de 1er ordre (intitulé d'origine)": items_origine,
            "Détails": text,
            "Période": periode,
            "Thème": theme,
            "Code spé": code_spe
        }
    
    def _determine_periode(self, ml_features: Dict[str, Any]) -> float:
        """Determine temporal period from ML features."""
        temporal_period = ml_features.get("temporal_period")
        if temporal_period in [2023.0, 2050.0, 2035.0]:
            return temporal_period
        
        # Default based on temporal context
        temporal_context = ml_features.get("temporal_context", "unknown")
        if temporal_context == "2023":
            return 2023.0
        elif temporal_context == "2050":
            return 2050.0
        else:
            return 2035.0  # Default to intermediate period
    
    def _determine_theme(self, ml_features: Dict[str, Any]) -> str:
        """Determine theme from performance and legitimacy scores."""
        performance_score = ml_features.get("performance_score", 0.0)
        legitimacy_score = ml_features.get("legitimacy_score", 0.0)
        
        # Use scores to determine theme
        if performance_score > legitimacy_score:
            return "Performance"
        elif legitimacy_score > performance_score:
            return "Légitimité"
        else:
            # Default to Performance if scores are equal or both zero
            return "Performance"
    
    def _determine_primary_tension(self, features: Dict[str, Any]) -> str:
        """Determine primary tension from tension patterns."""
        tension_patterns = features.get("tension_patterns", {})
        
        if not tension_patterns:
            return "croissance_decroissance"  # Default tension
        
        # Find tension with highest strength
        max_strength = 0
        primary_tension = "croissance_decroissance"
        
        for tension, data in tension_patterns.items():
            strength = data.get("tension_strength", 0)
            if strength > max_strength:
                max_strength = strength
                primary_tension = tension
        
        return primary_tension
    
    def _determine_concept_ordre(self, features: Dict[str, Any], primary_tension: str) -> str:
        """Determine second-order concept from conceptual markers."""
        conceptual_markers = features.get("conceptual_markers", [])
        
        if conceptual_markers:
            # Use first conceptual marker
            marker = conceptual_markers[0]
            return self.concept_mappings.get(marker, "MODELES SOCIO-ECONOMIQUES")
        
        # Default based on tension type
        if primary_tension in ["accumulation_partage", "croissance_decroissance"]:
            return "MODELES SOCIO-ECONOMIQUES"
        elif primary_tension in ["individuel_collectif"]:
            return "MODELES ORGANISATIONNELS"
        else:
            return "MODELES ENVIRONNEMENTAUX"
    
    def _generate_first_order_items(self, tension: str) -> tuple[str, str]:
        """Generate first-order items from tension."""
        mapping = self.tension_mappings.get(tension)
        if mapping:
            return mapping["reformulated"], mapping["original"]
        
        # Default fallback
        return "Présent / Futur", "modèle présent / modèle futur"
    
    def _generate_specialized_code(self, tension: str, theme: str) -> str:
        """Generate specialized code from tension and theme."""
        # Try tension-specific code first
        code = self.specialized_codes.get(tension)
        if code:
            return code
        
        # Fallback to theme-based default
        if theme == "Performance":
            return self.specialized_codes["default_performance"]
        else:
            return self.specialized_codes["default_legitimacy"]

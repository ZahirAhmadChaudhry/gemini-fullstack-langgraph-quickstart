from typing import List, Dict, Any
from pydantic import BaseModel, Field
import json
import os


class SegmentsList(BaseModel):
    segments: List[str] = Field(
        description="A list of text segments containing paradoxes or tensions."
    )


class TensionExtraction(BaseModel):
    original_item: str = Field(
        description="The exact quote from the text that shows the paradox (keep exact words)."
    )
    reformulated_item: str = Field(
        description="The paradox summarized as 'X vs Y' format."
    )


class Categorization(BaseModel):
    concept: str = Field(
        description="The second-order concept this tension belongs to."
    )
    code: str = Field(
        description="The specific code for this tension, or 'Unknown' if not found."
    )


class FullAnalysisResult(BaseModel):
    concepts_2nd_ordre: str = Field(description="Second-order concept")
    items_1er_ordre_reformule: str = Field(description="Reformulated first-order item")
    items_1er_ordre_origine: str = Field(description="Original first-order item")
    details: str = Field(description="Text excerpt details")
    synthese: str = Field(description="One-line synthesis")
    periode: str = Field(description="Time period (2023 or 2050)")
    theme: str = Field(description="Theme (Légitimité or Performance)")
    code_spe: str = Field(description="Specific code")
    imaginaire: str = Field(description="C/S and IFa/IFr classification")


# Domain knowledge mappings
CONCEPT_CODE_MAPPING = {
    "MODELES SOCIO-ECONOMIQUES": {
        "Accumulation / Partage": "10.tensions.alloc.travail.richesse.temps",
        "croissance / décroissance": "10.tensions.diff.croiss.dévelpmt",
        "actionnariat financier / actionnariat responsable": "10.tensions.retombees.positives.VS.absenc.négativ.NV",
        "automatisation / humanisation": "10.tensions.hommes versus machines.NV",
        "Cheap low cost / expensive écologique": "10.tensions.écologie.prix.coûts",
        "contrôle de l'environnement / dépendance à l'environnement": "10.tensions.dependance.environ.ressources.NV",
        "Création d'un besoin / Réponse à un besoin": "10.tensions.utilité.envie.besoin",
    }
}

THEME_KEYWORDS = {
    "Légitimité": [
        "transparence", "équité", "communs", "environnement", "société", "éthique",
        "responsable", "durable", "social", "collectif", "partage", "coopératif"
    ],
    "Performance": [
        "profit", "efficacité", "rentabilité", "compétitivité", "croissance",
        "productivité", "optimisation", "bénéfice", "économique", "financier"
    ]
}

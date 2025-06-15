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


# Domain knowledge mappings - Enhanced with expert-annotated codes
CONCEPT_CODE_MAPPING = {
    "MODELES SOCIO-ECONOMIQUES": {
        "Accumulation / Partage": "10.tensions.alloc.travail.richesse.temps",
        "croissance / décroissance": "10.tensions.diff.croiss.dévelpmt",
        "actionnariat financier / actionnariat responsable": "10.tensions.retombees.positives.VS.absenc.négativ.NV",
        "automatisation / humanisation": "10.tensions.hommes versus machines.NV",
        "Cheap low cost / expensive écologique": "10.tensions.écologie.prix.coûts",
        "contrôle de l'environnement / dépendance à l'environnement": "10.tensions.dependance.environ.ressources.NV",
        "Création d'un besoin / Réponse à un besoin": "10.tensions.utilité.envie.besoin",
        "innovation concurrentielle / besoin réel client": "10.tensions.utilité.envie.besoin",
        "court terme / long terme": "10.tensions.court.long.terme.NV",
        "financier / extra-financier": "10.tensions.financier.extra-financier.NV",
        "société / marché": "10.tensions.société.marché",
        "local / global": "10.tensions.local.global",
        "production / consommation": "10.tensions.prod.conso",
    },
    "GOUVERNANCE ET ORGANISATION": {
        "individuel / collectif": "10.tensions.inviduel vs collectif.NV",
        "responsabilité individuelle / collective": "10.tensions.respons.indiv.coll.etatique.NV",
        "règles / liberté": "10.tensions.règles/liberté.NV",
        "état / intérêts privés": "10.tensions.etat.interets_privés.NV",
        "contrainte légale / volontariat": "10.tensions.contrainte_legale.pression_marché.volontariat.NV",
        "minorité / majorité": "10.tensions.minorité.majorité.NV",
        "autonomie / dépendance": "10.tensions.autonomie.dépendance",
        "évaluation / confiance": "10.tensions.evaluation.confiance.contrôle.NV",
        "capacité d'agir": "10.tensions.capacite.agir.NV",
    },
    "INNOVATION ET TECHNOLOGIE": {
        "usage technologique": "10.tensions.usage.techno.",
        "standardisation / expérimentation": "10.tensions.standardisation.temps.activité.NV",
        "compétences à développer": "10.tensions.compétences.à.faire",
        "discours / pratiques": "10.tensions.discours vs pratiques.NV",
        "motivation intrinsèque / contrôle": "10.tensions.motivation.intrinseque.ctrl.social.NV",
    },
    "RESSOURCES ET ENVIRONNEMENT": {
        "dépendance environnementale": "10.tensions.dependance.environ.ressources.NV",
        "performance entreprise / environnement": "10.tensions.perf.entreprise.envrnmt.nat.",
        "ressources matérielles / travail": "10.tensions.ress.mat.trav.fourn.ress.fi",
        "contraintes / rentabilité": "10.tensions.contraintes_depenses.pertes_rentabilité.affaiblissement.NV",
        "continuité / rupture": "10.tensions.continuite.rupture.NV",
    },
    "PERFORMANCE ET MESURE": {
        "organisation du travail": "10.tensions.déf.place.orga.du.travail",
        "lien / isolement": "10.tensions.lien.isolement",
        "alignement individuel": "10.tensions.alignement.indiv.entre.NV",
        "ressources pour changement": "10.tensions.ressources.capacités.pour.gestion.changement.NV",
        "flexibilité / précarité": "10.tensions.autonomie.flexib.précarité.sante.NV",
        "conflits de priorité": "10.tensions.conflits.de.priorité.NV",
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

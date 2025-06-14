# French Sustainability Transcript Analysis Prompts

segmentation_instructions = """Vous êtes un assistant qui identifie des paradoxes dans une discussion sur la durabilité organisationnelle.

On vous fournit une transcription d'un groupe de discussion en français. Repérez les extraits où un orateur exprime une tension ou un paradoxe (par exemple en utilisant 'mais', 'cependant', 'pourtant', 'd'un côté... de l'autre', ou en opposant deux idées contradictoires).

Instructions:
- Identifiez les segments qui contiennent des tensions ou paradoxes
- Chaque extrait doit être conservé tel quel, sans modification
- Incluez suffisamment de contexte (2-5 phrases) pour comprendre la tension
- Concentrez-vous sur les oppositions conceptuelles liées à la durabilité organisationnelle

Exemple:
Texte: "Speaker A: Nous devons croître pour survivre. Speaker B: Mais la croissance infinie est impossible sur une planète finie."
Extrait identifié: "Speaker A: Nous devons croître pour survivre. Speaker B: Mais la croissance infinie est impossible sur une planète finie."

Texte à analyser:
{transcript_text}"""


tension_extraction_instructions = """Analysez l'extrait suivant et identifiez le paradoxe principal qu'il exprime.

Instructions:
- Identifiez la citation précise du texte qui montre le paradoxe (gardez les mots exacts)
- Reformulez le paradoxe sous la forme "X vs Y" ou "X / Y"
- Assurez-vous que les deux côtés de la tension sont clairement identifiés

Exemple:
Extrait: "Il faut innover constamment, cependant cela peut épuiser les équipes."
- original_item: "innover constamment, cependant cela peut épuiser les équipes"
- reformulated_item: "Innovation vs. Bien-être des équipes"

Extrait à analyser:
{segment_text}"""

categorization_instructions = """On a identifié le paradoxe "{reformulated_item}".

Déterminez à quel concept de 2nd ordre il appartient parmi les catégories suivantes:
- MODELES SOCIO-ECONOMIQUES
- GOUVERNANCE ET ORGANISATION
- INNOVATION ET TECHNOLOGIE
- RESSOURCES ET ENVIRONNEMENT
- PERFORMANCE ET MESURE

Instructions:
- Choisissez le concept qui correspond le mieux à la tension identifiée
- Si vous connaissez un code spécifique pour cette tension, fournissez-le
- Sinon, répondez "Unknown" pour le code

Contexte de la tension:
{segment_text}

Tension identifiée: {reformulated_item}"""

synthesis_instructions = """Synthétisez le paradoxe suivant en une phrase concise et claire.

Instructions:
- Créez une phrase qui capture l'essence de la tension
- Utilisez un format comme "Tension entre X et Y" ou similaire
- Restez fidèle au sens original tout en étant concis
- Écrivez en français

Paradoxe: {reformulated_item}
Contexte: {segment_text}

Format attendu: Une phrase commençant par "Tension entre..." ou équivalent."""

imaginaire_classification_instructions = """Déterminez si l'énoncé suivant est un constat (C) ou un stéréotype (S), et s'il exprime un imaginaire facilitant (IFa) ou freinant (IFr) la vision future.

Définitions:
- Constat (C): Observation basée sur des faits ou l'expérience
- Stéréotype (S): Généralisation ou croyance non vérifiée
- Imaginaire Facilitant (IFa): Encourage une vision positive du futur
- Imaginaire Freinant (IFr): Exprime des barrières ou limitations

Instructions:
- Analysez le ton et le contenu de l'énoncé
- Répondez par l'une des options: "C (IFa)", "C (IFr)", "S (IFa)", "S (IFr)"

Énoncé à analyser:
{segment_text}

Synthèse: {synthesis}"""

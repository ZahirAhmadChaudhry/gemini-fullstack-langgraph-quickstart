# French Sustainability Transcript Analysis Prompts
# Enhanced with expert-annotated examples and cultural discourse patterns

segmentation_instructions = """Vous êtes un expert en analyse de discours français sur la durabilité organisationnelle.

MISSION: Identifiez les segments de transcription contenant des tensions ou paradoxes liés à la durabilité organisationnelle.

CRITÈRES DE SEGMENTATION:
1. **Marqueurs linguistiques explicites**: 'mais', 'cependant', 'pourtant', 'néanmoins', 'par ailleurs', 'd'un côté... de l'autre', 'en même temps'
2. **Tensions implicites**: Oppositions conceptuelles sans marqueurs explicites
3. **Contradictions développées**: Discussions où plusieurs perspectives s'opposent
4. **Paradoxes organisationnels**: Dilemmes spécifiques aux entreprises durables

CONTEXTE REQUIS:
- Incluez 3-4 phrases pour préserver le sens complet
- Conservez les identifiants des locuteurs (1CJ, 2CJ, etc.)
- Maintenez les hésitations et répétitions authentiques du discours oral

EXEMPLES EXPERTS:

**Exemple 1 - Tension Accumulation/Partage:**
"3CJ : À condition de créer de la richesse pour la redistribuer. 1CJ : Oui, on va pas... on va pas redistribuer ce qu'on n'a pas. 3CJ : Parce que si on part du principe qu'il faut redistribuer des revenus, c'est le capital. 2CJ : Et alors... et y'a quand même un problème d'accessibilité, parce que... pour tous, parce que avec l'augmentation du coût des matières premières... 3CJ : Il faut par... c'est l'économie de l'usage. C'est l'économie de... du partage."

**Exemple 2 - Tension Innovation/Utilité:**
"L'innovation, ça peut être dans le bon sens, y'a un besoin, et ça peut être « bah l'autre il a fait ça, donc moi, je vais innover, mais du coup, je vais créer quelque chose qui... qui a pas une plus-value » et comment on se positionne envers les autres."

**Exemple 3 - Tension Croissance/Décroissance:**
"À un moment, tu dois pas te stabiliser pour te développer ? Les entreprises doivent vi... enfin il faut que le mot « décroissance » ne soit plus un gros mot et que les entreprises ne cherchent pas toujours à viser la croissance, mais peut-être à mieux produire, plutôt que produire plus."

VALIDATION:
- Le segment doit concerner la durabilité organisationnelle
- La tension doit être substantielle, pas anecdotique
- Le contexte doit permettre de comprendre l'enjeu

Texte à analyser:
{transcript_text}"""


tension_extraction_instructions = """Analysez l'extrait suivant et identifiez le paradoxe principal qu'il exprime.

MÉTHODE D'ANALYSE:
1. **Identifiez les éléments en opposition** dans le discours
2. **Localisez la citation précise** qui révèle la tension
3. **Reformulez clairement** la tension sous forme "X / Y"

EXEMPLES EXPERTS:

**Exemple 1 - Modèles Socio-Économiques:**
Extrait: "3CJ : À condition de créer de la richesse pour la redistribuer. 1CJ : Oui, on va pas... on va pas redistribuer ce qu'on n'a pas. [...] 3CJ : Il faut par... c'est l'économie de l'usage. C'est l'économie de... du partage."
- original_item: "créer de la richesse pour la redistribuer [...] c'est l'économie de l'usage [...] du partage"
- reformulated_item: "Accumulation / Partage"

**Exemple 2 - Innovation:**
Extrait: "L'innovation, ça peut être dans le bon sens, y'a un besoin, et ça peut être « bah l'autre il a fait ça, donc moi, je vais innover, mais du coup, je vais créer quelque chose qui... qui a pas une plus-value »"
- original_item: "innovation [...] y'a un besoin [...] créer quelque chose qui... qui a pas une plus-value"
- reformulated_item: "innovation concurrentielle / besoin réel client"

**Exemple 3 - Croissance:**
Extrait: "À un moment, tu dois pas te stabiliser pour te développer ? Les entreprises doivent vi... enfin il faut que le mot « décroissance » ne soit plus un gros mot et que les entreprises ne cherchent pas toujours à viser la croissance, mais peut-être à mieux produire, plutôt que produire plus."
- original_item: "viser la croissance [...] mieux produire, plutôt que produire plus"
- reformulated_item: "croissance / décroissance"

INSTRUCTIONS:
- Conservez les mots-clés exacts du discours original
- Formulez la tension de manière équilibrée (X / Y)
- Assurez-vous que les deux pôles sont clairement identifiables

Extrait à analyser:
{segment_text}"""

categorization_instructions = """On a identifié le paradoxe "{reformulated_item}".

CATÉGORIES DE 2ND ORDRE:

**MODELES SOCIO-ECONOMIQUES** - Tensions autour des modèles économiques, de la création/distribution de valeur, des relations marché/société
- Exemples: Accumulation/Partage, croissance/décroissance, actionnariat financier/responsable

**GOUVERNANCE ET ORGANISATION** - Tensions dans les structures de pouvoir, la prise de décision, les modes d'organisation
- Exemples: centralisation/décentralisation, hiérarchie/horizontalité, contrôle/autonomie

**INNOVATION ET TECHNOLOGIE** - Tensions autour du développement technologique, de l'innovation, de l'automatisation
- Exemples: innovation concurrentielle/besoin réel, automatisation/humanisation, standardisation/expérimentation

**RESSOURCES ET ENVIRONNEMENT** - Tensions liées aux ressources naturelles, à l'impact environnemental, à la durabilité
- Exemples: exploitation/préservation, local/global, court terme/long terme

**PERFORMANCE ET MESURE** - Tensions dans les critères d'évaluation, les indicateurs de succès, les objectifs organisationnels
- Exemples: performance financière/globale, efficacité/qualité, quantitatif/qualitatif

CODES SPÉCIFIQUES DISPONIBLES:
- 10.tensions.alloc.travail.richesse.temps
- 10.tensions.diff.croiss.dévelpmt
- 10.tensions.retombees.positives.VS.absenc.négativ.NV
- 10.tensions.hommes versus machines.NV
- 10.tensions.écologie.prix.coûts
- 10.tensions.dependance.environ.ressources.NV
- 10.tensions.utilité.envie.besoin

INSTRUCTIONS:
1. Analysez le contexte et la tension identifiée
2. Choisissez la catégorie la plus appropriée
3. Si la tension correspond à un code spécifique, utilisez-le
4. Sinon, répondez "Unknown" pour le code

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

DÉFINITIONS DÉTAILLÉES:

**Constat (C)**: Observation factuelle basée sur l'expérience directe ou des données vérifiables
- Exemple: "L'automatisation remplace déjà les caissières dans les supermarchés"
- Indicateurs: références à des faits observables, expériences concrètes, données mesurables

**Stéréotype (S)**: Généralisation simplificatrice ou croyance non vérifiée
- Exemple: "Les gens sont naturellement égoïstes et ne changeront jamais leurs habitudes"
- Indicateurs: généralisations absolues, jugements de valeur, suppositions non étayées

**Imaginaire Facilitant (IFa)**: Encourage une vision positive et constructive du futur
- Exemple: "On peut créer des modèles économiques plus équitables"
- Indicateurs: optimisme, solutions proposées, possibilités d'amélioration

**Imaginaire Freinant (IFr)**: Exprime des barrières, limitations ou pessimisme
- Exemple: "Il sera impossible de changer les mentalités consuméristes"
- Indicateurs: obstacles insurmontables, fatalisme, résistance au changement

EXEMPLES D'ANALYSE:

**Exemple 1**: "Les actionnaires devront prendre en compte les critères environnementaux"
→ **C (IFa)** - Constat d'une évolution en cours + vision positive

**Exemple 2**: "Limiter l'enrichissement remet en cause la liberté individuelle"
→ **S (IFr)** - Généralisation + frein au changement

INSTRUCTIONS:
1. Identifiez d'abord si c'est un constat factuel ou un stéréotype
2. Déterminez ensuite l'orientation vers le futur (facilitant/freinant)
3. Répondez par l'une des options: "C (IFa)", "C (IFr)", "S (IFa)", "S (IFr)"

Énoncé à analyser:
{segment_text}

Synthèse: {synthesis}"""

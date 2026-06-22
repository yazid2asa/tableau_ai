# PRD — Product Requirements Document
# Text-to-Viz Agent
### Agent IA de génération automatique de visualisations Tableau Desktop

---

| 📅 Version | 🎯 Scope | 🤖 Modèle LLM | ⚙️ Backend |
|---|---|---|---|
| v1.0 — 2025 | Tableau Desktop — Local | minimax/minimax-m2.5:free | FastAPI — Python |

---

## Table des Matières

1. [Vue d'ensemble du Projet](#1-vue-densemble-du-projet)
2. [Architecture Technique](#2-architecture-technique)
3. [Flux Fonctionnel Détaillé](#3-flux-fonctionnel-détaillé)
4. [Spécifications Fonctionnelles](#4-spécifications-fonctionnelles)
5. [Système d'Observabilité & Qualité](#5-système-dobservabilité--qualité)
6. [Spécifications API Backend](#6-spécifications-api-backend)
7. [Stack Technologique](#7-stack-technologique)
8. [Roadmap & Phases de Développement](#8-roadmap--phases-de-développement)
9. [Contraintes & Risques](#9-contraintes--risques)
10. [Critères de Succès — v1](#10-critères-de-succès--v1)
11. [Hors Scope — v1](#11-hors-scope--v1)

---

## Statut d'Implémentation

| Milestone | Statut | Tests |
|---|---|---|
| **M1 — Core Infrastructure** | ✅ COMPLETE | — |
| **M2 — Conversational Agent** | ✅ COMPLETE | — |
| **M2.5 — Datasource Wiring & Chart Cart** | ✅ COMPLETE | 30/30 passing |
| **M3 — Quality & Observability** | ✅ COMPLETE | 37/37 passing |
| **M4 — BI Enrichment (Filters, Calc Fields, Continuity)** | ✅ COMPLETE | 62/62 passing |
| **M5 — Advanced Viz & Production-Ready** | ⏳ Not started | — |

---

## 1. Vue d'ensemble du Projet

### 1.1 Contexte & Problématique

Tableau Desktop est l'un des outils de Business Intelligence les plus utilisés dans le monde. Cependant, la création de visualisations nécessite une maîtrise technique de l'interface, des types de graphiques, des champs de données et des calculs Tableau. Cette friction empêche les utilisateurs métiers de tirer pleinement parti de leurs données.

Einstein Copilot de Salesforce tente de répondre à ce besoin, mais reste limité à l'écosystème Salesforce/Tableau Cloud, est fermé, coûteux, et ne permet pas une personnalisation fine. Il n'existe pas de solution open/flexible pour Tableau Desktop standalone.

> 💡 **Opportunité** — Créer un agent conversationnel embarqué dans Tableau Desktop qui traduit automatiquement une question en langage naturel en un fichier Tableau Workbook (`.twb`) complet, opérationnel et ouvert directement dans l'application — sans aucune intervention manuelle de l'utilisateur.

### 1.2 Vision Produit

Text-to-Viz est un agent IA full-stack qui permet à tout utilisateur Tableau Desktop — quelle que soit son expertise technique — de créer des visualisations et dashboards professionnels en posant simplement une question en langage naturel.

| Dimension | Situation Actuelle | Avec Text-to-Viz |
|---|---|---|
| Création de viz | Manuelle — drag & drop + config | Automatique via prompt NL |
| Expertise requise | Maîtrise Tableau + types de viz | Aucune — langage naturel |
| Temps moyen | 15–45 min par dashboard | < 30 secondes |
| Itération | Recommencer depuis zéro | Raffinement conversationnel |
| Accès datasource | Configuration manuelle | Extraction auto des métadonnées |

### 1.3 Objectifs v1

- Générer automatiquement des worksheets et dashboards Tableau (`.twb`) à partir d'une question en langage naturel
- Extraire automatiquement les métadonnées de la datasource déjà ouverte dans Tableau Desktop
- Offrir une interface conversationnelle multi-turns pour le raffinement itératif des visualisations
- Ouvrir automatiquement le fichier `.twb` généré dans Tableau Desktop (avec fallback download)
- Intégrer un système de qualité LLM-as-a-Judge + observabilité complète
- Supporter les types de viz basiques et intermédiaires (10+ types)

---

## 2. Architecture Technique

### 2.1 Vue d'Architecture Globale

```
Tableau Extension (UI)
        │
        ▼
FastAPI Backend (localhost)
        │
        ├──► OpenRouter LLM (minimax/minimax-m2.5:free)
        │           │
        │           ▼
        │     TWB Generator (XML .twb)
        │           │
        │           ▼
        │     LLM-as-a-Judge (validation qualité)
        │
        ├──► SQLite (logs, feedback, métriques)
        ├──► LangFuse (traces LLM)
        └──► /monitoring Dashboard (localhost:8000/monitoring)
```

### 2.2 Composants Principaux

#### Composant 1 — Tableau Extension Panel (Frontend)

| Attribut | Description |
|---|---|
| Type | Tableau Desktop Extension (`.trex`) |
| Technologie | HTML5 + JavaScript + `tableau.extensions.1.latest.js` |
| Rôle principal | Interface conversationnelle chat multi-turns + extraction métadonnées |
| Extraction metadata | Noms de champs, types (dimension/mesure), sources de données, hierarchies |
| Communication | HTTP REST vers FastAPI Backend (localhost) |
| Livraison `.twb` | Option A (download) + Option B (ouverture auto via daemon local) |

#### Composant 2 — FastAPI Backend

| Attribut | Description |
|---|---|
| Framework | FastAPI (Python 3.11+) |
| Hébergement | Local (localhost) — v1. Cloud-ready pour versions futures |
| Endpoints clés | `/chat`, `/generate`, `/feedback`, `/health`, `/monitoring/metrics` |
| Responsabilités | Orchestration LLM, génération `.twb`, validation Judge, logging observabilité |
| Format échange | JSON REST + Server-Sent Events (SSE) pour streaming réponses chat |

#### Composant 3 — OpenRouter LLM Layer

| Attribut | Description |
|---|---|
| Provider | OpenRouter (openrouter.ai) |
| Modèle v1 | `minimax/minimax-m2.5:free` |
| Extensibilité | Swap de modèle via changement de la variable `MODEL_ID` uniquement |
| Utilisation | NL → Viz Intent Extraction + TWB XML Generation |
| LLM-as-a-Judge | Appel secondaire au LLM pour valider la qualité du `.twb` généré |

#### Composant 4 — TWB Generator

| Attribut | Description |
|---|---|
| Format output | Fichier XML `.twb` (Tableau Workbook) |
| Approche | LLM génère la structure JSON de la viz → **twilize** `TWBEditor` → XML `.twb` validé XSD |
| Datasource wiring | `_apply_data_file()` — rewire la connexion vers le fichier local de l'utilisateur (Excel/CSV) via manipulation XML directe (lxml) |
| Multi-sheet | `generate_multi_sheet_twb()` — un seul `.twb` avec N worksheets pour le cart complet |
| Validation | XSD validation intégrée à twilize (`validate=True`) avant livraison |
| Stockage | Dossier `output/` local, watchée par le daemon d'ouverture automatique |

#### Composant 5 — Local Daemon (Auto-Open)

| Attribut | Description |
|---|---|
| Type | Service Python léger (`daemon.py`) — démarrage manuel `python daemon.py` |
| Mécanisme | `watchdog` watch sur `output/` — événements `on_created` + `on_modified` → `tableau://open?src=` protocol handler |
| Debounce | Timestamp-based (`_MIN_INTERVAL = 3.0s`) — évite les doubles ouvertures sur save in-place |
| Health endpoint | `http://localhost:8765/status` (CORS open) — sondé par l'Extension pour afficher le badge daemon |
| Fallback | Si daemon non détecté → bouton "Download" dans l'Extension |

#### Composant 6 — Monitoring Dashboard

| Attribut | Description |
|---|---|
| URL | `http://localhost:8000/monitoring` |
| Stack | FastAPI + Jinja2 templates ou React SPA embarquée |
| Métriques affichées | Latence LLM, token usage, scores Judge, feedback 👍/👎, historique prompts |
| Stockage métriques | SQLite local (v1) — PostgreSQL pour versions futures |

---

## 3. Flux Fonctionnel Détaillé

### 3.1 Flux Principal — Génération d'une Visualisation

1. L'utilisateur ouvre le panneau Extension Text-to-Viz dans Tableau Desktop
2. L'Extension extrait automatiquement les métadonnées de la datasource active via `tableau.extensions.1.latest.js` (champs, types, mesures, dimensions)
3. L'utilisateur tape sa question en langage naturel dans le chat (ex: *"Montre-moi les ventes par région en bar chart pour 2024"*)
4. L'Extension envoie au Backend FastAPI : `{question, metadata, conversation_history}`
5. Le Backend construit un prompt enrichi avec les métadonnées et l'historique conversationnel
6. Appel OpenRouter → `minimax/minimax-m2.5:free` → extraction de l'intent viz (type, champs X/Y, filtres, couleurs, titre)
7. Le TWB Generator transforme l'intent en fichier `.twb` XML valide
8. Le LLM-as-a-Judge valide la qualité du `.twb` (cohérence champs/types, pertinence viz, structure XML)
9. Si score Judge ≥ seuil → livraison. Sinon → itération de correction automatique (max 2 retries)
10. Le `.twb` est sauvegardé dans le dossier output + aperçu JSON renvoyé à l'Extension
11. **Option B** : Le daemon local détecte le nouveau `.twb` et l'ouvre dans Tableau Desktop
12. **Option A (fallback)** : Bouton "Télécharger & Ouvrir" affiché dans le panneau Extension
13. L'utilisateur note la viz 👍/👎 → feedback loggé pour amélioration continue *(boutons Extension — M4)*

### 3.2 Flux Multi-Turn — Raffinement Itératif (M4: Conversational Continuity)

> 💬 **Exemple de conversation**
>
> **Tour 1 :** "Montre-moi les ventes par région" → `action: new` → Bar chart généré
> **Tour 2 :** "Maintenant filtre uniquement sur 2024" → `action: modify` → Filtre ajouté au chart existant
> **Tour 3 :** "Change en line chart" → `action: modify` → Type changé, champs préservés
> **Tour 4 :** "Montre-moi le taux de marge par catégorie" → `action: new` → Champ calculé auto-généré + nouveau chart
> **Tour 5 :** "Show West" → `action: clarify` → "Voulez-vous filtrer sur West ou créer un nouveau graphique?"

Chaque tour maintient le contexte via `SessionState` (historique des turns avec `VizIntent` résolu + chemin `.twb`). Le LLM reçoit le `previous_intent` et classifie l'action (new/modify/clarify). Le backend gère un `session_id` par conversation avec stockage en mémoire (v1).

---

## 4. Spécifications Fonctionnelles

### 4.1 Types de Visualisations Supportées — v1

#### Basiques (priorité haute)

| Type de Viz | Cas d'usage typique | Complexité Génération |
|---|---|---|
| Bar Chart (vertical/horizontal) | Comparaison catégories | Faible |
| Line Chart | Tendance temporelle | Faible |
| Pie / Donut Chart | Part de marché / répartition | Faible |
| Scatter Plot | Corrélation entre mesures | Moyenne |
| Area Chart | Volume cumulé dans le temps | Faible |

#### Intermédiaires (priorité moyenne)

| Type de Viz | Cas d'usage typique | Complexité Génération |
|---|---|---|
| Heatmap | Matrice de densité / performance | Moyenne |
| Treemap | Hiérarchie proportionnelle | Moyenne |
| Bullet Chart | KPI vs objectif | Haute |
| Dual-Axis Chart | Comparaison 2 mesures différentes | Haute |
| Highlight Table | Tableau colorisé par valeur | Moyenne |
| Dashboard multi-sheets | Vue consolidée avec filtres croisés | Haute |

### 4.2 Interface Conversationnelle — Spécifications

- Zone de chat scrollable avec historique de la conversation complète
- Champ de saisie texte avec support multilingue (FR/EN minimum)
- Indicateur de chargement animé pendant la génération (SSE streaming)
- Aperçu JSON de la viz générée (type, champs, titre) avant ouverture du `.twb`
- Boutons d'action : "Ouvrir dans Tableau" / "Télécharger"
- Bouton "Nouvelle conversation" pour reset du contexte
- Affichage des métadonnées détectées (datasource active + nb champs)
- Gestion des erreurs avec messages clairs et suggestions de reformulation

### 4.3 Système de Filtres (M4)

| Type de Filtre | Opérateur | Exemple |
|---|---|---|
| Valeur exacte | `eq` | "uniquement la région West" |
| Multi-valeurs | `in` | "catégories Furniture et Technology" |
| Numérique (seuil) | `gt`, `gte`, `lt`, `lte` | "ventes supérieures à 10000" |
| Numérique (plage) | `between` | "profit entre 0 et 5000" |
| Date (année/trimestre/mois) | `year`, `quarter`, `month` | "en 2024", "Q3" |
| Date relative | `last_n_days`, `last_n_months` | "les 30 derniers jours" |
| Top/Bottom N | `top_n`, `bottom_n` | "top 10 clients" |
| Exclusion nulls | `not_null` | "sans valeurs nulles" |
| Sémantique implicite | Inféré par le LLM | "rentable" → Profit > 0 |

### 4.4 Champs Calculés Automatiques (M4)

| Question Type | Champ Calculé | Formule Tableau |
|---|---|---|
| Taux de marge | Taux de Marge | `SUM([Profit])/SUM([Sales])` |
| Panier moyen | Panier Moyen | `SUM([Sales])/COUNTD([Order_ID])` |
| Profit par client | Profit par Client | `SUM([Profit])/COUNTD([Customer])` |
| Part du total | Part du Total | `SUM([Sales])/TOTAL(SUM([Sales]))` |
| Classement | Classement | `RANK(SUM([Sales]))` |

### 4.5 Continuité Conversationnelle (M4)

| Intent Class | Signal | Action Agent |
|---|---|---|
| **Nouveau chart** | Question sans contexte précédent | `action: new` → Nouveau `.twb` |
| **Modifier existant** | "ajoute un filtre", "change en line", "trie par" | `action: modify` → Merge VizIntent |
| **Créer depuis existant** | "même chose mais pour 2023" | `action: new` → Copie + modification |
| **Clarifier** | Ambigu — "show West" | `action: clarify` → Question de clarification |

### 4.6 Gestion Multi-Sources de Données

| Scénario | Comportement | Statut |
|---|---|---|
| **Option C (défaut)** | Extraction auto métadonnées de la datasource active dans Tableau Desktop via `tableau.extensions` API | ✅ Implémenté |
| **Option D — CSV/Excel local** | Chemin de fichier saisi dans l'Extension → `data_file_path` envoyé au backend → `_apply_data_file()` rewire la connexion dans le `.twb` | ✅ Implémenté |
| **Option D — Base de données** | Paramètres de connexion saisis dans l'Extension → `.twb` avec Live Connection | ⏳ M4 |
| **Datasource non détectée** | Message d'erreur clair + guide pour ouvrir une datasource dans Tableau Desktop | ✅ Implémenté |

---

## 5. Système d'Observabilité & Qualité

### 5.1 Architecture d'Observabilité — 3 couches

#### Couche 1 — Logging LLM (Minimal Structured Logging)

- Log de chaque requête : prompt, modèle, latence, `token_usage` (input/output), timestamp
- Log du `.twb` généré : type de viz, champs utilisés, taille fichier XML
- Format : JSON structuré → fichier local `logs/llm_traces.jsonl`
- Rotation automatique des logs (max 100MB, 30 jours)

#### Couche 2 — Observabilité LLM Complète (LangFuse / Arize Phoenix)

- Intégration **LangFuse** (open-source, self-hostable) pour traces complètes
- Spans par étape : `llm_call` → `judge_validation`
- Token usage et coût par trace (même si modèle gratuit en v1)
- Scoring automatique via LLM-as-a-Judge rattaché à chaque trace
- Dashboard LangFuse accessible sur `cloud.langfuse.com`

#### Couche 3 — Feedback Utilisateur

- Boutons 👍/👎 dans l'Extension après chaque génération *(M4 — endpoint `POST /feedback` implémenté, boutons UI à venir)*
- Stockage : table `feedback` en SQLite avec `trace_id`, `score`, `commentaire`, `timestamp`
- Utilisé pour analyse qualité et amélioration des prompts système

### 5.2 LLM-as-a-Judge — Système de Validation Qualité

> **Principe** — Après chaque génération de `.twb`, un second appel LLM évalue la qualité de la visualisation selon 4 critères. Si le score global est insuffisant, une itération de correction automatique est déclenchée (max 2 retries avant livraison avec avertissement).

| Critère Judge | Description | Poids (M4) |
|---|---|---|
| Pertinence viz/question | Le type de viz est approprié à la question posée | **45%** |
| Cohérence champs/types | Les champs utilisés correspondent aux types Tableau (dim/mesure) | **30%** |
| Complétude | Filtres, champs calculés, titre, tri correctement configurés | **15%** |
| Validité XML `.twb` | Vérification résiduelle (twilize validate=True couvre le XSD) | **10%** |

| Paramètre | Valeur |
|---|---|
| Score minimum pour livraison | ≥ 0.75 / 1.0 |
| Retries automatiques max | 2 itérations |
| Comportement si score < seuil après retries | Livraison avec warning + badge "Qualité partielle" dans l'Extension |
| Modèle Judge | Même modèle que génération (`minimax/minimax-m2.5:free`) — v1 |

### 5.3 Dashboard de Monitoring — `localhost/monitoring`

Interface web accessible à `http://localhost:8000/monitoring` affichant en temps réel :

- **Métriques globales** : nb total de générations, taux succès Judge, latence moyenne P50/P95
- **Graphique temporel** : générations par heure/jour
- **Distribution** des types de viz générés
- **Scores Judge** : histogramme + évolution dans le temps
- **Feedback utilisateur** : ratio 👍/👎 + commentaires récents
- **Logs en temps réel** : dernières traces LLM avec détail prompt/réponse
- **Alertes** : latence > 10s, score Judge < 0.5 consécutifs, erreurs OpenRouter

---

## 6. Spécifications API Backend

### 6.1 Endpoints FastAPI

| Endpoint | Méthode + Body | Description |
|---|---|---|
| `POST /chat` | `{question, session_id, metadata, data_file_path, workbook_name}` | Point d'entrée principal — retourne intent + `.twb` généré, appende au cart |
| `POST /chat/stream` | SSE stream | Streaming temps réel (Server-Sent Events) — events: `status` → `intent` → `result` → `done` |
| `GET /download/{filename}` | — | Télécharge un fichier `.twb` généré (path traversal protégé) |
| `GET /session/{session_id}/charts` | — | Retourne le cart accumulé pour une session (liste de viz intents) |
| `POST /download/{session_id}` | `{metadata, data_file_path, charts?}` | Génère un `.twb` multi-sheets depuis le cart (ou liste client) |
| `POST /session/reset` | `{session_id}` | Reset historique conversationnel + cart pour une session |
| `POST /feedback` | `{trace_id, score, comment}` | Enregistre le feedback utilisateur 👍/👎 |
| `GET /health` | — | Health check du service + status OpenRouter |
| `GET /monitoring` | — | Dashboard HTML de monitoring (Jinja2) — *M3* |
| `GET /monitoring/metrics` | — | API JSON des métriques — *M3* |

### 6.2 Schémas de Données Principaux

#### Request — `POST /chat`

```json
{
  "session_id": "uuid-v4",
  "question": "Montre les ventes par région en 2024",
  "metadata": {
    "datasource_name": "SalesData",
    "fields": [
      {"name": "Region", "type": "string", "role": "dimension"},
      {"name": "Sales", "type": "float", "role": "measure"},
      {"name": "Order Date", "type": "date", "role": "dimension"}
    ]
  },
  "conversation_history": [],
  "data_file_path": "C:/Users/you/data/sales.xlsx"
}
```

#### Response — `POST /chat`

```json
{
  "trace_id": "uuid-v4",
  "session_id": "uuid-v4",
  "viz_intent": {
    "viz_type": "bar_chart",
    "title": "Ventes par Région 2024",
    "x_field": "Region",
    "y_field": "Sales",
    "color_field": null,
    "filters": [{"field": "Order Date", "op": "year", "value": 2024}],
    "calculated_fields": [],
    "clarification_needed": null,
    "sort": "descending",
    "aggregation": "SUM",
    "color_scheme": "tableau10",
    "action": "new"
  },
  "twb_filename": "abc12345_bar_chart.twb",
  "twb_download_url": "/download/abc12345_bar_chart.twb",
  "message": "Generated Bar Chart — Ventes par Région 2024",
  "mode": "new_workbook",
  "judge_score": 0.92,
  "judge_feedback": "Viz pertinente et structure XML valide",
  "warning": null,
  "clarification_needed": null
}
```

---

## 7. Stack Technologique

| Composant | Technologie | Justification |
|---|---|---|
| Backend API | FastAPI (Python 3.11+) | Performance async, typage Pydantic, écosystème IA |
| LLM Provider | OpenRouter API | Modèle-agnostic, swap facile, accès modèles gratuits |
| Modèle LLM v1 | `minimax/minimax-m2.5:free` | Gratuit pour tests, remplaçable via config |
| Extension UI | HTML5 + Vanilla JS + `tableau.extensions.1.latest.js` | Compatibilité native Tableau Desktop |
| TWB Generation | **twilize** (XSD-validated `.twb` generation via `TWBEditor`) | Validation XSD intégrée, API haut-niveau pour worksheets et charts |
| Observabilité LLM | LangFuse (self-hosted local) | Open-source, traces complètes, dashboard inclus |
| Stockage local | SQLite (via SQLAlchemy) | Zéro dépendance externe pour v1 locale |
| Monitoring Dashboard | FastAPI + Jinja2 / React SPA | Intégré au backend, pas de service supplémentaire |
| Local Daemon | Python (`watchdog` library) | Cross-platform, watch dossier + open Tableau |
| Tests | Pytest + HTTPX | Tests unitaires endpoints + génération TWB |

---

## 8. Roadmap & Phases de Développement

### Phase 1 — Fondations *(Semaines 1–3)*

> **Objectif** : Infrastructure de base fonctionnelle — Extension Tableau connectée au Backend, génération de `.twb` simple pour Bar Chart et Line Chart, ouverture manuelle du fichier.

- Setup projet FastAPI + structure dossiers + SQLite
- Création Extension Tableau (`.trex`) avec extraction métadonnées basique
- Intégration OpenRouter avec `minimax/minimax-m2.5:free` — premier appel fonctionnel
- Template TWB pour Bar Chart et Line Chart
- Flux end-to-end : question → `.twb` → download manuel

### Phase 2 — Agent Conversationnel *(Semaines 4–6)*

> **Objectif** : Interface chat multi-turns complète, support de 8+ types de viz, daemon d'ouverture automatique.

- Interface chat conversationnelle dans l'Extension (historique multi-turns)
- Session management côté Backend
- Extension des templates TWB : Pie, Scatter, Area, Heatmap, Treemap
- Local Daemon (`watchdog`) pour ouverture automatique dans Tableau Desktop
- Gestion des erreurs et messages utilisateur

### Phase 3 — Qualité & Observabilité *(Semaines 7–9)*

> **Objectif** : LLM-as-a-Judge opérationnel, intégration LangFuse, dashboard monitoring, système feedback.

- Implémentation LLM-as-a-Judge avec les 4 critères de scoring
- Intégration LangFuse local pour traces complètes
- Dashboard monitoring sur `/monitoring` (métriques temps réel)
- Système feedback 👍/👎 dans l'Extension *(endpoint backend livré, boutons UI reportés en M4)*
- Logging structuré JSONL + rotation des logs

### Phase 4 — BI Enrichment *(Semaines 10–12)* ✅ COMPLETE

> **Objectif** : Système de filtres complet, champs calculés automatiques, continuité conversationnelle, validation de champs, repondération du Judge.

- Système de filtres : 12 types d'opérateurs (eq, in, gt/gte/lt/lte, between, year/quarter/month, top_n/bottom_n, not_null)
- Champs calculés automatiques : détection et génération de formules Tableau (taux de marge, panier moyen, etc.)
- Continuité conversationnelle : classification d'action (new/modify/clarify), fusion d'intents, historique par session
- Validation de champs : vérification contre les métadonnées, suggestions de correspondance proche
- Flux de clarification : questions de désambiguïsation sans génération de .twb
- Repondération Judge : viz_relevance 45%, completeness 15%, xml_validity 10%
- 62/62 tests passants

### Phase 5 — Advanced Viz & Production-Ready *(Next)*

> **Objectif** : Bullet Chart, Highlight Table, Dashboards multi-sheets avec filtres croisés, connexion Live DB, documentation.

- Templates TWB : Bullet Chart, Highlight Table
- Support Dashboard multi-sheets avec filtres croisés
- Connexion Live DB (paramètres saisis dans l'Extension)
- Boutons feedback 👍/👎 dans l'Extension
- Documentation technique + guide d'installation

---

## 9. Contraintes & Risques

### 9.1 Contraintes Techniques

| Contrainte | Mitigation |
|---|---|
| Tableau Extensions API : lecture seule (pas d'écriture native) | Approche `.twb` generation + daemon contourne cette limite |
| Format `.twb` est non-documenté officiellement | Reverse engineering + communauté Tableau + tests extensifs |
| `minimax/minimax-m2.5:free` : qualité variable / rate limits | LLM-as-a-Judge + retry logic + swap facile de modèle |
| Compatibilité versions Tableau Desktop | Cibler Tableau Desktop 2021.x+ (format `.twb` stable) |
| Daemon local : permissions OS (Windows/macOS) | Script d'installation avec guide permissions + fallback download |

### 9.2 Risques Produit

| Risque | Probabilité / Impact / Mitigation |
|---|---|
| LLM génère un `.twb` invalide (XML malformé) | Haute / Haute → Validation XML systématique + LLM-as-a-Judge |
| Métadonnées incomplètes depuis l'Extension | Moyenne / Haute → Enrichissement metadata + fallback prompt générique |
| Latence LLM > 10s (expérience dégradée) | Moyenne / Moyenne → SSE streaming + indicateur de progression |
| Modèle gratuit retiré d'OpenRouter | Faible / Haute → Architecture swap-ready (1 variable de config) |

---

## 10. Critères de Succès — v1 (M4 Updated)

| KPI | Cible v1 | Cible M4 | Mesure |
|---|---|---|---|
| Taux de génération `.twb` valide | ≥ 85% | ≥ 90% | Validation XML automatique |
| Score LLM-as-a-Judge moyen | ≥ 0.80 | ≥ 0.82 | Dashboard monitoring |
| Latence bout-en-bout | < 8 secondes | < 10s | P95 sur logs LangFuse |
| Taux feedback positif 👍 | ≥ 70% | ≥ 75% | Table feedback SQLite |
| Types de viz supportés | 10 types | 11 types | Tests d'intégration |
| Taux d'ouverture auto (daemon) | ≥ 90% | ≥ 90% | Logs daemon local |
| Précision des filtres | N/A | ≥ 85% | Filtres correctement appliqués |
| Précision champs calculés | N/A | ≥ 80% | Formule générée correctement |
| Précision modify vs new | N/A | ≥ 90% | Action correcte par turn |
| Taux de clarification | N/A | ≤ 15% | Agent résout la plupart sans demander |

---

## 11. Hors Scope — v1

> ⚠️ Les éléments suivants ne sont **pas inclus** dans la v1 :

- Authentification / gestion multi-utilisateurs
- Hébergement cloud / déploiement distant
- Tableau Server / Tableau Cloud
- Tableau Prep intégration
- Fine-tuning ou entraînement de modèle personnalisé
- Calculs LOD avancés (Level of Detail)
- Dashboard Tableau pour visualiser les métriques de l'agent
- Intégration Slack / email / notifications

---

*Text-to-Viz Agent — PRD v1.0 — Document confidentiel — Usage interne*

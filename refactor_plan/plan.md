# Technical Plan for AI Agent Refactoring

## Problem & Data Overview

French-language group discussion transcripts (\~300 pages) on organizational sustainability are manually annotated into a structured worksheet with **12 fields**. These fields capture a hierarchy of insights, from broad **Second-Order Concepts** down to specific **Original First-Order Items**, with contextual **Details** and a one-line **Synthesis** summary. Each row in the worksheet represents a paradox or tension identified in the conversations, including metadata like the discussion **Group Code**, **Time Period** (2023 vs 2050), thematic category (**Legitimit�** vs **Performance**), and evaluative tags (**Constat/S** for observation/stereotype and **IFa/IFr** indicating if it facilitates or hinders future vision). The goal is to **automate the extraction and categorization** of these paradoxes/tensions from raw French transcripts, replicating expert analysis. Key challenges include detecting contrastive language (e.g. *"mais," "cependant"*) that signals a paradox, grouping tensions under correct conceptual categories, and summarizing insights, all while preserving context (210 line excerpts). The system must support **collaborative refinement**, allowing researchers to review and edit outputs via a web GUI, and provide visualization and interpretability tools (e.g. to compare group outputs or highlight text spans). It should run locally in Docker and leverage **agentic AI workflows** to orchestrate the NLP tasks with memory and feedback loops for continuous improvement.

## Existing Repository Architecture (Gemini LangGraph Quickstart)

The starting codebase is a Google Gemini LangGraph full-stack quickstart, featuring a React frontend and a LangGraph-based Python backend. The original agent is a *research assistant* that takes a user query and performs iterative web searches to produce a cited answer. Key components of the repository include:

* **Frontend (React + Vite)**  The `frontend/` directory contains a React app (TypeScript). Notable files:

  * `frontend/src/App.tsx`: Main app component configuring the UI and API endpoint. It likely sets `apiUrl` for backend requests (points to `http://localhost:2024` in dev or `http://localhost:8123` in Docker). It may also define routing (the UI is served under `/app` path).
  * `frontend/src/components/InputForm.tsx`: A form for user input (originally a text field for questions).
  * `frontend/src/components/ActivityTimeline.tsx`: UI component to display the agents step-by-step actions or messages (e.g. timeline of search queries, reflections, final answer). This was used to visualize the agents internal reasoning and results in the quickstart.
  * Other config files: `vite.config.ts`, `index.html`, etc., to bundle the app.

* **Backend (FastAPI + LangGraph)**  The `backend/` directory is a Python package (installed via `pip install .`) containing the agent logic and a FastAPI app:

  * `backend/src/agent/app.py`: Defines the FastAPI application. It mounts the compiled React build as static files under `/app` so the frontend is served by the same server. (No additional API routes are defined here, as LangGraph likely handles the agent invocation via its own endpoint).
  * `backend/src/agent/graph.py`: **Core of the agent logic**  defines the LangGraph `graph` object that orchestrates agent nodes and state. The quickstarts graph is a sequence of nodes for query generation, web search, reflection, and answer finalization:

    * *Nodes:* `generate_query` (LLM generates search queries), `web_research` (performs a Google search via Gemini API), `reflection` (LLM analyzes gaps in gathered info), and `finalize_answer` (LLM synthesizes the final answer with citations). Each node returns an updated state.
    * *Graph Edges:* The graph is built with `StateGraph` by adding nodes and connecting them: START � `generate_query`, then a conditional edge spawning parallel `web_research` nodes for each query, then `web_research` � `reflection`, a conditional edge from `reflection` either looping back to `web_research` (with follow-up queries) or proceeding to `finalize_answer`, and finally `finalize_answer` � END. This implements an **iterative loop** where the agent searches and reflects until info is sufficient or a max loop count is reached.
    * *State & Data Flow:* The agents shared state (defined in `state.py`) includes fields like `messages` (conversation history), `query_list`, `web_research_result` (accumulated search summaries), `sources_gathered` (citation links), and reflection outputs (`is_sufficient`, `knowledge_gap`, `follow_up_queries`). LangGraph automatically merges parallel node results (e.g. multiple `web_research` outputs are combined into lists) which the reflection node then evaluates.
  * `backend/src/agent/state.py`: Defines Pydantic TypedDicts or dataclasses representing the state schema for different stages. For example, **OverallState** (global state carrying user query and aggregated data), **QueryGenerationState**, **WebSearchState**, **ReflectionState**, etc., each with the fields that relevant nodes read/write.
  * `backend/src/agent/prompts.py`: Houses prompt templates and prompt-construction functions for LLM calls. In the quickstart, it provides formatted instructions such as:

    * `query_writer_instructions`  how to prompt the LLM (Gemini) to produce search queries based on the user question.
    * `web_searcher_instructions`  template for prompting Gemini to perform a web search on a given query (likely instructing use of the search tool).
    * `reflection_instructions`  prompt to analyze gathered info and output JSON (with keys like `is_sufficient` and `follow_up_queries`).
    * `answer_instructions`  prompt to synthesize the final answer with citations.
    * Utility functions (like `get_current_date`) are included to inject context into prompts.
  * `backend/src/agent/tools_and_schemas.py`: Defines data schemas and possibly configures external tools. For example, it contains Pydantic models for structured outputs such as `SearchQueryList` (for LLM to output a list of queries) and `Reflection` (for LLMs reflection output with fields `is_sufficient`, `knowledge_gap`, etc.). It may also set up tool use with LangChain, but in this quickstart the Google Search API was invoked via Geminis native tool integration (no custom tool class defined here, the `tools` param is passed directly in the prompt API call).
  * `backend/src/agent/configuration.py`: Defines configurable parameters (possibly via a `Configuration` Pydantic model). The quickstart likely uses this to centralize things like which Gemini model IDs to use for query generation vs reasoning (e.g., Gemini Flash models), number of initial queries, and max research loops. These are loaded from `RunnableConfig` at runtime, possibly using environment variables or defaults.
  * `backend/src/agent/utils.py`: Helper functions for the agent. In the current code, this includes functions to post-process Gemini outputs: e.g. `resolve_urls()` to shorten URLs from search results, `get_citations()` to format source metadata, `insert_citation_markers()` to annotate answer text with citation indices, and `get_research_topic()` to extract the users core question topic for prompting.
  * **Environment & Dependencies**: The backend relies on `langchain_core`/`langchain_google_genai` libraries for LLM integration with Gemini, and on LangGraph (LangChains graph framework) for agent orchestration. It uses FastAPI (through LangGraphs uvicorn server) and requires a `GEMINI_API_KEY` in `.env` for the LLM calls. In production, LangGraph uses Redis and Postgres (the quickstart includes a `docker-compose.yml` to run these services).

**Summary:** The current system is optimized for *research Q\&A*: it takes a single query and goes through a **multi-step, looped workflow** (query generation � web search � gap reflection � refinement � answer) to output a **single answer message with citations**. The frontend is designed as a conversational interface where the user enters a question and sees the agents process (via timeline) and final answer.

## Mapping Existing Capabilities to New Requirements

The existing LangGraph architecture provides a foundation for the new tool, but significant adaptations are needed to meet the **opinion analysis** requirements:

* **Multi-step Processing:** Both systems are multi-step, but the *nature* of steps differs. The quickstarts steps (search-focused) must be replaced with steps for **NLP analysis of transcripts** (segmentation, classification, summarization, etc.). We will reuse the ability to chain LLM calls and functions in a graph, leveraging LangGraphs sequential and conditional flows, but configure new nodes tailored to this pipeline.

* **Parallel and Iterative Workflow:** The quickstart demonstrates parallel execution (multiple search queries in parallel) and iterative loops (repeat searches until sufficient info). For transcript analysis, we similarly need to handle **multiple segments** concurrently or sequentially. We can utilize **parallel node spawning** to analyze multiple transcript segments simultaneously (to speed up processing of many segments, similar to parallel web searches) or implement a loop to iterate through segments. The LangGraph `Send(...)` mechanism used for spawning search tasks will be repurposed for distributing work over transcript chunks. The reflection loop logic in quickstart (deciding to continue or stop) could inspire a **validation step** in our pipeline  for example, an agent reflection on whether all tensions have been found or if further passes are needed (though initial version may simply process once without iterative refinement).

* **Structured State & Memory:** In the Q\&A agent, state is maintained across steps (accumulating results, tracking loops, etc.). We will define a new state structure to hold intermediate results of transcript analysis: e.g., list of identified segments, current segment being analyzed, lists of outputs for each category, etc. LangGraphs state passing will ensure context (like the current transcript excerpt and previously extracted info) can flow between nodes. The system will also benefit from **long-term memory/persistence**: using LangGraphs Postgres integration to save analysis results and allow session resumes. For instance, once a transcript is processed, those results could be stored and retrieved instead of recomputation, and user edits could be logged for continuous learning.

* **Tool Use and External Libraries:** The original agent uses an external **Google Search tool** via the Gemini API. In our case, we wont call external search, but we will integrate **NLP tools** (like spaCy for French parsing, sentiment lexicons, etc.) as part of the workflow. We can implement these as custom LangGraph nodes (pure Python functions) or pre-processing steps. For example, a *segmentation node* might use spaCy rules to split text by conjunctions like "mais". The agent can also leverage **domain knowledge** (like a dictionary of known tension codes) as a tool  e.g., a function that maps a tension to its code.

* **Prompting and LLM Usage:** The quickstarts prompt templates will be replaced with prompts suited to classification and summarization tasks. We can still use **structured output formatting** (via Pydantic schemas and `llm.with_structured_output(...)` as seen in the quickstart) for reliable parsing of LLM outputs (for example, to get a JSON of categories). The agent will likely call an LLM multiple times per segment (unless we consolidate tasks). We might use the same Gemini API (if multilingual and capable in French) or switch to another model (OpenAI GPT-4 or a local French model). The architecture allows switching LLM providers via configuration, similar to how `ChatGoogleGenerativeAI` is initialized with model names from config. This flexibility will be used to possibly incorporate a French LLM or fine-tuned model in the future.

* **Frontend UI adjustments:** The existing UI is geared to a Q\&A chat format (one input, one answer). We need to transform it into a **data review interface** for transcripts:

  * Instead of a single input question, the UI must allow users to **upload a transcript file or select a discussion**. We will reuse the frontends ability to send data to the backend (it currently sends a text query via the InputForm component) but extend it to handle file input or a large text input area.
  * The output is not a single message but a **table of results**. We may display it as an interactive table with 12 columns (scrollable) or a series of cards for each identified tension. The ActivityTimeline could be repurposed to show steps (e.g., "8 segments found", "Segment 1 categorized...") for transparency, but the primary output view should list the extracted tensions with their categories.
  * The UI must support **editing**: for each extracted entry, researchers should be able to modify fields (e.g., correct the concept label or synthesis text). Well incorporate editable fields or an Edit mode for the table rows. Edits can be sent back to the backend (perhaps via a save button that writes changes to the database or state).
  * **Visualizations:** Though not in the quickstart, we plan to add modules for visualization (maybe a separate tab or component). For example, charts of frequency of concepts or a network graph of paradox relationships could be integrated using a library like D3 or Chart.js. This is an extension beyond the original scope, but the plan is to structure the data such that visualizations (like distribution of themes or concept co-occurrences) can be generated on the fly from the results.

* **Local Docker Deployment:** The original project already supports Docker deployment with all pieces (frontend, backend, Redis, Postgres) orchestrated via `docker-compose`. We will build on this. The modifications (adding spaCy, etc.) need to be reflected in the Dockerfile and compose setup. The system will still run as a single container serving both backend API and static frontend (as in quickstart, the Dockerfile copies the built frontend into the Python image). We must ensure any new dependencies (e.g., spaCy model downloads, additional Python libraries) are included in the build. Also, if we use a different LLM (OpenAI or a local one), we may need to pass a different API key or model weights into the container.

In summary, the LangGraph quickstart provides a strong skeleton for orchestrating complex LLM workflows with a full-stack approach. We will **refactor the backend graph** to implement the transcript analysis pipeline and **adjust the frontend** for multi-record output and editing, while preserving the dev/reload convenience and deployment pipeline of the original project.

## Refactoring Plan by Module

### Backend Refactoring

We will rework the backend agent code to implement the new analysis workflow, while pruning the search-oriented logic. Below is a file-by-file plan of changes:

* **`backend/src/agent/graph.py`**  **Rewrite the agents graph definition** to represent our transcript processing steps instead of web research:

  * **New Nodes:** Implement nodes corresponding to each major pipeline stage:

    1. `segment_transcript`  reads the full transcript text (from user input/state) and outputs a list of relevant segments (text spans) likely containing tensions.
    2. `analyze_segment`  takes one segment and produces the structured analysis (all required fields for that segment).
    3. (Optional) `review_analysis`  an agent reflection node that checks if additional tensions might be missing or if any results need double-check. This could use the entire transcript and the list of extracted tensions to identify gaps (similar in spirit to the original `reflection`).
    4. `finalize_output`  aggregates results from all segments into the final format (e.g., compile into a CSV or JSON, or prepare the message for the frontend).
  * **Graph Flow:**

    * Start � `segment_transcript` (initial node).
    * `segment_transcript` � use a conditional edge (like `continue_to_analysis`) to spawn an `analyze_segment` node for each segment. This is analogous to how `generate_query` spawned multiple `web_research` nodes. We will create a function (e.g., `spawn_analysis_tasks(state)`) that returns a list of `Send("analyze_segment", {"segment": segment_text, "id": idx})` for each segment in state. This parallelizes the analysis of segments. Each `analyze_segment` node will receive a chunk of text and an ID.
    * The outputs of all `analyze_segment` nodes will be merged back into state. We will design `analyze_segment` to return its results in list form (each field as a single-element list, or a single dict that LangGraph can combine into a list of dicts). For example, if `analyze_segment` returns `{"concept": [concept_str], "tension": [tension_str], ... }`, after N segments, the state might have `concept` as a list of N concepts, `tension` as a list of N tensions, etc. Alternatively, we return a custom Pydantic model for one result and rely on LangGraph to accumulate them into a list  well confirm LangGraphs merging behavior and use whichever is consistent.
    * After all segments are processed, add an edge from `analyze_segment` to a `review_analysis` node (if implemented). This could be a conditional edge triggered once all segments are done (similar to how after all `web_research` tasks completed, the original graph went into `reflection`). The `review_analysis` node might inspect the list of extracted tensions and possibly spawn another iteration (e.g., if it finds a knowledge gap). This is an advanced feature; for the initial version, we might skip iterative review and directly proceed to finalize.
    * Finally, `finalize_output` node will run. This could simply take the accumulated results and format them. For instance, it might produce a final `messages` entry (an AIMessage) containing a summary or a note that analysis is complete and results are ready. Or it might write results to a persistent store.
    * End of graph: `finalize_output` � END (graph termination).
  * **Remove Old Nodes:** Delete or disable the original nodes (`generate_query`, `web_research`, `reflection`, `finalize_answer`) and their edges. Those are not needed in the new workflow. Instead, integrate any needed parts of their logic into the new nodes. For example, `generate_query` and `web_research` logic (LLM calls and Google API usage) will be entirely removed. The `reflection` and loop logic may be partially reused conceptually (e.g., how to loop until a condition).
  * **Why & Implementation:** These changes repurpose the backbone of the agent for the new domain. Implementing this via LangGraph nodes allows us to maintain a clear modular structure and potential parallelism. We will follow the LangGraph pattern shown in the quickstart: use `builder = StateGraph(NewOverallState)` then `builder.add_node("segment_transcript", segment_transcript)`, etc., then `builder.add_edge(START, "segment_transcript")`, and so on. A conditional edge with a custom function (like our `spawn_analysis_tasks`) will connect `segment_transcript` to multiple `analyze_segment` nodes (similar to `continue_to_web_research`). We will ensure to call `builder.add_conditional_edges("segment_transcript", spawn_analysis_tasks, ["analyze_segment"])` to fan out. After that, either use a direct edge from `analyze_segment` to finalize (LangGraph likely treats multiple parallel nodes connecting to the same next node as synchronization) or a conditional edge that triggers finalize after the last segment. We need to confirm how to gather parallel results; one approach is to have `analyze_segment` always return and proceed to the same next node (which only executes after all parallel tasks complete).
  * **Structured Outputs:** We might utilize `langchain_core.runnables.StructuredRunnable` or the `with_structured_output` approach as in the original for certain nodes. For example, if `analyze_segment` uses a single LLM call to output JSON of all fields, we will define a Pydantic schema (like `FullAnalysisResult` with fields: concept, reformulated\_item, original\_item, details, synthesis, theme, period, code, etc.) and do `llm = ChatModel(...).with_structured_output(FullAnalysisResult)`. This function will be called inside `analyze_segment`. The node function then simply returns the parsed object as a dict. This yields clean data for each segment without manual string parsing.
  * **Parallel vs Sequential:** If needed, we can also implement sequential looping rather than parallel. For example, use a state index and have `analyze_segment` call itself for the next index until done (similar to recursion or iterative edge). However, this is more complex; the parallel spawn approach is simpler and leverages LangGraphs concurrency if available. We will proceed with parallel spawns for clarity, and throttle concurrency if needed (e.g., config says not to process more than M segments at once to manage LLM load).

* **`backend/src/agent/state.py`**  **Redefine the state schemas** for the new workflow:

  * Create a `TranscriptState` or modify `OverallState` to include the input transcript text (or a reference to it). For example: `OverallState = TypedDict("OverallState", {"transcript": str, "segments": List[str], "results": List[AnalysisResult], ...})`. We might define multiple state classes:

    * `SegmentationState` carrying the list of segments (output of `segment_transcript`).
    * `SegmentAnalysisState` for an individual segments processing (including that segments text and possibly a placeholder for its output).
    * `AnalysisResult` model (possibly as a Pydantic BaseModel) for the final fields of one tension.
  * The quickstarts state definitions like `QueryGenerationState`, `WebSearchState` will be removed. Instead, define analogous structures:

    * e.g., `SegmentState` could contain `segment` (the text snippet) and maybe an `id`.
    * We might not need a separate ReflectionState unless we implement a review loop; if we do, it might contain something like `remaining_segments` or a flag `all_tensions_found`.
    * If using structured output models for LLM, we can incorporate them directly (e.g., if `FullAnalysisResult` is a Pydantic model, our state can use it or its fields).
  * **Session Metadata:** Ensure state can carry metadata like the current **Group Code (Entretien)** and **Period** if those arent directly inferable from text. Perhaps the transcript input is associated with a group and year. We might pass that in state (for instance, store `state["group_id"] = "Groupe_A_2023"` if known, to fill the Code Entretien column). If transcripts are processed one file at a time, we might get this from file name or user input.
  * **Why:** The state definitions are crucial for type safety and for LangGraph to validate data between nodes. We update them so each nodes expectations match the new data flow: e.g., `segment_transcript` will take an OverallState with `transcript` and output a state with `segments` list; `analyze_segment` will take a state that includes a `segment` and produce part of the analysis. By defining the schemas, we also prepare for using `with_structured_output` which requires Pydantic models for the JSON structure.

* **`backend/src/agent/prompts.py`**  **Design new prompt templates** for each LLM-powered step:

  * Remove the Gemini search prompts (`query_writer_instructions`, `web_searcher_instructions`, etc.).
  * Add prompts such as:

    * `segmentation_instructions`: If using an LLM for segmentation, a prompt like: *Identify all excerpts in the following transcript that express a paradox or tension (contrasting ideas using words like 'mais', 'pourtant', etc.). Provide each excerpt, up to 2-5 sentences, that encapsulates a single tension.* This could instruct the model to output a list of text spans or line numbers. We might use structured output or instruct it to list them as bullet points.
    * `tension_extraction_instructions`: Prompt for extracting the core tension from a given excerpt. E.g.: *Analyze the following excerpt and identify the two opposing concepts or viewpoints it contains. Return: (1) a direct quote of the original statement that captures the opposition (the **Original First-Order Item**), and (2) a concise formulation of the tension as 'X vs Y' (the **Reformulated First-Order Item**).* We can also ask for supporting detail if needed (but since the excerpt itself is detail, maybe not).
    * `categorization_instructions`: Prompt to categorize a given tension into higher-order concepts. *Given the tension 'X vs Y' and its context, determine the broad category (Second-Order Concept) it falls under (e.g., Accumulation/Partage, Croissance/Soutenabilit�, etc.), and assign the appropriate specific code if known. Also determine whether it relates to 'L�gitimit�' or 'Performance', and whether it is stated as an observation (C) or stereotype (S) and if its facilitating (IFa) or hindering (IFr) future vision. Respond with a JSON containing fields: concept, code, theme, imaginaire.* We might break this into sub-prompts if its too much at once.
    * `synthesis_instructions`: *Summarize the essence of this tension in one sentence (the Synth�se). Focus on the core conflict.*
    * Alternatively, a single prompt could do all these for one segment, but multi-step prompts give us more control and the ability to insert rules-based checks in between. We will likely create distinct templates for each task (extraction, classification, synthesis) and use them in sequence within `analyze_segment` or split into sub-nodes.
  * Include **few-shot examples** in prompts for reliability: For instance, provide a short example excerpt and a model answer (with all fields filled) in the prompt to demonstrate format. This is part of our prompt strategy (detailed below) and can be included in these template strings.
  * Ensure prompts are in **French or bilingual** so that the model handles French text properly. For example, category names and instructions can be given in French (since output fields like L�gitimit� or Accumulation/Partage are domain-specific French terms).
  * **Why:** Clear, structured prompts are needed to guide the LLM in this complex classification task. The new prompts will encapsulate domain knowledge (e.g., instructing about paradox markers like *"dun c�t�... de lautre"*, the list of possible second-order concepts, etc.). By adding these as part of the prompt or as a reference (perhaps we provide the list of concept labels in the prompt context), we ground the models outputs in the expected taxonomy.

* **`backend/src/agent/tools_and_schemas.py`**  **Introduce new schemas and possibly tools**:

  * Define Pydantic models for structured outputs used in prompts:

    * e.g., `SegmentsList` schema with one field: `segments: List[str]` for segmentation output.
    * `TensionExtraction` schema with fields like `original_item: str` and `reformulated_item: str`.
    * `Categorization` schema with fields like `second_order: str`, `code: str`, `theme: str`, `imaginaire: str` (where `imaginaire` could be something like `"C (IFa)"` etc.).
    * `FullAnalysisResult` schema combining all fields for a row, if we decide to do one-shot output for a segment.
  * These schemas will be used with `llm.with_structured_output(...)` so that the LLMs JSON can be parsed into them automatically.
  * Implement mapping dictionaries or helper functions as *tools*:

    * A dictionary for known `first_order reformulated -> code` mappings (from the provided data, we can compile pairs like `"Accumulation vs. Partage" -> "10.tensions.richesse.communs"` etc. from examples). We might store this mapping in a JSON or Python dict and load it here.
    * Functions like `assign_code(reformulated_item: str) -> str`: looks up the reformulated tension in the mapping and returns the code (or "Unknown" if not found). This can be called inside a node or by the LLM (we might also consider letting the LLM produce the code by giving it the mapping as context).
    * If spaCy is used for segmentation or part-of-speech, define a function `find_tension_spans(transcript: str) -> List[str]` that does rule-based extraction (find sentences containing contrastive conjunctions, then expand to a few sentences around them). This function can serve as the implementation of the `segment_transcript` node instead of an LLM.
    * Sentiment/tonality detection: possibly a small function using a lexicon to decide IFa/IFr (facilitating vs hindering) by checking for positive vs negative tone words related to future or change. Alternatively, we rely on the LLM classification.
  * Remove unused schemas: `SearchQueryList` and `Reflection` from the old agent can be deleted, as they pertain only to web search output. Similarly, if any tool specific to Google Search or Gemini is present, it can be removed.
  * **Why:** These schema definitions will streamline parsing and validating the outputs at each step, reducing errors. The mapping tools encode domain-specific rules (e.g., how to get Code sp� or how to identify a period from text) that arent naturally known by an LLM, making the agent more deterministic for those fields.

* **`backend/src/agent/utils.py`**  **Adapt utility functions**:

  * Many of the current util functions relate to web search results and citation formatting, which we can drop. For example, `get_citations`, `insert_citation_markers`, `resolve_urls` are irrelevant once we remove external search.
  * Add new utilities:

    * `clean_transcript(text: str)`: remove timestamps or speaker names if needed (the data quality notes mention no speaker IDs but if transcripts have some artifacts, clean them).
    * `detect_period(text: str) -> str`: rule or regex to find 2023 or 2050 in a segment. This can help set the **P�riode** field automatically. If found in the segment, use that; if not, perhaps default to the session context (e.g., Group\_A\_2023 implies period 2023).
    * `determine_theme(text: str) -> str`: simple logic to decide **Th�me** (L�gitimit� vs Performance) by looking for certain keywords. The data doc suggests *transparence implies legitimacy, rentabilit� implies performance*. We can create a small keyword list for each theme and check occurrence. This can serve as a baseline classifier which the LLM can override if needed.
    * If needed, `format_csv(results: List[AnalysisResult]) -> str/csv`: generates a CSV string or file from the results list. Possibly integrated into finalize node.
  * Rationale: keeping these domain-specific helper functions separate improves clarity and testability. They implement the rule-based parts of the pipeline outlined by the experts (temporal tagging, theme classification by keywords, etc.), complementing the LLMs capabilities.

* **`backend/src/agent/configuration.py`**  **Update configuration options**:

  * Remove search-related config fields (like `number_of_initial_queries`, `max_research_loops`) and add new ones as needed:

    * `max_segments`  perhaps to limit how many segments to extract (in case a transcript is extremely long, we might restrict for performance or interactive processing).
    * `model_temperature`  separate temperature settings for different tasks: e.g., a lower temperature for categorization (for consistency) and a slightly higher one for summarization (for fluency). We can allow these in config so theyre tunable.
    * `use_gpu` or model selection flags  if planning to support local models (like a flag to use `OpenAIGPT4` vs `Gemini` vs `LocalModel`). The config can store which LLM class or API to use. For instance, an enum for `llm_provider` (google\_genai, openai, transformers\_local) and corresponding model names or paths.
    * Possibly credentials or keys for alternative services (like `OPENAI_API_KEY`) if we integrate them.
  * The Configuration class can also hold lists like `theme_keywords` or `concept_keywords` if we want to store them centrally.
  * Ensure `.env.example` is updated with any new required env vars (like OpenAI key if needed, or mention that GEMINI\_API\_KEY is optional if user chooses a different model).

* **Backend API Endpoints:** Currently, the FastAPI app serves the frontend at `/app` and presumably LangGraph provides an endpoint for running the agent (likely the LangGraph GraphQL or REST interface on the compiled graph). We should verify how the frontend triggers the backend:

  * Possibly the frontend sends a request to `POST /runs` or similar (LangGraphs run creation) when the user submits a query. If so, we need to ensure it passes the transcript text properly in the request (as the `messages` or a variable). We might have to adjust the frontend to send the whole transcript as the user message content in the conversation history.
  * If customizing, we could create an explicit endpoint like `/analyze_transcript` that accepts a file or text, invokes the LangGraph run programmatically, then returns the structured result. However, leveraging LangGraphs built-in job handling is preferred. We might not need to write a custom route if LangGraph automatically mounts one for the graph (the `LANGSERVE_GRAPHS` env in Dockerfile suggests the graph is served as an endpoint named "agent").
  * We will confirm this by testing the quickstarts behavior; likely, the frontend hitting the backend triggers the graphs execution, then the result is streamed. Well ensure our `graph` variable at end of `graph.py` (compiled with `name="paradox-agent"` for example) is still being picked up by LangGraph and accessible.
  * If needed, write minimal logic to accept a file upload. FastAPI can handle file uploads; we could mount a small router for this. But a simpler approach: allow the frontend to send transcript text (since its not extremely large, text is fine) in the JSON payload for running the graph.

### Frontend Refactoring

The frontend must transform from a Q\&A interface to a **transcript analysis dashboard**. Planned changes per component:

* **`frontend/src/App.tsx`**  Adapt the main application flow:

  * The App currently likely sets up routes and an `InputForm` and then displays either the conversation or some main screen. We will modify it to use a different initial view: e.g., a file upload form or selection interface for transcripts.
  * If multiple transcripts (groups AH) are available, we can provide a dropdown or list to choose one to analyze. Or allow uploading a new file.
  * Ensure the `apiUrl` is correctly set for our use (still will use `http://localhost:2024` in dev, or 8123 in Docker unless changed). If we rename the graph or endpoint, update it here.
  * Possibly include a state to store the analysis results fetched, to pass to child components like a ResultsTable or Visualization.
  * Remove or repurpose the chat-related state. For instance, if App was storing messages for the timeline, we may drop that or use it differently (maybe use it to store log messages of the agent process for debugging).
  * **Why:** App.tsx is the integration point between backend and UI; these changes will ensure it sends the right data and properly receives the results for display.

* **`frontend/src/components/InputForm.tsx`**  Change the input mechanism:

  * Replace the single-line text input (for question) with a **text area** or **file input**:

    * If file input: allow uploading a `.txt` or `.docx` (if we implement docx parsing) and then read it client-side to text (or send file directly to backend).
    * If text area: user can paste transcript text directly. Given 300 pages \~ maybe 150k words, that's quite large for a text area, but could be manageable if chunked. Perhaps file upload is cleaner.
  * Add fields for metadata if needed (like a dropdown for theme if known, or to specify which analysis steps to run or model selection if we want to expose).
  * The forms submit handler should bundle the transcript text (or file content) into the request payload. Likely, it will now call something like `fetch(apiUrl + "/runs", { method: 'POST', body: JSON.stringify({ transcript: ..., ... }) })` or use LangGraphs JS client if provided.
  * If using file upload, we might have to use FormData. Alternatively, we convert the file to text in the browser (FileReader) and then treat it like a big text input.
  * Remove the concept of a "user message" in conversation  instead, one analysis per form submit.
  * **Why:** The input form should accommodate large texts rather than short questions, and possibly more user guidance. This sets the stage for analyzing arbitrary transcripts.

* **New Component: ResultsTable/ResultsViewer**  Create a component to display the analysis output:

  * This component will render the 12 columns for each identified tension. Given potentially \~50 rows per group, a table is suitable. We will implement it with scrollable overflow if too tall.
  * Each cell either displays text or a category label. We will make certain cells editable:

    * The text fields **Original Item**, **Details**, **Synthesis** might need editing for corrections, though **Details** is extracted verbatim so maybe not edited (unless to trim or tweak).
    * Category fields **Second-Order Concept**, **Reformulated Item**, **Theme**, **Imaginaire**, etc., should be editable via dropdown or free text, so researchers can adjust classification.
    * When a cell is edited, update the component state. Possibly highlight edited cells.
  * Include a Save Changes button that sends the modified data back to backend (maybe via a PUT/PATCH request or triggers a new LangGraph run that just logs feedback).
  * This component can also incorporate small visual cues: e.g., color-code the **Th�me** column (green for Legitimit�, blue for Performance), or icons for IFa/IFr.
  * If feasible, implement row selection highlight: when user clicks a row, we could show the corresponding **Details** excerpt in a side panel or highlight it within the full transcript (if we display the entire transcript text somewhere).
  * **Why:** An interactive results table is central for the collaborative aspect  it allows viewing all tensions at a glance and fine-tuning the results. The quickstart timeline was more for debugging; our table focuses on end-user interpretability and editing.

* **New Component: TranscriptViewer**  (Optional) A component to show the original transcript with highlights:

  * To improve interpretability, we can display the full transcript text and highlight segments that were extracted as tensions (maybe color-shade those sentences). This helps users see context beyond the excerpt.
  * This could be implemented by taking the original transcript and wrapping spans of text (matching the `Details` excerpts) in a highlighted `<mark>`. Clicking on a tension row in the ResultsTable could scroll/focus the transcript viewer to that segment.
  * This requires the frontend to have access to the full transcript text (which we do, since we sent it). We can pass it into ResultsTable or context so TranscriptViewer knows it.
  * Not a priority for initial version, but a recommended addition for interpretability.

* **ActivityTimeline (optional)**  We might simplify or remove the timeline UI for the end-user release. The timeline that shows Agent generated queries etc., may not be needed for them. However, during development, it could be useful to monitor the agents steps (segmentation done, segment 1 analyzed, etc.). We could keep it behind a debug flag or repurpose it to log summary of each step (e.g., Found 8 tension segments, Analyzed segment 1: Identified tension = X vs Y). If we keep it:

  * Ensure it listens to the backends streaming events. LangGraph may stream intermediate messages. We could send custom progress messages via the state `messages` (for example, the agent could append a message after each segment analyzed).
  * This timeline would then show a step-by-step progress which can reassure the user that work is ongoing (since analyzing \~50 segments might take some time, streaming status is helpful). For now, we plan to at least output a message when done.

* **General UI/UX**:

  * Add a loading indicator or progress bar while analysis is running (especially if it's not streaming many intermediate messages).
  * Possibly disable the input form during processing to prevent double submissions.
  * After analysis, allow user to download results as CSV/Excel. We can implement a button that calls an endpoint like `/export_csv?run_id=XYZ` which the backend can implement to send a CSV file of the last runs results. Alternatively, generate a CSV in JavaScript from the results and use a blob download. Either approach is fine; given we have the data in front-end state, a client-side CSV export might be simplest.
  * Check for performance with large text in browser; we may chunk it or at least warn if extremely large. The target 300 pages might be borderline  we might test with \~100k characters in a text area.

### Adding New Dependencies & Files

* **NLP Libraries:** Integrate spaCy with French model for linguistic tasks. We will:

  * Add `spacy` and `fr-core-news-lg` (or `fr-core-news-md` for smaller) to the `requirements.txt` or `pyproject.toml` of backend. Also ensure the Dockerfile installs the model (e.g., `RUN python -m spacy download fr_core_news_lg`) so its available at runtime.
  * Use spaCy for sentence segmentation and perhaps for part-of-speech to refine tension extraction (e.g., ensure that around "mais" we indeed have a clause on each side).
  * If spaCy is heavy to load for each run, consider loading one global nlp instance at startup (perhaps in `agent/utils.py` or `app.py`) and reuse it in the node functions. This might require some global or singleton pattern. Alternatively, use quick regex as a first cut (for speed) and spaCy only if needed.

* **Domain Data Files:** Possibly include a JSON or CSV in the repo for the mapping of tensions to codes and second-order concepts:

  * e.g., `backend/src/agent/taxonomy.json` containing an array of objects: `{ "second_order": "Accumulation/Partage", "first_order_variants": ["Accumulation vs Partage", "actionnariat vs coop�ratif", ...], "code": "10.tensions.richesse.communs", "model_tension": "accumulation vs. redistribution", "change_tension": "Transition vers des mod�les coop�ratifs" }`, etc. We can populate this from the documentations examples and existing annotated data if available.
  * Then, provide a utility to load this file. The agent can consult this structure to:

    * Find which second\_order concept and code match a given reformulated tension (or at least get close via fuzzy match of keywords).
    * Provide the standardized **Tension de mod�le** (model-level tension) and **Tension li�e au changement** if available. If the taxonomy lists those fields, we can fill columns 11 and 12 directly. For instance, for concept "Accumulation/Partage", model tension might be "accumulation vs. redistribution", and change tension something like "moving from accumulation to sharing models".
  * This file would be new in the repo, perhaps stored in `backend/data/` or directly in `agent/`.
  * New Python file `backend/src/agent/taxonomy.py`: we could encapsulate the logic to search within this taxonomy data (e.g., a function `lookup_tension(reformulated_item) -> (second_order, code, model_tension, change_tension)` that tries to match the reformulated text or its keywords).

* **Potential Model Integration:** If the user might want to run an open-source model locally (instead of Gemini which requires API access), we should add support:

  * Possibly include `transformers` or `OpenAI` Python SDK in requirements. We can use LangChain or direct API calls to OpenAI. But mindful of offline requirement, perhaps an open model like BLOOM or LLaMA2 (French tuned) could be used. However, running those for 50 segments might be slow without GPU.
  * At least, structure the code to allow easy swapping: e.g., a function `get_llm(model_type)` that returns a LangChain LLM instance based on config. If `model_type = "gemini"`, returns `ChatGoogleGenerativeAI(...)`; if `model_type = "openai"`, returns `ChatOpenAI(...)`; if `model_type = "local"`, returns a `HuggingFacePipeline` with a local model.
  * This may also involve adding a dependency for the chosen local model (e.g., `pip install transformers accelerate`).
  * Document in README how to choose the model. Possibly provide an .env flag or config setting.

* **Testing and Example Data:** It could be helpful to include a **small example transcript** (few lines) and an expected output (CSV) in a `examples/` directory for demonstration and testing. Not strictly necessary for deployment, but aids development.

### Removing Unnecessary Components

To avoid confusion and reduce bloat, we will strip out features not needed in the new context:

* **Google Search & Gemini-specific logic:** All code involving the `google.genai.Client` and the tool usage in `web_research` will be removed. The environment variable `GEMINI_API_KEY` might still be used if we continue using Gemini for LLM, but if switching to another LLM, we can remove this requirement (or make it optional). The search API dependencies (if any additional) can be dropped.
* **Citations Handling:** We will not produce web citations, so remove `sources_gathered` from state and any insertion of `[1]`, `[2]` markers in text. If the final answer assembly logic in `finalize_answer` is kept around, it will be repurposed to maybe compile the results textually, but citation handling lines will go.
* **Reflection Loop:** If we decide not to implement an iterative refinement for missing tensions initially, we can remove the loop logic akin to `evaluate_research`. That means the agent will do a single pass. (We can keep the door open to add a refinement node later if needed).
* **LangSmith Integration:** The quickstarts Docker compose mentions `LANGSMITH_API_KEY` (for LangChain monitoring). If not needed, we wont require the user to set this. We can disable any LangSmith callbacks unless the user specifically wants to use it for logging. Removing it simplifies deployment (one less key).
* **Front-end chat history:** Remove the display of user question and AI answer in a chat bubble format (since our output isnt a conversational turn). Instead of showing an AI message, well show the results table. This means any code treating `messages` in state for UI can be trimmed down. For example, if the App or timeline expects a final assistant message content (which was the answer text), we might bypass that and directly render our structured results.
* **Mobile UI optimizations for table:** If the original was a chat, it might be mobile-friendly. Our table might not be easily viewable on mobile due to width. This is a minor concern, but we could hide some columns on small screens or require horizontal scroll. We wont prioritize mobile in this refactor but note the limitation.

## Redesigned Agent Workflow

We propose a **modular agent workflow** that breaks down the complex task into manageable LLM or rule-based nodes, with the ability for experts to intervene at each stage. Below is the end-to-end process the agent will execute, aligning with expert methodology:

1. **Preprocessing & Segmentation (Node: `segment_transcript`)**
   **Input:** Raw transcript text (one discussion group).
   **Process:**

   * Clean the text (remove timestamps or irrelevant markers).
   * Identify segments containing potential paradoxes/tensions. Use a combination of rule-based scanning for contrastive connectors (e.g., *"mais", "cependant", "d'un c�t�... de l'autre..."*) and possibly an LLM prompt for segmentation. For instance, the agent might prompt: *"Find up to 10 key tension excerpts in this text."*
   * Alternatively, run a spaCy pipeline to split sentences and find sentences with opposing clauses (we can detect dependency relations for "mais" to ensure it's used as a conjunction). Group 25 sentences around each occurrence to form a segment (as experts say relevant spans can vary in length).
     **Output:** A list of segment texts, each presumably containing one tension. This populates the **Details** column content. Also output meta-tags if found: e.g., tag each segment with the Period (if 2050 or 2023 appears in it), and potentially with the Group code (which might be constant for the whole transcript or deduced from the file name).
     *Rationale:* This step mimics the manual expert step of scanning transcripts for paradoxical statements. By isolating segments first, subsequent analysis can treat each tension independently, which simplifies LLM prompts and allows parallel processing.

2. **Tension Analysis for Each Segment (Node: `analyze_segment`)**  *This is a composite of sub-tasks executed sequentially for each segment:*

   * **2a. Extract Original and Reformulated Items:**
     **Input:** One segment (detail excerpt).
     **Process:** The agent identifies the exact quote that represents the tension and formulates it as a concise **X vs Y** statement. For example, from *"La croissance est n�cessaire, mais elle nuit � la soutenabilit�."*, it would extract original phrasing (could be the whole sentence or clause) and reformulate as *"Croissance vs. Soutenabilit�"*.
     **Implementation:** Use an LLM with a structured prompt focusing on this task. Possibly output a JSON: `{ "original_item": "...", "reformulated_item": "A vs B" }`. The model will be instructed to preserve the exact wording from text for the original item and to ensure the reformulated is a balanced "A vs B" phrase (well have it capitalize or format similar to examples in the data). If the segment oddly contains multiple distinct tensions, the model could either return multiple (we might instruct to choose the primary one, or return an array  but prefer one tension per segment as assumption).
     **Output to state:** `original_item`, `reformulated_item` for this segment.

   * **2b. Categorize Second-Order Concept & Assign Code:**
     **Input:** The identified reformulated tension (A vs B) and possibly the full segment for context.
     **Process:** Determine which broad **Second-Order Concept** this tension falls under (e.g., "Accumulation/Partage", "Croissance/Soutenabilit�", etc.). Also find the corresponding **Code sp�** (specific code) if available.
     **Implementation:** This can combine rules and LLM:

     * First, attempt a direct lookup: compare keywords in `reformulated_item` against our taxonomy dictionary (e.g., if "Profit vs. d�croissance" appears, we know it's under "croissance/soutenabilit�"). If a match is found, we retrieve the exact standardized label and code.
     * If no match or unsure, use an LLM prompt: *"Which of these categories best fits 'X vs Y'? Answer with the exact category label."* Provide it with the list of \~15 second-order concepts from the data schema as options. The model will output one of them. Then use that to fetch the code via mapping. If even the concept is new, we may output "Other" and code "Unknown".
       **Output:** `second_order_concept` and `code` fields for the segment.

   * **2c. Determine Theme (Legitimacy vs Performance):**
     **Input:** The segment text (and possibly the emerging synthesis).
     **Process:** Classify whether the tension relates to **L�gitimit�** (social, environmental values) or **Performance** (economic, efficiency outcomes).
     **Implementation:** We apply a rule-based classifier first: e.g., search for presence of keywords list for legitimacy (e.g., *transparence, �quit�, communs, environnement, soci�t�* etc.) and performance (e.g., *profit, efficacit�, rentabilit�, comp�titivit�*). If one list matches more, assign that theme. If unclear or mixed, we can default to a dominant one or ask an LLM: *"Does this excerpt focus more on legitimacy (values, ethics) or performance (profit, efficiency)?".* The LLM can output "L�gitimit�" or "Performance". We aim for high precision in this field because the keywords are relatively distinct.
     **Output:** `theme` field.

   * **2d. Imaginaire Classification (C/S and IFa/IFr):**
     **Input:** Full segment and possibly the synthesis (once generated).
     **Process:** Determine whether the statement is a **Constat (C)** or **St�r�otype (S)** and whether it reflects an **Imaginaire Facilitant (IFa)** or **Imaginaire Freinant (IFr)**. In simpler terms, is the speaker expressing an observation based on evidence (C) or a general belief (S)? And does the content promote a sustainable future (facilitating) or imply a barrier to it (hindering)?
     **Implementation:** This is nuanced, so we likely use an LLM with guidelines: e.g., *"Classify the tone: is the speaker making a factual observation (C) or a stereotype/generalization (S)? Also, does the content suggest a positive, enabling vision of the future (IFa) or a negative, limiting view (IFr)? Answer with one of: 'C (IFa)', 'C (IFr)', 'S (IFa)', 'S (IFr)'."* We might give definitions and examples in the prompt. The model can decide based on sentiment and wording (e.g., statements like "We will never be able to..." might be Stereotype hindering (S IFr)). We will keep temperature low to maintain consistency.
     **Output:** `imaginaire` field (formatted as in the dataset, e.g. "C (IFa)").

   * **2e. Synthesize the Tension:**
     **Input:** The segment (and possibly the identified A vs B and concept).
     **Process:** Generate a concise **Synth�se**  one sentence capturing the paradox. This typically rephrases the two sides in a balanced way (e.g., "Tension between individual freedom and preservation of commons" for a segment about accumulation limits vs commons).
     **Implementation:** Use an LLM with an abstractive summarization prompt: *"Summarize the core tension in this excerpt in one sentence."* We can instruct it to follow the format "Tension entre X et Y." in French for consistency. We will give an example: *Excerpt: " ... " -> Synth�se: "Tension entre X et Y."* so it understands to output in French.
     **Output:** `synth�se` field.

   * **2f. Assign Model/Change Tensions:**
     **Input:** The determined second-order concept and context.
     **Process:** Fill in **Tension de mod�le** (systemic model tension) and **Tension li�e au changement** (change-related tension). These often are more abstract rewordings of the tension or direct mappings from the second-order concept.
     **Implementation:** If our taxonomy dataset includes these fields, simply lookup by second-order concept. For instance, if second\_order is "Accumulation/Partage", the model tension might be "accumulation vs. redistribution" (as per documentation), and the change tension might be "Transition vers des mod�les coop�ratifs". We populate them directly. If not in the mapping, we can attempt a simple approach: take the reformulated item and generalize terms (maybe using a thesaurus or asking an LLM to "express this tension in terms of overarching models or paradigms"). But in the interest of accuracy, it's better to define these pairs with expert input. Initially, we might populate the major ones from the provided analysis and leave others blank or identical to reformulated item.
     **Output:** `tension_modele` and `tension_changement` fields.

   Each sub-steps output is added to the segments result. In implementation, these sub-steps could be separate LangGraph nodes in sequence, or combined in fewer LLM calls (to optimize). For example, we might merge 2a and 2b into one LLM prompt that outputs original, reformulated, and concept if we trust the model, then do 2c and 2d via simpler rules, and 2e by another prompt. We will carefully orchestrate these to ensure correctness and allow intermediate verification. The **state memory** ensures the context of the segment and partial results can be passed along nodes, and any errors can trigger feedback loops (e.g., if the model fails to identify a tension, the agent could mark that segment as needs review and still continue).

3. **Aggregation & Output (Node: `finalize_output`)**
   **Input:** The collection of all analyzed segments (each with 12 fields now).
   **Process:** Compile the results into the final format. This could involve:

   * Merging the lists of results into a single data structure (like a list of dicts, or a pandas DataFrame if we use it).
   * Possibly sorting them by some key (e.g., by second-order concept or by appearance order in transcript).
   * Preparing the data for presentation: e.g., formatting a table or storing to database.
   * The agent could also generate a short **summary report** for the run  e.g., "In Group A (2023), 5 key tensions were identified across themes of Performance and Legitimacy" etc., but this is optional.
     **Output:** Typically, LangGraph would end by outputting a final `AIMessage` in `state["messages"]` with content. In our case, the final message might be something like: *"Analysis complete. Identified **N** tensions. See structured results below."* and perhaps include a markdown table preview of Concept, Reformulated Item, Synthesis, etc., truncated. However, since our frontend will render a custom table, we might not need to put all details in a message. The structured data is likely stored in memory/DB. We can choose to still have a message for completeness (for example, listing the high-level concepts found).
   * Additionally, we ensure the data is saved. LangGraph with Postgres can automatically persist the state of each run. Each `AnalysisResult` could be stored as part of the run record. This way, the frontend could query it or if the user refreshes, the data isnt lost. If not using LangGraphs persistence, we could manually implement an `export_csv` at this step (write to a file or return it).

4. **User Review & Feedback (Post-run, not an automated node)**
   After the agent finishes, the user can review the results in the GUI. If any field is incorrect, the user edits it. We plan to incorporate a feedback loop:

   * The system can log these corrections (e.g., differences between original output and edited output). This data can be used to refine the LLM prompts or train a future classifier. For instance, if the agent consistently mislabeled a certain tensions theme, that indicates a need to adjust the keyword lists or provide a better prompt hint.
   * In a more dynamic agent, we could even allow the user to signal re-run analysis with these corrections and the agent could adjust its process (e.g., if an entire second-order concept was wrongly assigned, it might re-evaluate similar items). However, for now, feedback will be used offline for model improvement.
   * We will incorporate a mechanism to incorporate changes: maybe a Finalize button that calls an endpoint to save the curated results (e.g., to a database or file). This ensures the cleaned data is stored for later use (and not overwritten by the raw model output next time).
   * Memory: If the same or similar transcripts are analyzed in the future, the agent could recall prior results to avoid redundancy. With LangGraphs persistent storage, we can implement a memory lookup at the start: e.g., if a transcripts name matches one in DB, we could fetch existing tensions as a starting point (though this goes into active learning territory; likely not needed immediately but possible).

Throughout this workflow, the agent leverages **modular tasks**: segmentation, extraction, categorization, etc., each of which can be improved or swapped out independently (for example, replacing a rule-based step with a fine-tuned classifier later). The graph architecture ensures these tasks share information via state, and allows insertion of checking nodes if necessary (like verifying if each segment got all fields). The design also supports **parallelism** in step 2, so processing time scales with number of segments (and can be parallelized if the LLM or compute resources allow multiple concurrent calls).

Moreover, this breakdown aligns with the original expert process as documented, ensuring the systems behavior is interpretable and traceable at each stage.

## Prompt Template Designs & Examples

Carefully crafted prompts will be crucial for the LLM to perform reliably. We will use **structured, example-augmented prompts** for each major LLM task. Below are key prompt templates with examples:

* **Segmentation Prompt (for LLM)**  *Goal:* Instruct the model to find tension-containing excerpts.
  **Template:**
  *"Vous �tes un assistant qui identifie des paradoxes dans une discussion. On vous fournit une transcription d'un groupe de discussion sur la durabilit� organisationnelle. Rep�rez les extraits o� un orateur exprime une tension ou un paradoxe (par exemple en utilisant 'mais', 'cependant' ou en opposant deux id�es). Listez chaque extrait pertinent en le conservant tel quel, sans le modifier, et s�parez les extraits par `***`.*"
  *"Texte : {transcript\_text}"*
  *"Extraits identifi�s : ..."*

  **Example (embedded):**
  *Texte (extrait): "Speaker A: Nous devons cro�tre pour survivre. Speaker B: Mais la croissance infinie est impossible sur une plan�te finie."*
  *Extraits identifi�s:*
  *"Speaker A: Nous devons cro�tre pour survivre. Speaker B: Mais la croissance infinie est impossible sur une plan�te finie."*

  Here we instruct the model in French to output the raw text spans. We use a delimiter (`***`) to separate multiple spans in case of more than one. The example guides the model to pick the full dialogue turn containing the paradox.

* **Tension Extraction Prompt**  *Goal:* From a given excerpt, extract original and reformulated tension.
  **Template:** (using structured output)
  *"Analyse l'extrait suivant et identifie le **paradoxe principal** qu'il exprime.*
  *- Extrait : "{segment\_text}"*
  \*Donne :

  * `original_item`: la citation pr�cise du texte qui montre le paradoxe (garde les mots exacts).
  * `reformulated_item`: le paradoxe r�sum� sous la forme "X vs Y"."\*

  We will likely attach a `with_structured_output(TensionExtraction)` to this prompt, expecting JSON like:

  ```json
  {
    "original_item": "La croissance est n�cessaire, mais elle nuit � la soutenabilit�.",
    "reformulated_item": "Croissance vs. Soutenabilit�"
  }
  ```

  **Few-shot example:** We can include an example before the actual task:
  *Exemple:*
  *- Extrait : "Il faut innover constamment, cependant cela peut �puiser les �quipes."*
  *- original\_item: "innover constamment, cependant cela peut �puiser les �quipes."*
  *- reformulated\_item: "Innovation vs. Bien-�tre des �quipes"*

  This shows the model how to pick the part of text and form the vs phrase. The example also demonstrates using synonyms for clarity ("bien-�tre des �quipes" for "�puiser les �quipes" concept).

* **Categorization Prompt**  *Goal:* Assign second-order concept and Code (if LLM-based).
  **Template:**
  *"On a identifi� le paradoxe "{reformulated\_item}". � quel concept de 2nd ordre appartient-il ? Choisissez parmi : {list\_of\_concepts}. Donnez exactement le libell�. Sil correspond � un code sp�cifique connu, fournissez le code �galement.*
  *R�pondez au format:
  `concept`: \<Concept 2nd ordre>,
  `code`: \<Code sp� ou "Unknown">.*"

  **Example usage:**
  If `reformulated_item = "Croissance vs. Soutenabilit�"`, the list\_of\_concepts will include "croissance/soutenabilit�". The model should output:

  ```json
  {
    "concept": "croissance/soutenabilit�",
    "code": "10.tensions.�cologie.croissance"
  }
  ```

  (Assuming that is the code from taxonomy). We will supply the codes for known ones in the prompt context or instruct the model to leave "Unknown" if not sure. Possibly we dont rely entirely on the model for code; instead, we will likely do code via lookup. So this prompt might actually be simplified to just get the concept, and our code assignment function does the rest.

* **Imaginaire Classification Prompt**  *Goal:* Classify C/S and IFa/IFr.
  **Template:**
  *"D�terminez si l'�nonc� suivant est un **constat (C)** ou un **st�r�otype (S)**, et s'il exprime un **imaginaire facilitant (IFa)** ou **freinant (IFr)** la vision future.
  �nonc�: "{segment\_or\_synthesis}"
  R�pondez par l'une des options: "C (IFa)", "C (IFr)", "S (IFa)", "S (IFr)". Justifiez en une phrase."*

  We included justifiez  this is optional but could help the model think, though we might not use the justification in output. Perhaps we tell it to output just the label and keep reasoning internally (to avoid parsing issues, maybe instruct "R�ponse: C (IFa). Raison: ..."). But since we'll parse this likely as text, we can allow a rationale that is ignored. We will test a few examples ourselves:

  * If the text says: "Les gens pensent que sans croissance, on ne peut pas prosp�rer"  likely a stereotype hindering (S IFr) because it's a general belief hindering new models. The model should output "S (IFr)".
  * If text: "On constate que la collaboration am�liore la performance environnementale"  observation facilitating (C IFa) -> "C (IFa)".

* **Synthesis Prompt**  *Goal:* Summarize tension in one line.
  **Template:**
  *"Synth�tisez le paradoxe suivant en une phrase concise.
  Paradoxe: {reformulated\_item}
  D�tails: "{segment\_text}"
  Format attendu: Une phrase commen�ant par "Tension entre ... et ..."."*

  **Example:**
  Paradoxe: *"Innovation vs. Tradition"*
  D�tails: *"On veut �tre innovant, mais on a peur d'abandonner nos traditions."*
  R�ponse attendue: *"Tension entre besoin d'innovation et attachement aux traditions."*

  We give the model both the reformulated label and the original excerpt to ensure it captures nuance, but instruct it to follow a pattern in French. The output will fill the **Synth�se** column.

* **Combined Prompt (alternative):** We might try a single prompt that does multiple things to reduce API calls. For instance, after getting the segments, have one prompt per segment like: *"Voici un extrait: ... Analyse-le et fournis un JSON avec les champs: original\_item, reformulated\_item, concept, theme, imaginaire, synthesis."* While tempting, this may reduce reliability (the model might confuse fields or produce invalid JSON if too much). We prefer stepwise prompting with checks. However, we will design prompts such that they could be concatenated if needed, and ensure consistency (e.g., use the same wording for categories in all prompts so the model doesn't mix languages or synonyms unexpectedly).

**Prompt Generation Strategies:**
To enhance the quality of LLM responses, we will employ these strategies:

* **Few-Shot Examples:** As shown above, each prompt template will include an example or two illustrating the desired output format. By demonstrating on a similar French excerpt, we guide the model to mimic the structure. This is especially important for structured outputs (to avoid the model deviating from JSON) and for tricky classifications (C/S, IFa/IFr). We will ensure examples are representative of real data scenarios (covering a facilitating vs hindering case, etc.).

* **Controlled Language & Hints:** We instruct in French for tasks deeply tied to French semantics (segmentation, extraction), to ensure the model fully grasps nuances. For categorical outputs, we explicitly provide the allowed values or list of categories (reducing open-ended guessing). For example, listing all second-order concepts and requiring an exact match will harness the models ability to choose from known options rather than invent new labels.

* **Structured Output with Pydantic:** Wherever feasible, use `with_structured_output` to have the model output JSON that is directly parsed. This strategy was successfully used for search queries and reflection in the original project; well extend it to our domain. It dramatically reduces the need for fragile string parsing and ensures we catch format errors immediately (the LLMs output is validated against the schema, and if it fails, an exception is raised which we can handle, e.g., by retrying with a simpler prompt or error message).

* **Temperature Tuning:** We will set **low temperature (0 to 0.3)** for classification nodes (concept, theme, imaginaire) to get consistent outputs (we want the same input to always map to the same category, not creative answers). For generative tasks like summarization (synth�se), a moderate temperature (0.7) can be used to allow fluent phrasing, but we might keep it modest to avoid too much variation in wording (since consistency in style is valued in a structured dataset). The query generation in quickstart used temperature 1.0 for brainstorming queries; in our case, creativity is less needed, so we lean more conservative.

* **Handling Ambiguity:** If the model expresses uncertainty (perhaps none of the categories clearly apply), we plan for fallback: e.g., if concept classification returns something outside our list, we map it to "Other". If imaginaire classification is uncertain, we could default to "C (IFa)" as a neutral safe guess or flag for review. Including an "Unknown/Other" option in prompts (and schema) is important to not force the model to choose incorrectly; better it says "Unknown" than a wrong category. Our system can later highlight "Unknown" entries for manual attention.

* **Iteration and Refinement:** We can incorporate a mini feedback loop for certain prompts. For example, after an LLM outputs a reformulated tension, we could automatically verify if both sides of "X vs Y" appear in the original text (or are synonyms thereof). If not, that might indicate the model misinterpreted, and we could prompt it again or mark for review. Similarly, after categorization, if the assigned second-order concept doesnt match the code (in case LLM gave a concept but a different code was looked up), well reconcile by trusting the concept label and regenerating code.

* **Memory Use in Prompts:** If needed, we can feed the model some **global context**  e.g., a list of all second-order concepts found so far in this transcript, to nudge it to be consistent (if it already assigned 10 tensions to "Accumulation/Partage" and a new similar one comes, it should probably also use that label). We could keep track and include a line like "Concepts already identified in this group: X, Y..." in the prompt, influencing the model to choose among those if appropriate. This might reduce random use of synonyms.

By combining these strategies, the agents prompts will be robust and tailored to the task, leveraging both the structured knowledge from the dataset and the generative understanding of the LLM. The prompts will be tested on sample data and iterated as needed to ensure alignment with expert annotations (for instance, verifying on a few labeled examples from the "data\_labelled.xlsx" if available).

## Deployment & Docker Adaptation

Deploying this refactored system locally in Docker requires updating the configuration to include our new dependencies and ensuring all services (LLM backend, database, etc.) work in concert:

* **Dockerfile Changes:**

  * Add installation of spaCy and the French model in the build steps. For example, after installing our backend package, add:

    ```Dockerfile
    RUN uv pip install --system spacy==3.6.1 fr-core-news-lg==3.6.1
    RUN python -m spacy download fr_core_news_lg
    ```

    (or similar, depending on version locking). We might pin to a specific version for reproducibility.
  * If using a local model, ensure the model files are available. One approach: download the model in the Dockerfile (e.g., using `curl` or `git lfs` if its HuggingFace). For instance, if we decided on a smaller French model for offline use (like CamemBERT for classification), we'd fetch or install it. This can increase image size, so we mention it as an option but might rely on API by default to keep things lighter.
  * Retain the base image `langchain/langgraph-api:3.11` for consistency, which already has Uvicorn, LangGraph, etc. This image likely includes LangChain and dependencies so it saves us from installing those. We just layer ours on top. We confirm that removing search features wont break the base image usage (the base image might have had Google API installed, which is fine to leave).
  * The `LANGGRAPH_HTTP` and `LANGSERVE_GRAPHS` env in Dockerfile still point to our app and graph by path. We will update `LANGSERVE_GRAPHS='{"agent": "/deps/backend/src/agent/graph.py:graph"}'` if the name "agent" is used or maybe rename "agent" to something like "paradox\_agent" for clarity (front-end needs to know if changed). Keeping it "agent" is okay.
  * Well remove any leftover references to search (like if the Dockerfile installed Google-specific packages; it looks like it didn't explicitly besides using the base image).
  * Verify that the `.env` file (with API keys) is still needed. If using Gemini, keep `GEMINI_API_KEY`. If using OpenAI, add `OPENAI_API_KEY`. If using local model, no key needed. We can make the key usage conditional by config.

* **Docker-Compose Updates:**

  * The original `docker-compose.yml` likely defines services for the app, Redis, and Postgres. We will maintain those because:

    * **Postgres**: needed for LangGraph to store state, and we can use it to store results. Ensure the compose has a Postgres service (with a volume for persistence) and the `backend` service has `DATABASE_URL` configured (the LangGraph docs likely have environment for DB connection). Possibly the base image or LangGraph auto-uses a default Postgres connection from env `PGURI` or something; well set accordingly.
    * **Redis**: LangGraph uses Redis for streaming outputs (pub-sub for agent messages). We keep Redis to not break real-time update functionality. The compose should have a Redis service, and the backend should have `REDIS_URL` if needed (or perhaps its auto-configured by base image to connect to a local redis).
  * We might add a volume mount for any data files (if we keep taxonomy JSON in the image, it's fine; if we want to allow mounting new transcripts, maybe expose a folder).
  * If we incorporate a local large language model, we might need to consider GPU usage. Possibly we wont go that route in initial deployment. If we did, wed need to base the image off something with CUDA and allow `runtime: nvidia` in compose. This complicates deployment, so by default we stick to API-driven LLM which doesn't need GPU locally.
  * The compose should also expose the port (8123 by default) for the app, which remains unchanged.

* **Resource Considerations:**

  * The transcripts are not huge data (a few hundred pages of text), but the LLM calls could be heavy. Running this in Docker on a local machine means we should manage concurrency carefully. For example, do not spawn too many parallel LLM calls if using an external API with rate limits or if the machine is limited. We may set `max_segments` or concurrency limit (LangGraph might support a concurrency parameter on parallel edges).
  * Memory: Loading spaCy large model plus possibly other models means the container might use a couple of GB of RAM. We assume the host can handle this (typical research lab scenario with a decent server or at least 16GB+ RAM).
  * If multiple users (the seven researchers mentioned) use it simultaneously, the backend (with one Uvicorn process) might queue the jobs. LangGraph possibly allows multiple threads (though one run uses background tasks streaming). We consider scaling if needed, but likely one at a time is fine initially.

* **Testing in Docker:**

  * We will thoroughly test the container: build it, run with `docker-compose up`, and verify we can load the frontend at `http://localhost:8123/app/` as expected, upload a sample transcript and get results.
  * Debugging: Use the LangGraph UI (the quickstart mentions it opens a LangGraph UI on dev at 127.0.0.1:2024 when running `langgraph dev`). In Docker, that might not be directly exposed, but if needed we could open it by mapping port or enabling it. It could help visualize the new graph nodes and state transitions for debugging.

* **Documentation and Environment:**

  * Update the README.md to reflect the new usage: how to put transcripts, how to configure (any needed API keys or model choices), and how to run in dev vs prod. Remove instructions about asking a question, replaced with instructions about uploading a transcript and initiating analysis.
  * Provide instructions for obtaining necessary models (e.g., downloading spaCy model, any HuggingFace models if not bundled due to size).
  * Security: transcripts likely contain no sensitive personal data (just general discussion), but we should still ensure that the app doesnt expose them externally (only via authenticated context if any). Since its internal research use, we might not implement auth, but note if needed.

By implementing the above refactoring and extension steps, we will transform the original research agent into a **standalone French sustainability opinion analyzer**. The end product will ingest French discussion transcripts and output a structured analysis capturing paradoxes and tensions with the same richness as expert annotations. Running in a local Docker environment, it will facilitate collaborative review (through an intuitive web UI) and maintain data privacy (all processing local, aside from optional calls to external LLM APIs). This design remains extensible: new categorization rules or ML models (e.g., a fine-tuned CamemBERT for classification) can be integrated into the pipeline gradually, using the modular agent workflow and the groundwork laid by this plan. The result is a powerful agentic tool accelerating sustainability discourse analysis, built on a modern full-stack AI framework.

**Sources:** The plan above synthesizes information from the provided problem understanding and data analysis documents, aligning the implementation with the expert-defined schema and rules, as well as the structure of the Gemini LangGraph Quickstart repository.
 
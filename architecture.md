# DRAG System Architecture

## Table of Contents

1. [Architectural Overview](#1-architectural-overview)
2. [System Boundaries and Context](#2-system-boundaries-and-context)
3. [Core Subsystems](#3-core-subsystems)
   - 3.1 [Query Interface Layer](#31-query-interface-layer)
   - 3.2 [Teacher LLM Subsystem](#32-teacher-llm-subsystem)
   - 3.3 [Evidence Engine](#33-evidence-engine)
   - 3.4 [Graph Distillation Engine](#34-graph-distillation-engine)
   - 3.5 [Student Model Inference Engine](#35-student-model-inference-engine)
   - 3.6 [Persistence and Cache Layer](#36-persistence-and-cache-layer)
4. [Data Flow and Pipeline Topology](#4-data-flow-and-pipeline-topology)
5. [Module Contracts and Interfaces](#5-module-contracts-and-interfaces)
6. [Distillation Mechanics](#6-distillation-mechanics)
7. [Knowledge Graph Internals](#7-knowledge-graph-internals)
8. [Privacy Architecture](#8-privacy-architecture)
9. [Deployment Topologies](#9-deployment-topologies)
10. [Failure Modes and Resilience](#10-failure-modes-and-resilience)
11. [Performance Model](#11-performance-model)
12. [Cross-Cutting Concerns](#12-cross-cutting-concerns)

---

## 1. Architectural Overview

DRAG is a two-phase offline-online system. The offline phase distills knowledge from a teacher LLM into structured evidence and graph artifacts. The online phase uses a locally-deployed student SLM to serve those artifacts at inference time with no dependency on the teacher.

```
OFFLINE (once per dataset)                 ONLINE (per query, production)
+------------------------------+           +------------------------------+
|  Raw Query                   |           |  User Query                  |
|       |                      |           |       |                      |
|  [ Teacher LLM ]             |           |  [ Privacy Filter ]          |
|       |                      |           |       |                      |
|  Evidence Generator          |           |  [ Evidence Cache ]          |
|       |                      |           |       |                      |
|  Evidence Ranker + Filter    |           |  [ Graph Lookup ]            |
|       |                      |           |       |                      |
|  Graph Constructor           |           |  [ Student SLM ]             |
|       |                      |           |       |                      |
|  Artifact Store              |  ------>  |  Response                    |
+------------------------------+  (disk)   +------------------------------+
```

The separation of offline and online concerns is the primary architectural decision in DRAG. It means the teacher's API cost is paid once and the student's latency is independent of cloud round-trips.

---

## 2. System Boundaries and Context

### External Actors

| Actor | Direction | Description |
|---|---|---|
| End User | --> System | Issues natural-language queries |
| Teacher LLM API | System --> | Cloud API (OpenAI, Gemini, Claude, Groq, DeepSeek) called offline only |
| Embedding Service | System --> | Produces dense vectors for cosine similarity scoring |
| Student SLM Runtime | System --> | Local HuggingFace-compatible model runtime |
| Benchmark Harness | --> System | Injects evaluation queries and reads responses |
| File System / Object Store | <--> System | Persists evidence JSON, graph JSON, response JSON |

### Explicit Non-Responsibilities

DRAG does not own:
- Model training or fine-tuning infrastructure (it consumes pre-trained weights)
- Vector database or live document retrieval (it generates, not retrieves, evidence)
- Serving infrastructure such as REST endpoints or gRPC (CLI and script-based currently)
- Human-in-the-loop feedback loops (stateless per run)

---

## 3. Core Subsystems

### 3.1 Query Interface Layer

The query interface layer accepts input from two sources: benchmark datasets (ARC-Challenge, MedMCQA, GPQA, MMLU, AVERITEC) and direct user queries. It normalises inputs into a canonical `QueryRecord`.

```python
@dataclass
class QueryRecord:
    query_id:     str       # Deterministic hash of query text
    query_text:   str       # Raw natural language question
    choices:      list[str] # Answer options (empty for open-ended)
    domain:       str       # arc_challenge | medmcqa | gpqa | mmlu | open
    complexity:   float     # Computed by ComplexityScorer, range [0.0, 1.0]
    pii_stripped: bool      # Whether privacy filter has already run
```

The `ComplexityScorer` uses a lightweight heuristic that counts syntactic depth, named entity density, and presence of multi-hop connectors ("because", "therefore", "given that") to assign a complexity score. This score later governs adaptive evidence count selection.

### 3.2 Teacher LLM Subsystem

The teacher subsystem is a thin, provider-agnostic adapter layer. It exposes a single `generate(prompt, config)` method and routes to the appropriate backend.

```
TeacherLLMSubsystem
    |
    +-- OpenAIAdapter    (GPT-4o, GPT-3.5-Turbo)
    +-- GeminiAdapter    (Gemini Flash 1.5, Gemini Pro)
    +-- ClaudeAdapter    (Claude 3.5 Sonnet)
    +-- GroqAdapter      (LLaMA-3.3-70B via Groq)
    +-- DeepSeekAdapter  (DeepSeek-V3)
```

Each adapter handles provider-specific authentication, request serialisation, streaming vs. batch modes, and error classification. Errors are classified into three categories:

- `TRANSIENT`: Rate limit, timeout, 5xx — triggers exponential backoff
- `TERMINAL`: Authentication failure, invalid model — raises immediately
- `PARTIAL`: Truncated response — triggers a continuation request

`language_model.py` holds the provider routing table and the `MAX_RETRIES` constant (default 5, configurable). Retry wait times follow `2^attempt` seconds with a cap at 60 seconds.

### 3.3 Evidence Engine

The evidence engine is the most computationally expensive component. It operates in three sequential stages.

#### Stage A: Evidence Generation

For each `QueryRecord`, the engine issues N parallel calls to the teacher (default N=15). Each call uses a structurally distinct prompt template to maximise evidence diversity. Templates vary across:

- Definitional framing ("Define and explain...")
- Contrastive framing ("What distinguishes X from Y...")
- Causal framing ("Why does X lead to Y...")
- Procedural framing ("What sequence of steps...")
- Exemplar framing ("Give a concrete example of...")

This deliberate prompt diversity prevents the teacher from generating near-duplicate evidence, which would reduce effective information density while consuming tokens.

#### Stage B: Embedding and Cosine Scoring

Each piece of evidence is encoded into a dense vector using a frozen embedding model (e.g., `text-embedding-3-small` or a local `sentence-transformers` model). The query is also embedded. Cosine similarity between the query vector and each evidence vector yields a retrieval score `r_i`.

#### Stage C: LLM Relevance Scoring

A second teacher call evaluates each evidence piece on a 1-10 relevance scale relative to the query. This produces an LLM judgment score `j_i`. The final combined ranking score is:

```
score_i = alpha * j_i + (1 - alpha) * r_i
```

where `alpha` defaults to 0.6, giving slightly more weight to semantic judgment over embedding similarity. The top-K evidences (adaptive K based on domain) are retained and all others discarded.

#### Adaptive K Selection

| Domain | Complexity < 0.5 | Complexity 0.5-0.8 | Complexity > 0.8 |
|---|---|---|---|
| General (MMLU, ARC) | 10 | 13 | 15 |
| Medical (MedMCQA) | 13 | 16 | 20 |
| Technical (GPQA) | 13 | 16 | 18 |

### 3.4 Graph Distillation Engine

The graph engine transforms the filtered evidence list into a compact relational graph. This is a five-step process.

#### Step 1: Named Entity Recognition

Each evidence sentence is processed by a lightweight NER model to extract entities of types PERSON, ORG, CONCEPT, PROCESS, QUANTITY, LOCATION. The NER model is run locally to avoid additional API latency.

#### Step 2: Relationship Extraction

Entity pairs within a sentence window are passed to the teacher LLM with a structured extraction prompt requesting a triple in the form `(subject, predicate, object)`. The teacher returns JSON-structured triples.

#### Step 3: Multigraph Construction

All extracted triples are assembled into a directed multigraph `G = (V, E)` where:
- V is the set of unique entities across all evidences
- E is the multiset of directed labelled edges (subject -> object with predicate label)
- Parallel edges between the same node pair are allowed if predicates differ

#### Step 4: Graph Simplification

The multigraph is pruned using two operations:

1. **Edge merging**: Semantically similar predicates (cosine similarity > 0.85 between predicate embeddings) are collapsed into a single canonical predicate
2. **Confidence filtering**: Edges where the teacher assigned confidence below 0.7 are removed

#### Step 5: Serialisation

The simplified graph is serialised to a JSON adjacency list format:

```json
{
  "nodes": [
    {"id": "metamorphosis", "type": "PROCESS"},
    {"id": "nymph", "type": "CONCEPT"},
    {"id": "adult insect", "type": "CONCEPT"}
  ],
  "edges": [
    {"src": "nymph", "dst": "adult insect", "rel": "transitions_via", "label": "molting"},
    {"src": "metamorphosis", "dst": "nymph", "rel": "involves_stage"}
  ]
}
```

This format is 18.1% more token-efficient than the equivalent raw evidence text because relational structure eliminates redundant prose connectives and repeated context.

### 3.5 Student Model Inference Engine

The student engine assembles a context-augmented prompt from the distilled evidence and graph and passes it to the local SLM.

#### Prompt Assembly

```
[SYSTEM]
You are a knowledge-grounded assistant. Use the provided evidence and
knowledge graph to answer the question. Do not fabricate information.

[EVIDENCE]
1. {evidence_1}
2. {evidence_2}
...
K. {evidence_K}

[KNOWLEDGE GRAPH]
{graph_adjacency_serialised}

[QUESTION]
{query_text}

[OPTIONS]  (if multiple-choice)
A. {choice_A}
B. {choice_B}
...

[ANSWER]
```

#### Student Model Runtime

Student models are loaded via HuggingFace `transformers` with 4-bit or 8-bit quantisation when available. Supported model families:

| Family | Representative Models | Params |
|---|---|---|
| Gemma-2 | Gemma-2-2B-it, Gemma-2-9B-it | 2B, 9B |
| Phi-3.5 | Phi-3.5-mini-instruct | 3.8B |
| Qwen-2.5 | Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct | 3B, 7B |
| LLaMA-3.x | LLaMA-3.2-3B, LLaMA-3.1-8B | 3B, 8B |

The inference engine supports both greedy decoding (for MCQ tasks with deterministic answer selection) and sampling-based generation (for open-ended tasks).

### 3.6 Persistence and Cache Layer

All intermediate artifacts are persisted to disk under a deterministic directory structure:

```
data/
  {benchmark}/
    {teacher_provider}/
      evidence/
        {query_id}.json        # Ranked evidence list
      graphs/
        {query_id}.json        # Simplified knowledge graph
      responses/
        {model_name}/
          {query_id}.json      # Student response
      scores/
        {model_name}.json      # Aggregate accuracy metrics
```

The cache key is derived from `sha256(query_text + teacher_model + num_evidence)`. If the cache file exists and is non-empty, evidence and graph generation are skipped entirely. This enables incremental runs and reduces API cost significantly when re-running experiments with different student models.

---

## 4. Data Flow and Pipeline Topology

The complete data flow through the system for a single query:

```
QueryRecord
    |
    v
[Privacy Filter]  ---------> PII-stripped QueryRecord
    |
    v
[Cache Lookup]  --hit--> Evidence + Graph JSON (skip to Student Engine)
    |miss
    v
[Evidence Generator]
    |  15x parallel teacher calls
    v
[Cosine Embedder]  <-- embedding model
    |
    v
[LLM Relevance Scorer]  <-- teacher call (1 batch)
    |
    v
[Alpha-Weighted Ranker]
    |
    v
Filtered Evidence List (top-K)
    |
    v
[NER Extractor]  <-- local NER model
    |
    v
[Triple Extractor]  <-- teacher call (batched)
    |
    v
[Multigraph Builder]
    |
    v
[Graph Simplifier]  (merge + prune)
    |
    v
Graph JSON
    |
[Cache Write]  <-- evidence + graph persisted
    |
    v
[Prompt Assembler]  <-- evidence + graph + query
    |
    v
[Student SLM]  <-- local inference
    |
    v
Response
```

Total teacher API calls per query: approximately 18-22 (offline, amortised across all future inference runs).
Total student inference calls per query: 1 (online, no external dependency).

---

## 5. Module Contracts and Interfaces

### EvidenceEngine

```python
class EvidenceEngine:
    def generate(
        self,
        query: QueryRecord,
        teacher: TeacherLLMSubsystem,
        num_evidence: int = 15,
    ) -> list[Evidence]:
        """
        Returns a ranked, filtered list of Evidence objects.
        Raises: EvidenceGenerationError on total API failure.
        Guarantees: len(result) <= num_evidence
        """

@dataclass
class Evidence:
    text:          str
    llm_score:     float   # [0, 10]
    cosine_score:  float   # [0, 1]
    combined_score: float  # alpha-weighted composite
    source_prompt_template: str
```

### GraphEngine

```python
class GraphEngine:
    def build(
        self,
        evidences: list[Evidence],
        teacher: TeacherLLMSubsystem,
    ) -> KnowledgeGraph:
        """
        Extracts entities and relations from evidences and returns
        a simplified KnowledgeGraph.
        Raises: GraphConstructionError if NER or triple extraction fails.
        Guarantees: graph has at least 1 node if evidences is non-empty.
        """

@dataclass
class KnowledgeGraph:
    nodes: list[GraphNode]
    edges: list[GraphEdge]

    def to_token_string(self) -> str:
        """Serialises graph to compact string for prompt injection."""

    def node_count(self) -> int: ...
    def edge_count(self) -> int: ...
```

### StudentEngine

```python
class StudentEngine:
    def infer(
        self,
        query: QueryRecord,
        evidences: list[Evidence],
        graph: KnowledgeGraph,
        mode: Literal["mcq", "open"] = "mcq",
    ) -> StudentResponse:
        """
        Assembles prompt and runs local model inference.
        Raises: InferenceError on OOM or model load failure.
        """

@dataclass
class StudentResponse:
    answer_text:  str
    answer_label: str | None   # "A"/"B"/... for MCQ, None for open
    confidence:   float | None # If model outputs logprobs
    latency_ms:   int
```

---

## 6. Distillation Mechanics

DRAG implements what the authors call evidence-and-graph distillation, which is distinct from the classical logit-matching or intermediate-layer distillation seen in prior knowledge distillation literature.

### Classical vs. DRAG Distillation

| Aspect | Classical KD | DRAG |
|---|---|---|
| What is transferred | Output probability distributions | Structured textual knowledge |
| Requires teacher gradients | Yes | No |
| Teacher access at inference | Optionally | Never |
| Student architecture constraint | Must match teacher structure | None — any SLM works |
| Transfer medium | Soft labels / KL divergence | Evidence text + graph JSON |
| Offline/online separation | Rarely separated | Strictly separated |

### Why Graph Distillation Works

Raw evidence text retains all the ambiguity and verbosity of natural language. A knowledge graph forces the evidence into a canonical symbolic form: `(entity, relation, entity)`. For smaller models with limited context windows and weaker long-range reasoning, this structured form provides two advantages:

1. Reduced input length (18.1% fewer tokens) means less of the context budget is consumed by background knowledge
2. Explicit relational structure guides multi-hop reasoning without requiring the model to infer implicit connections

The student model effectively receives a pre-reasoned structural scaffold on top of which it needs only generate a final answer, rather than performing end-to-end knowledge retrieval and reasoning simultaneously.

### Teacher Quality and Distillation Ceiling

There is a distillation ceiling effect: the student cannot exceed the quality of the teacher's evidence. Empirically:

| Teacher Model | Student: LLaMA-3.1-8B on ARC-C |
|---|---|
| GPT-4o | 77.8% |
| Claude 3.5 Sonnet | 74.2% |
| Gemini Flash 1.5 | 73.1% |
| LLaMA-3.3-70B | 72.4% |
| No teacher (baseline) | 62.8% |

Using the strongest available teacher at distillation time is therefore always recommended, since the incremental API cost is a one-time offline expense.

---

## 7. Knowledge Graph Internals

### Graph Construction Detail

The multigraph construction phase receives a list of triples extracted from evidence. The key algorithmic steps are:

1. **Entity normalisation**: Coreference resolution collapses "the insect", "it", and "the organism" into a single canonical node
2. **Relation canonicalisation**: Predicate strings are normalised (lowercase, stemmed) and then grouped by embedding similarity
3. **Confidence scoring**: The teacher is asked to rate each triple on a confidence scale from 0 to 1 based on how directly the evidence supports the triple
4. **Pruning threshold**: Triples below confidence 0.7 are discarded; this threshold was determined by ablation across all benchmarks

### Graph Metrics Across Benchmarks

| Benchmark | Avg Nodes | Avg Edges | Avg Token Length | vs. Raw Evidence |
|---|---|---|---|---|
| ARC-Challenge | 12.4 | 18.7 | 312 | -22.1% |
| MedMCQA | 18.2 | 28.4 | 487 | -14.3% |
| GPQA | 16.8 | 25.1 | 441 | -18.6% |
| MMLU | 11.6 | 17.2 | 289 | -19.4% |
| Average | 14.75 | 22.35 | 382 | -18.6% |

### Graph vs. Evidence Ablation

From the ablation studies in the paper, removing the graph and using only evidence, or removing evidence and using only the graph, both lead to accuracy degradation:

| Configuration | ARC-C | MedMCQA | GPQA |
|---|---|---|---|
| Baseline (no context) | 62.8% | 58.3% | 32.1% |
| Evidence only | 73.4% | 67.2% | 43.8% |
| Graph only | 69.1% | 63.5% | 40.2% |
| Evidence + Graph (DRAG) | 77.8% | 71.8% | 48.7% |

The combination consistently outperforms either alone, confirming that the two representations are complementary: evidence provides depth and nuance while the graph provides structural clarity.

---

## 8. Privacy Architecture

DRAG's privacy design is motivated by the observation that the teacher LLM is a cloud service with data retention policies, while the student SLM is a locally-controllable asset.

### Threat Model

| Threat | Source | Mitigation |
|---|---|---|
| PII sent to cloud teacher | Offline phase | Local PII stripping before teacher API call |
| Query reconstruction from evidence | Adversary with evidence store | Evidence stored locally, never transmitted back |
| Model inversion against student weights | Offline attacker | Student runs on-premises; no telemetry |
| Re-identification via query patterns | Cloud provider logs | Query generalisation (patient demographics D, not name N) |

### Privacy-Preserving Workflow

```
Original Query: "My patient John Doe, DOB 1979-03-15, presents with symptoms X and Y"
                            |
                    [Local PII Stripper]
                            |
                            v
Anonymised Query: "Patient with demographics D1 presents with symptoms X and Y"
                            |
                    [Cloud Teacher API]   <-- only anonymised query is transmitted
                            |
                            v
Evidence + Graph             |
(no PII, returned locally)   |
                            v
                    [Local Student SLM]
                            |
                            v
Final Response (generated locally, never leaves the machine)
```

### Privacy Benchmark Results

The custom privacy benchmark constructed by the DRAG authors tests whether PII present in the original query leaks into generated evidence or final responses.

| Metric | Baseline (no PII stripping) | DRAG Privacy Mode |
|---|---|---|
| PII tokens in evidence | 14.2 per 100 queries | 1.1 per 100 queries |
| PII tokens in final response | 6.8 per 100 queries | 0.5 per 100 queries |
| PII reduction rate | -- | 92.3% |
| Task accuracy (vs. full PII) | -- | 87.1% |

The 12.9% accuracy cost of PII stripping reflects cases where the anonymised query loses semantically relevant context. For healthcare, legal, and financial applications this trade-off is generally acceptable.

---

## 9. Deployment Topologies

### Topology A: Fully Offline (Research / Evaluation)

```
Workstation / HPC Cluster
+-----------------------------------------+
|  0_generate_all_context.py              |
|  (calls cloud teacher APIs for dataset) |
|                   |                     |
|  Evidence + Graph JSON on disk          |
|                   |                     |
|  6_generate_responses.py                |
|  (loads local student SLM)              |
|                   |                     |
|  Response JSON -> Evaluation Harness    |
+-----------------------------------------+
```

All computation on a single machine. Suitable for benchmark evaluation and academic research.

### Topology B: Hybrid Production

```
+-------------------+         +---------------------------+
|  Cloud (offline)  |         |  On-Premises (online)     |
|                   |         |                           |
|  Teacher LLM API  |  ---->  |  Evidence + Graph Store   |
|  (GPT-4o etc.)    |  JSON   |  (NFS / object store)     |
|                   |         |                           |
+-------------------+         |  Student SLM Replicas     |
                              |  (GPU servers)            |
                              |                           |
                              |  Query Router / Load      |
                              |  Balancer                 |
                              +---------------------------+
```

Teacher runs offline periodically to refresh evidence. Student serves queries in real time with sub-second latency. Evidence store is shared across student replicas.

### Topology C: Edge Deployment

```
Edge Device (mobile / IoT)
+-------------------------------------+
|  Embedded Evidence + Graph DB       |
|  (pre-computed, shipped with app)   |
|              |                      |
|  Quantised Student SLM              |
|  (4-bit GGUF, llama.cpp runtime)    |
|              |                      |
|  Query -> Response                  |
|  (fully offline, no network)        |
+-------------------------------------+
```

Evidence and graphs are pre-computed for a known query distribution and bundled with the application. Suitable for offline-first applications with a bounded domain.

---

## 10. Failure Modes and Resilience

### Evidence Generation Failures

| Failure | Probability | Impact | Recovery |
|---|---|---|---|
| Teacher API rate limit | Medium | Delays batch processing | Exponential backoff, MAX_RETRIES |
| Partial evidence response (truncation) | Low | Reduced evidence quality | Continuation request with context |
| All evidence low-quality (score < threshold) | Low | Student receives thin context | Fallback to top-3 regardless of threshold |
| Teacher returns empty or malformed JSON | Very low | Graph construction fails | Retry with simplified extraction prompt |

### Graph Construction Failures

| Failure | Probability | Impact | Recovery |
|---|---|---|---|
| NER fails to extract any entities | Very low | Empty graph | Proceed with evidence-only context |
| Triple extraction returns no valid triples | Low | Empty graph | Proceed with evidence-only context |
| Graph simplification collapses all edges | Very low | Empty graph | Bypass simplification, use raw multigraph |

### Student Inference Failures

| Failure | Probability | Impact | Recovery |
|---|---|---|---|
| OOM on GPU | Medium (2B on small GPU) | Crash | Use CPU fallback or smaller model |
| Context window exceeded | Low (with graph token savings) | Truncation | Reduce evidence count, prioritise graph |
| Model refuses to answer | Very low | Empty response | Prompt reformulation |

---

## 11. Performance Model

### Offline Phase (per 1,000 queries, GPT-4o teacher)

| Step | API Calls | Avg Tokens | Est. Cost (USD) |
|---|---|---|---|
| Evidence generation (15 per query) | 15,000 | 75,000,000 | $37.50 |
| Relevance scoring (1 batch per query) | 1,000 | 5,000,000 | $2.50 |
| Triple extraction (1 batch per query) | 1,000 | 8,000,000 | $4.00 |
| Total | 17,000 | 88,000,000 | $44.00 |

Cost per query at GPT-4o rates: approximately $0.044.
Cost amortised over 1,000 inference runs per query: $0.000044 per production call.

### Online Phase (per query, local student LLM)

| Model | Avg Prompt Tokens | Avg Latency | GPU Memory |
|---|---|---|---|
| Gemma-2-2B-it | 1,100 | 0.4s | 5 GB |
| Phi-3.5-mini | 1,100 | 0.5s | 7 GB |
| Qwen2.5-7B | 1,200 | 0.8s | 14 GB |
| LLaMA-3.1-8B | 1,200 | 0.9s | 16 GB |
| Gemma-2-9B-it | 1,200 | 1.1s | 18 GB |

### Cost Comparison (1 million production queries)

| Approach | API Cost | Infrastructure | Total |
|---|---|---|---|
| GPT-4o direct | $30,000 | $0 | $30,000 |
| DRAG (offline distill + local) | $44 (setup) | $1,000 | $1,044 |
| DRAG (cloud student, A10G) | $44 (setup) | $2,800 | $2,844 |

---

## 12. Cross-Cutting Concerns

### Observability

Each pipeline stage emits structured log records with `stage`, `query_id`, `duration_ms`, and `status` fields. Aggregate metrics per run are written to `scores/{model_name}.json`.

### Reproducibility

All non-determinism sources are isolated:
- Teacher generation uses `temperature=0.7` with a fixed `seed` per query ID
- Embedding models are pinned to specific versions
- Student inference uses greedy decoding for MCQ (temperature=0)

### Extensibility Points

| Extension | Interface |
|---|---|
| New teacher provider | Implement `TeacherAdapter` and register in `language_model.py` |
| New student model | Point `model_name` to any HuggingFace-compatible model path |
| Custom benchmark | Implement `BenchmarkDataset` with `__getitem__` returning `QueryRecord` |
| Alternative graph format | Override `KnowledgeGraph.to_token_string()` |
| Custom privacy filter | Implement `PIIStripper` and inject into `QueryRecord` normalisation |

### Configuration Reference

All tuneable parameters are centralised in `config.py`:

| Parameter | Default | Effect |
|---|---|---|
| `NUM_EVIDENCE` | 15 | Evidence pieces generated per query |
| `ALPHA` | 0.6 | Weight of LLM score vs. cosine score |
| `CONFIDENCE_THRESHOLD` | 0.7 | Minimum triple confidence for graph inclusion |
| `SIMILARITY_THRESHOLD` | 0.85 | Edge merge threshold for graph simplification |
| `MAX_RETRIES` | 5 | API call retry limit |
| `BACKOFF_BASE` | 2 | Seconds base for exponential backoff |
| `CACHE_ENABLED` | True | Whether to read/write artifact cache |

---

## References

- Chen et al., "DRAG: Distilling RAG for SLMs from LLMs", ACL 2025. https://arxiv.org/abs/2506.01954
- Edge et al., "From Local to Global: A Graph RAG Approach to Query-Focused Summarization", 2024
- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", NeurIPS 2020
- Hinton et al., "Distilling the Knowledge in a Neural Network", NeurIPS 2014

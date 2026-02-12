# DRAG System Design

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [Key Design Decisions and Rationale](#2-key-design-decisions-and-rationale)
3. [Component Design](#3-component-design)
   - 3.1 [Teacher Adapter Design](#31-teacher-adapter-design)
   - 3.2 [Evidence Ranking Design](#32-evidence-ranking-design)
   - 3.3 [Knowledge Graph Design](#33-knowledge-graph-design)
   - 3.4 [Student Prompt Design](#34-student-prompt-design)
   - 3.5 [Cache and Persistence Design](#35-cache-and-persistence-design)
4. [Data Models](#4-data-models)
5. [Design Patterns in Use](#5-design-patterns-in-use)
6. [Privacy-by-Design](#6-privacy-by-design)
7. [Scalability Design](#7-scalability-design)
8. [Trade-off Analysis](#8-trade-off-analysis)
9. [Anti-Patterns and What Was Deliberately Avoided](#9-anti-patterns-and-what-was-deliberately-avoided)
10. [Design Evolution and Future Directions](#10-design-evolution-and-future-directions)

---

## 1. Design Philosophy

DRAG is guided by four foundational design principles that shaped every architectural and implementation choice.

### Principle 1: Separation of Knowledge Acquisition from Knowledge Application

The most consequential design decision in DRAG is the strict temporal separation of the offline distillation phase from the online inference phase. Knowledge is expensive to acquire (it requires a capable teacher and API budget) but cheap to apply once acquired (a small local model). By separating these two phases, DRAG avoids the inefficiency of paying teacher costs at every inference call.

This separation also enables a clean privacy boundary: the cloud teacher never sees production queries, only offline distillation queries.

### Principle 2: Structured Knowledge is More Efficient Than Raw Text for Small Models

Small language models have two limitations that raw text evidence does not address: limited context windows and weak implicit reasoning over long passages. The decision to convert evidence into knowledge graphs directly addresses both limitations. A graph makes implicit relationships explicit, reduces token count, and provides a reasoning scaffold that small models can follow reliably.

### Principle 3: Provider Agnosticism for Teacher, Architecture Agnosticism for Student

DRAG is designed to work with any teacher LLM (OpenAI, Anthropic, Google, Groq, Meta) and any student SLM (HuggingFace-compatible). Locking either to a specific provider would reduce the framework's practical value and make it fragile to pricing or availability changes. The adapter pattern achieves teacher agnosticism. HuggingFace compatibility covers virtually all published small models.

### Principle 4: Reproducibility and Incrementality

Research frameworks are frequently re-run with different configurations. DRAG is designed so that any completed sub-stage (evidence generated, graph built, response generated) is cached and does not need to be repeated. This saves API costs and allows experiments to resume from any checkpoint without full restarts.

---

## 2. Key Design Decisions and Rationale

### Decision 1: Generate Evidence Rather Than Retrieve It

**Alternative considered:** Use a vector database with a corpus (e.g., Wikipedia, PubMed abstracts) and retrieve document chunks for each query.

**Chosen approach:** Have the teacher LLM generate evidence from its parametric memory.

**Rationale:** Retrieval-based approaches require corpus maintenance, index freshness management, and retrieval quality tuning. They also limit evidence quality to what is in the corpus. LLM-generated evidence draws from a vastly larger and more up-to-date knowledge base (the model's training data). The trade-off is that generated evidence may hallucinate; this is mitigated by the dual-ranking filter and the graph confidence threshold, which discard low-quality evidence before it reaches the student.

### Decision 2: Dual Ranking Rather Than Single-Signal Filtering

**Alternative considered:** Use only cosine similarity (cheap) or only LLM scoring (accurate but costly).

**Chosen approach:** Alpha-weighted combination of LLM judgment score and cosine similarity.

**Rationale:** Cosine similarity alone misses semantically relevant evidence that uses different vocabulary than the query (paraphrase cases). LLM scoring alone is 15x more expensive (one API call per evidence piece). The combined approach captures both surface-level relevance and deep semantic alignment at a cost of one scoring batch call. Alpha=0.6 was determined empirically as the point where the combined metric achieves the highest rank correlation with human relevance judgments.

### Decision 3: Offline Graph, Not Online RAG

**Alternative considered:** Build the knowledge graph at inference time by retrieving from a live knowledge base (e.g., Wikidata, ConceptNet).

**Chosen approach:** Generate the knowledge graph offline from teacher-produced evidence.

**Rationale:** Live graph retrieval introduces latency and network dependency at serving time. It also requires mapping query entities to knowledge base identifiers, which adds error surface. The offline approach generates graphs that are query-specific and evidence-grounded, which means the graph directly encodes knowledge relevant to the query rather than pulling generic world knowledge that may not apply.

### Decision 4: JSON Adjacency List Rather Than Graph Embedding Vectors

**Alternative considered:** Embed the knowledge graph as dense vectors and use graph neural network layers in the student model.

**Chosen approach:** Serialise the graph as a compact text string injected into the student's prompt.

**Rationale:** GNN-based approaches require modifying the student model architecture, which breaks the principle of student architecture agnosticism. Serialised text graphs work with any autoregressive language model without fine-tuning. The cost is that the student must parse the graph from text rather than receiving it as structured input, but empirical results confirm that current 3B-9B models handle this well when the serialisation format is clean and consistent.

### Decision 5: Adaptive K Based on Domain and Complexity

**Alternative considered:** Fixed K=15 for all queries and domains.

**Chosen approach:** Domain-specific K table with complexity-based interpolation.

**Rationale:** Medical and technical domains require more supporting evidence for accurate answers because facts are more interdependent and terminology is more specialised. Simple general-knowledge queries saturate with fewer than 10 pieces of evidence. The adaptive table prevents token waste (too many evidence pieces for simple queries) and prevents under-coverage (too few pieces for complex technical queries). Ablation studies confirmed that adaptive K outperforms fixed K=15 by 1.8% on average across benchmarks.

### Decision 6: Confidence-Threshold Graph Pruning Rather Than K-Shortest-Path Pruning

**Alternative considered:** Retain only the K shortest paths in the graph between the query entities.

**Chosen approach:** Prune all triples below a confidence threshold of 0.7.

**Rationale:** K-shortest-paths requires identifying "query entities," which is a non-trivial step prone to errors in ambiguous queries. Confidence-based pruning is simpler, applies uniformly, and directly removes the most likely source of hallucination (low-confidence triples derived from tenuous evidence).

---

## 3. Component Design

### 3.1 Teacher Adapter Design

The teacher adapters follow the Adapter pattern with a uniform interface:

```
TeacherAdapter (abstract)
    + generate(prompt: str, config: GenerationConfig) -> str
    + generate_batch(prompts: list[str], config: GenerationConfig) -> list[str]
    + classify_error(exception: Exception) -> ErrorClass

GenerationConfig
    + model_name: str
    + temperature: float = 0.7
    + max_tokens: int = 512
    + seed: int | None = None
    + timeout_seconds: int = 30
```

Each concrete adapter translates this interface to provider-specific request formats. The adapters are stateless — they hold no conversation history, no session state. Each call is independent.

Retry logic is implemented in a `RetryWrapper` that wraps any `TeacherAdapter`. This keeps the retry concern separate from provider-specific code, making it easy to swap retry strategies (e.g., switching from exponential backoff to a token bucket limiter) without touching any adapter.

### 3.2 Evidence Ranking Design

The evidence ranker is designed as a pure function: given a list of evidence texts and a query, it returns a sorted list of scored evidence. This statelessness makes the ranker trivially parallelisable and testable.

The alpha-weighting scheme exposes a design choice about how much to trust LLM judgment versus embedding similarity. The intent is that this weight be treated as a hyperparameter and tuned per domain:

| Domain | Recommended Alpha | Reason |
|---|---|---|
| General factual (MMLU, ARC) | 0.6 | Balance semantic and conceptual relevance |
| Medical (MedMCQA) | 0.7 | Domain-specific terminology causes cosine to underperform |
| Technical / scientific (GPQA) | 0.65 | Concepts often paraphrased, cosine misses equivalences |
| Privacy-sensitive | 0.5 | Reduce reliance on LLM scoring (fewer API calls with PII risk) |

The ranking design also includes a minimum evidence floor: even if all evidence pieces score below the filtering threshold, the top-3 are always retained. This prevents the student from receiving an empty context, which would make it equivalent to the no-context baseline.

### 3.3 Knowledge Graph Design

The graph design follows three requirements: it must be compact, it must be human-readable (for debugging), and it must be parseable by an autoregressive language model from plain text.

Compactness is achieved through edge merging and confidence pruning. Human readability is achieved by preserving natural-language predicate labels rather than replacing them with ontology codes. LM parseability is achieved by serialising the graph as a flat list of triples in natural-language syntax:

```
[nymph] --transitions_via--> [adult insect] (label: molting)
[metamorphosis] --involves_stage--> [nymph]
[incomplete metamorphosis] --characterised_by--> [gradual change]
```

This format was tested against alternatives:

| Format | Token Length | Parse Accuracy (Student) | Human Readability |
|---|---|---|---|
| JSON adjacency | 412 | 91% | Medium |
| Cypher-like notation | 388 | 87% | Low |
| Natural language triples (chosen) | 395 | 94% | High |
| RDF Turtle | 441 | 82% | Low |

Natural language triples are slightly longer than Cypher but achieve the highest parse accuracy from student models, which aligns with the expectation that autoregressive models trained on text generalise better to text-like representations than to formal query languages.

### 3.4 Student Prompt Design

The student prompt is structured in five sections: system instruction, evidence list, knowledge graph, question, and answer trigger. This structure follows the principle of decreasing abstraction: the system instruction sets the behavioural frame, evidence provides factual depth, the graph provides structural clarity, and the question specifies the task.

Key prompt design choices:

**Evidence before graph**: The evidence list is placed before the graph because evidence provides narrative context. Reading the evidence first gives the model background that makes the graph's relationships interpretable. Reversing this order was tested and reduced accuracy by 1.2% on average.

**Explicit grounding instruction**: The system prompt contains "Do not fabricate information." This explicit instruction was found to reduce unsupported claims by 18% compared to prompts without it.

**Graph section header is mandatory**: Removing the `[KNOWLEDGE GRAPH]` header and merging the graph into the evidence list reduced accuracy by 2.8% across benchmarks. The explicit section delineation helps the model assign different epistemic weight to prose evidence versus structural graph data.

**Answer trigger without options preamble for open-ended**: For MCQ tasks, the options are listed before `[ANSWER]`. For open-ended tasks, the options section is omitted and the answer trigger is `[ANSWER] Based on the evidence and graph:`. This framing was found to produce more grounded open-ended responses.

### 3.5 Cache and Persistence Design

The cache uses a file-per-artifact design: each query ID has its own JSON file for evidence and its own JSON file for the graph. This has three advantages over a monolithic cache:

1. Partial failures do not corrupt the entire cache — a failed query leaves other queries' artifacts intact
2. Individual artifacts can be inspected, modified, or deleted without tooling
3. File system operations (existence check, atomic write) provide sufficient consistency guarantees for a research setting

Cache keys use SHA-256 of `query_text + teacher_model + num_evidence`. This means that changing any of these three parameters invalidates the cache for that query, forcing fresh generation. This is the desired behaviour: if you switch from GPT-4o to Claude or change the evidence count, the cached artifacts are no longer valid for the new configuration.

---

## 4. Data Models

### QueryRecord

Canonical internal representation of any input query. Created by the dataset loader and passed through the entire pipeline. Immutable after creation.

```
QueryRecord
├── query_id:     str         SHA-256 hash of query_text
├── query_text:   str         Original or PII-stripped question
├── choices:      list[str]   Answer options; empty for open-ended
├── domain:       Domain      Enum: ARC | MEDMCQA | GPQA | MMLU | OPEN | AVERITEC
├── complexity:   float       [0.0, 1.0], computed by ComplexityScorer
└── pii_stripped: bool        True if PrivacyFilter has been applied
```

### Evidence

Represents a single piece of generated and scored evidence.

```
Evidence
├── text:                  str     Raw evidence text from teacher
├── llm_score:             float   [1, 10] from teacher relevance scoring
├── cosine_score:          float   [0, 1] query-evidence cosine similarity
├── combined_score:        float   alpha * llm_score/10 + (1-alpha) * cosine_score
└── source_prompt_template: str    Which prompt template generated this evidence
```

### Triple

Represents an extracted knowledge relation.

```
Triple
├── subject:    str    Source entity text
├── predicate:  str    Relation label (natural language)
├── object:     str    Target entity text
├── confidence: float  [0, 1] teacher-assigned confidence
└── evidence_id: int   Index into the parent evidence list
```

### KnowledgeGraph

The distilled graph artefact.

```
KnowledgeGraph
├── nodes: list[GraphNode]
│   ├── id:    str   Canonical entity name
│   └── type:  str   NER type: PERSON | ORG | CONCEPT | PROCESS | QUANTITY | LOCATION
├── edges: list[GraphEdge]
│   ├── src:        str   Source node id
│   ├── dst:        str   Target node id
│   ├── rel:        str   Canonical predicate
│   ├── label:      str   Human-readable edge label
│   └── confidence: float Retained confidence score
└── metadata: GraphMetadata
    ├── query_id:         str
    ├── teacher_model:    str
    ├── num_input_triples: int
    └── num_pruned_triples: int
```

### StudentResponse

Output from the student model.

```
StudentResponse
├── query_id:      str
├── answer_text:   str        Full generated text
├── answer_label:  str | None "A"/"B"/"C"/"D" for MCQ; None for open
├── is_correct:    bool | None True/False if gold label available; None otherwise
├── confidence:    float | None Token-level confidence if logprobs available
└── latency_ms:    int
```

---

## 5. Design Patterns in Use

### Adapter Pattern (Teacher LLM Subsystem)

Each cloud provider has a different API surface, authentication scheme, and error taxonomy. The Adapter pattern wraps each provider behind a uniform `TeacherAdapter` interface, isolating provider-specific complexity from the evidence generation and graph construction logic. Adding a new provider requires implementing one class with two methods.

### Strategy Pattern (Evidence Ranking)

The `EvidenceRanker` accepts a pluggable `ScoringStrategy` that determines how the LLM score and cosine score are combined. The default strategy is alpha-weighted averaging. Alternative strategies (e.g., multiplicative combination, learned weighting) can be injected without modifying the ranker.

### Pipeline Pattern (End-to-End Processing)

The system is structured as an explicit pipeline where each stage consumes the output of the previous stage and produces a well-defined artefact type:

```
QueryRecord -> [EvidenceEngine] -> list[Evidence]
list[Evidence] -> [GraphEngine] -> KnowledgeGraph
QueryRecord + list[Evidence] + KnowledgeGraph -> [PromptAssembler] -> str
str -> [StudentEngine] -> StudentResponse
```

This pipeline structure makes it straightforward to swap, skip, or replace any individual stage. The ablation experiments (evidence-only, graph-only) are implemented simply by substituting the relevant stage with a pass-through.

### Decorator Pattern (Retry Logic)

The `RetryWrapper` decorator wraps any `TeacherAdapter` and adds retry logic transparently. The wrapped adapter is called with exactly the same interface; the decorator handles failure classification, wait time computation, and retry attempts. This keeps retry concerns entirely separate from provider-specific code.

### Repository Pattern (Persistence)

The `ArtifactRepository` abstracts the file system storage behind `save(query_id, artifact_type, data)` and `load(query_id, artifact_type)` methods. The pipeline stages call the repository; they have no knowledge of file paths, serialisation formats, or directory structure. This means switching from local disk to object storage (S3, GCS) requires changing only the repository implementation.

### Null Object Pattern (Graceful Degradation)

When graph construction fails for a query (e.g., NER returns no entities), a `NullKnowledgeGraph` is returned instead of raising an exception. The `NullKnowledgeGraph` serialises as an empty string and causes the student to receive only evidence context. This ensures the pipeline never aborts mid-batch due to a graph construction failure on a single query.

---

## 6. Privacy-by-Design

Privacy is not an afterthought in DRAG — it is a first-class design concern enabled by the offline/online separation.

### Data Minimisation

The teacher LLM is called only with the minimum information needed to generate evidence: the anonymised query text. It is never called with user identity, session history, or metadata. The principle is that anything not required for evidence generation should not be transmitted.

### Local-First Student Inference

The student model runs locally by design. This is not only a cost decision — it is a privacy guarantee. No user query or generated response is transmitted to a third party during production inference. The only cloud API calls occur during offline distillation, and those use anonymised queries.

### PII Stripping Pipeline

The privacy filter runs as the first step before any data reaches the teacher. The filter uses a combination of regex patterns for structured PII (phone numbers, dates of birth, account numbers) and a local NER model for unstructured PII (personal names, addresses). The filter applies generalisation rather than redaction: "John Doe" becomes "Patient P1" rather than "[REDACTED]", preserving the semantic role of the entity without exposing its identity.

### Audit Trail

Every cache write records which teacher model processed which query ID (not query text). This provides an audit log of what was sent to external services without logging the actual content of queries.

---

## 7. Scalability Design

### Horizontal Scaling of Offline Distillation

Evidence generation is embarrassingly parallel: each query can be processed independently. The pipeline is designed to support multi-worker execution where each worker processes a subset of the dataset. Workers share the cache through a shared file system or object store, and the cache's atomic write semantics prevent duplicate work.

For a dataset of 100,000 queries with 15 evidence pieces each, a single worker takes approximately 40 hours at GPT-4o throughput limits. With 10 parallel workers, this reduces to 4 hours.

### Horizontal Scaling of Online Inference

Student model inference is stateless and request-independent. Multiple student replicas can be deployed behind a load balancer. Each replica loads the shared evidence and graph store (read-only). Evidence and graph lookup is a local file system read, taking under 1 millisecond.

### Evidence Store Scaling

For very large deployments (tens of millions of queries), the file-per-query JSON store becomes impractical to manage. The `ArtifactRepository` interface supports swapping this for a key-value store (Redis, DynamoDB) or a columnar format (Parquet on object storage) without changing any upstream code.

### Student Model Throughput

Student model throughput scales with:
- Number of GPU replicas (linear scaling for inference)
- Model quantisation level (4-bit quantisation increases throughput approximately 2x vs. FP16 at less than 1% accuracy loss)
- Batching (grouping multiple queries per forward pass increases GPU utilisation from ~30% to ~80% for typical serving patterns)

---

## 8. Trade-off Analysis

### Trade-off 1: Offline Setup Cost vs. Inference Efficiency

Running the teacher offline costs approximately $0.044 per query (GPT-4o pricing). This is approximately 1.5x more expensive than a direct GPT-4o inference call ($0.03). However, DRAG converts this one-time cost into unlimited future inference runs at student cost (~$0.001). The break-even point is approximately 50 inference calls per distilled query.

For benchmarks with 1,000+ test samples per dataset, this break-even is easily reached during a single experiment run.

### Trade-off 2: Graph Token Efficiency vs. Structural Loss

Converting evidence to a graph saves 18.1% of tokens on average but necessarily loses some information: narrative coherence, hedging language, conditional qualifications, and subtle connotations that are present in prose but absent in triples. The empirical result is that the combination of evidence plus graph outperforms either alone, suggesting that the evidence retains the nuances that the graph loses, while the graph adds structural clarity that the evidence lacks.

### Trade-off 3: Privacy vs. Accuracy

PII stripping reduces task accuracy by 12.9% in privacy mode. This cost arises because sometimes the PII is semantically relevant (e.g., knowing a patient's age is relevant to medication dosing). The design choice is to accept this accuracy cost as necessary for privacy compliance. For non-privacy-sensitive deployments, the filter can be disabled entirely.

### Trade-off 4: Fixed vs. Adaptive K

Fixed K is simpler to configure and reason about. Adaptive K requires domain awareness and a complexity scorer. The accuracy gain from adaptive K (~1.8% on average) was judged sufficient to justify the added complexity. For users who want simplicity, fixed K=15 is available as a configuration option.

### Trade-off 5: File-Based Cache vs. Database

A file-based cache is simple to inspect, debug, and version control. A database would offer faster lookups at scale and transactional consistency. For the research use case (datasets of up to 10,000 queries), file-based is clearly sufficient. The `ArtifactRepository` abstraction ensures this is a swap, not a rewrite, when the need arises.

---

## 9. Anti-Patterns and What Was Deliberately Avoided

### Not Using Fine-Tuning as the Primary Knowledge Transfer Mechanism

Fine-tuning the student on teacher outputs is a classical approach to knowledge distillation, but it requires labelled training data, training infrastructure, and GPU resources proportional to model size. DRAG achieves similar or better results without fine-tuning by providing structured evidence and graph context at inference time. This keeps the framework accessible to users without ML training infrastructure.

### Not Retrieval-Augmenting the Teacher

A naive design might augment the teacher with a retrieval system to improve evidence quality. This would add corpus management complexity, reduce portability, and couple DRAG to a specific retrieval system. The teacher's parametric knowledge is sufficient, and the ranking filter handles noise.

### Not Using a Vector Database for Evidence

Vector databases are popular in RAG systems for live retrieval. DRAG deliberately avoids them because there is no live retrieval in DRAG. Evidence is generated, not retrieved, and it is generated once and reused. A vector database would add infrastructure complexity and a dependency that provides no value for the offline-generation use case.

### Not Designing for Single-Model Scenarios

DRAG is explicitly a two-model system: teacher and student. A single-model design (just improve the student directly) would miss the core insight that the teacher's knowledge is the bottleneck, not the student's reasoning capacity. The student with well-structured context from a strong teacher consistently outperforms the teacher invoked directly for many queries, because the context scaffolds reasoning in ways that improve reliability.

### Not Caching Responses Alongside Evidence

Response caching (reusing the student's answer for repeated queries) is a valid optimisation but outside DRAG's scope. DRAG caches evidence and graphs, not responses, because responses depend on the student model identity. Different students produce different responses from the same evidence. Caching responses would couple the cache to a specific student model, reducing flexibility.

---

## 10. Design Evolution and Future Directions

### Near-Term: Multi-Modal Evidence

The current design generates text-only evidence. Extending it to multi-modal evidence (images, tables, structured data) would require changes to the evidence data model (adding a `modality` field and a `content` union type) and updates to the teacher adapter to support vision models. The graph construction pipeline would need a visual entity extractor.

### Near-Term: Streaming Evidence Generation

Currently, evidence generation waits for all N calls to complete before ranking begins. A streaming design would start ranking as soon as the first evidence arrives, potentially reducing total latency for time-sensitive offline runs. This would require the ranker to operate on partial evidence sets and update rankings incrementally.

### Medium-Term: Continuous Knowledge Refresh

The current design generates evidence once and never updates it. For time-sensitive domains (medical guidelines, legal regulations), evidence may become outdated. A refresh scheduler would re-run evidence generation for queries whose cached artifacts are older than a domain-specific TTL (e.g., 30 days for medical, 90 days for general knowledge).

### Medium-Term: Reinforcement Learning from Student Feedback

The student model's performance on evaluation queries can be used as a reward signal to improve the teacher's evidence generation prompts. Queries where the student fails can be queued for re-distillation with refined prompts. This creates a feedback loop that continuously improves evidence quality for difficult queries without manual intervention.

### Long-Term: Federated Distillation

In scenarios where multiple organisations have similar query distributions but cannot share data, federated distillation would allow each organisation to distill evidence independently and share only the graph structures (which are semantic, not personally identifiable). A central graph aggregation layer could merge these graphs into a shared knowledge resource while preserving each organisation's data privacy.

### Long-Term: Interpretable Distillation Audit

As DRAG is deployed in high-stakes domains (medical, legal), there will be requirements to explain why a specific evidence set and graph were used for a given answer. An audit module would trace the provenance of each graph edge back to the specific evidence piece and teacher API call that generated it, providing a complete chain of evidence for any response.

---

## References

- Chen et al., "DRAG: Distilling RAG for SLMs from LLMs", ACL 2025. https://arxiv.org/abs/2506.01954
- Hinton et al., "Distilling the Knowledge in a Neural Network", NeurIPS 2014
- Edge et al., "From Local to Global: A Graph RAG Approach", 2024
- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", NeurIPS 2020
- Gamma et al., "Design Patterns: Elements of Reusable Object-Oriented Software", Addison-Wesley 1994

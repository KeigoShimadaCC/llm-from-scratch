# NORTH_STAR.md — Mac Local LLM From Scratch

## 0. Project Identity

**Project name:** `mac-local-llm-from-scratch`
**Working name:** `KeigoGPT-Lab`
**Primary machine:** Mac mini M4 Pro, 64GB unified memory
**Primary purpose:** educational, but technically serious
**Target outcome:** a small but coherent causal language model trained from scratch, with a clear path from first-principles Transformer implementation to Mac-optimized inference.

---

## 1. North Star

### 1.1 What this project is trying to understand

The project exists to understand, by building, the full stack of a modern decoder-only LLM:

1. How text becomes tokens.
2. How tokens become embeddings.
3. How causal self-attention performs sequence modeling.
4. How Transformer blocks scale from toy models to small LLMs.
5. How next-token prediction creates general language behavior.
6. How data quality and tokenizer choices affect behavior.
7. How pretraining, instruction tuning, evaluation, and inference fit together.
8. How Apple Silicon changes the practical engineering choices.

The goal is not to create a GPT-5 competitor. The goal is to create a technically honest, inspectable, reproducible, from-scratch LLM laboratory.

A strong version of success is:

> A person already familiar with LLM architecture can open this repository and think:
> “This is a clean, serious, Mac-native LLM reconstruction project. It is educational, but not toy-only.”

---

### 1.2 What “from scratch” means here

“From scratch” means:

- We implement the model architecture ourselves.
- We implement the training loop ourselves.
- We build or train the tokenizer ourselves.
- We construct the dataset pipeline ourselves.
- We train the model weights ourselves from random initialization.
- We own the evaluation harness.
- We own the inference path.

“From scratch” does **not** mean:

- writing all linear algebra kernels manually;
- avoiding PyTorch/MLX autograd;
- avoiding all existing open-source utilities;
- refusing to use reference implementations for comparison;
- pretending that a local Mac can economically reproduce frontier-scale pretraining.

The project should distinguish three branches:

| Branch | Meaning |
|---|---|
| `scratch-core` | model, tokenizer, training loop, and weights built from scratch |
| `reference-baselines` | nanoGPT/GPT-2/other references used only for comparison |
| `practical-local-llm` | optional MLX/Ollama/LoRA experiments using existing open-weight models |

The first branch is the philosophical core. The third branch is allowed only as a practical comparison layer.

---

### 1.3 Non-goals

This project explicitly does **not** aim to:

- train a frontier model;
- train a production chatbot;
- beat open-weight 1B–10B models;
- collect a massive copyrighted web corpus;
- implement CUDA kernels;
- implement distributed GPU training;
- optimize for benchmark leaderboard performance;
- hide complexity behind Hugging Face `AutoModelForCausalLM`.

---

### 1.4 Success criteria

The project is successful if it produces:

1. A working decoder-only Transformer trained from random initialization.
2. A tokenizer trained or configured for English/Japanese mixed text.
3. At least three model scales:
   - micro model;
   - small model;
   - stretch model.
4. Reproducible training runs with saved configs and logs.
5. Held-out validation loss and perplexity tracking.
6. Qualitative generation samples over time.
7. A small instruction-tuned variant.
8. A Mac-native inference path.
9. A final technical write-up explaining:
   - architecture;
   - training dynamics;
   - data choices;
   - failure modes;
   - what changed between phases.

---

## 2. High-Level Project Plan

This section defines the phase structure. Each phase should later receive its own detailed plan.

---

### Phase 0 — Repository and Lab Foundation

**Goal:** create a clean research-engineering environment.

**Deliverables:**

- repository skeleton;
- config system;
- experiment logging;
- deterministic seeding;
- checkpoint format;
- local data directory convention;
- `README.md`;
- `NORTH_STAR.md`;
- minimal test suite.

**Key design decisions:**

- PyTorch first for readability.
- MLX later for Mac-native optimization.
- No Hugging Face model wrapper in core training path.
- All experiments must be config-driven.

**Exit criteria:**

- A dummy model can train for one epoch on fake token data.
- Checkpoint save/load works.
- Loss log and config artifact are written.
- Unit tests pass.

---

### Phase 1 — MicroGPT: Character-Level Language Model

**Goal:** build the smallest complete language-modeling loop.

**Scope:**

- character vocabulary;
- tiny Transformer or even bigram baseline first;
- next-character prediction;
- greedy and sampling-based generation;
- validation loss;
- overfitting test on tiny corpus.

**Why this phase matters:**

Before dealing with BPE tokenizers, multilingual corpora, and memory pressure, the project needs to prove that:

- masking works;
- loss decreases;
- sampling works;
- checkpointing works;
- training curves are interpretable.

**Exit criteria:**

- The model overfits a tiny dataset.
- The model generates recognizable text patterns.
- There is a written explanation of underfitting, overfitting, and sampling temperature.

---

### Phase 2 — Tokenizer and Dataset Pipeline

**Goal:** move from character-level modeling to real token-level LLM training.

**Scope:**

- train a small BPE or unigram tokenizer;
- compare English-only and bilingual Japanese/English tokenization;
- implement streaming/chunked tokenization;
- create train/validation splits;
- create binary token files or memory-mapped arrays;
- implement batch sampling.

**Core questions:**

- How large should the vocabulary be?
- How badly does Japanese fragment under a naive tokenizer?
- How does vocabulary size change sequence length?
- What is the trade-off between tokenizer quality and model size?

**Suggested tokenizer experiments:**

| Tokenizer | Use |
|---|---|
| character-level | debugging |
| byte-level BPE | robust baseline |
| SentencePiece unigram/BPE | Japanese/English bilingual path |
| GPT-2 tokenizer | reference only, not core scratch path |

**Exit criteria:**

- Tokenizer is trained or selected deliberately.
- Tokenization statistics are reported.
- Dataset can feed batches into the training loop.
- A small model trains on tokenized data.

---

### Phase 3 — Core Decoder-Only Transformer

**Goal:** implement the real model architecture.

**Architecture target:**

- decoder-only causal Transformer;
- token embeddings;
- positional representation;
- pre-norm Transformer blocks;
- multi-head self-attention;
- causal mask;
- MLP block;
- residual connections;
- tied input/output embeddings;
- autoregressive generation.

**Minimum architecture:**

```text
tokens
→ token embedding
→ position encoding
→ repeated Transformer blocks
   → norm
   → causal self-attention
   → residual
   → norm
   → MLP
   → residual
→ final norm
→ language-model head
→ next-token logits
```

**Stretch architecture options:**

- RoPE instead of learned positional embeddings;
- RMSNorm instead of LayerNorm;
- SwiGLU instead of GELU MLP;
- grouped-query attention;
- KV cache for inference;
- weight tying;
- gradient checkpointing.

**Exit criteria:**

- The model trains stably at micro scale.
- Attention mask is tested.
- Generation is autoregressive and causal.
- Parameter count is computed and logged.

---

### Phase 4 — Tiny Pretraining Run

**Goal:** train a small but real LLM from random initialization.

**Model scale:** roughly 5M–20M parameters.

**Training data:** small curated text corpus.

**Focus:**

- stable loss decrease;
- learning-rate schedule;
- batch size and gradient accumulation;
- validation loss;
- generated sample tracking;
- failure mode logging.

**What to learn:**

- What loss level corresponds to gibberish?
- When does syntax emerge?
- When does memorization occur?
- How much does data cleanliness matter?
- What happens when context length increases?

**Exit criteria:**

- Validation loss improves meaningfully.
- Samples become less random over training.
- Training run is reproducible from config.
- A short experiment report exists.

---

### Phase 5 — Small Practical Model

**Goal:** train a model large enough to be educationally meaningful.

**Model scale:** roughly 30M–100M parameters.

**Possible names:**

- `kgpt-30m`
- `kgpt-50m`
- `kgpt-100m`

**Focus:**

- bilingual data mixture;
- tokenizer quality;
- longer context;
- more robust eval;
- checkpoint averaging or best-checkpoint selection;
- Mac memory and throughput profiling.

**Expected behavior:**

The model should not be “smart” in the ChatGPT sense. It should, however:

- complete text coherently in-domain;
- preserve syntax better than the tiny model;
- show some factual-style continuation patterns;
- display clear scaling behavior relative to smaller runs.

**Exit criteria:**

- At least one 30M+ model trains from scratch.
- A stretch 50M–100M run is attempted if feasible.
- The project has credible loss curves and generation snapshots.
- Training bottlenecks are documented.

---

### Phase 6 — Instruction Tuning Layer

**Goal:** convert the pretrained model into a simple instruction-following model.

**Scope:**

- create or collect a small instruction dataset;
- implement supervised fine-tuning;
- define prompt format;
- compare base vs instruction-tuned outputs;
- avoid pretending it is aligned or safe in a production sense.

**Prompt format example:**

```text
### Instruction:
{user_instruction}

### Response:
{model_response}
```

**Core question:**

How much instruction-following behavior can be induced in a small model when base language ability is limited?

**Exit criteria:**

- Instruction-tuned checkpoint exists.
- Base-vs-SFT comparison is documented.
- The model can follow some narrow commands.
- Limitations are clearly described.

---

### Phase 7 — Evaluation and Failure Analysis

**Goal:** avoid “vibes-only” evaluation.

**Evaluation layers:**

1. **Training metrics**
   - train loss;
   - validation loss;
   - perplexity;
   - tokens/sec;
   - memory usage.

2. **Generation metrics**
   - sample quality over checkpoints;
   - repetition rate;
   - average completion length;
   - entropy/top-k behavior.

3. **Task probes**
   - copy task;
   - arithmetic toy task;
   - translation-like toy task;
   - Japanese continuation;
   - English continuation;
   - instruction-following micro set.

4. **Failure analysis**
   - repetition loops;
   - hallucinated facts;
   - language mixing;
   - bad tokenization artifacts;
   - memorization;
   - degeneration under long context.

**Exit criteria:**

- Evaluation script can run against any checkpoint.
- Each model scale has a comparable eval report.
- The project can explain not only what worked, but why things failed.

---

### Phase 8 — Mac-Native Inference and Optimization

**Goal:** make the model pleasant to run locally on Apple Silicon.

**Scope:**

- PyTorch MPS inference baseline;
- MLX inference path;
- quantization experiment if supported;
- KV cache;
- simple CLI;
- optional local web UI;
- latency and throughput benchmark.

**Deliverables:**

- `generate.py`;
- `chat.py`;
- `bench_inference.py`;
- model card;
- example outputs.

**Exit criteria:**

- The trained model can run locally without cloud services.
- Inference speed and memory footprint are measured.
- There is a clear comparison between PyTorch MPS and MLX paths.

---

### Phase 9 — Final Write-Up and Portfolio Layer

**Goal:** make the project legible to technical reviewers.

**Final write-up structure:**

1. Motivation.
2. What “from scratch” means.
3. Architecture.
4. Dataset and tokenizer.
5. Training methodology.
6. Scaling experiments.
7. Instruction tuning.
8. Evaluation.
9. Mac optimization.
10. What I would do differently with more compute.
11. What this taught me about LLMs.

**Portfolio signal:**

This should read as:

> “I understand LLMs below the API layer.”

Not:

> “I ran a fine-tuning notebook.”

---

## 3. Expected Technical Specification

---

### 3.1 Hardware Assumptions

**Primary machine:**

- Mac mini M4 Pro
- 64GB unified memory
- Apple Silicon GPU via Metal/MPS
- Local SSD plus optional external SSD for datasets/checkpoints

**Practical implications:**

- unified memory is useful for local model work;
- CUDA-only assumptions should be avoided;
- PyTorch MPS is acceptable for initial training;
- MLX is likely the best Mac-native optimization path;
- 1B+ from-scratch pretraining is out of scope;
- 5M–100M from-scratch experiments are in scope;
- existing 1B–10B local inference/fine-tuning can be a separate practical branch.

---

### 3.2 Software Stack

#### Core language

- Python 3.11 or 3.12

#### Environment

- `uv` or `conda`
- `pyproject.toml`
- `ruff`
- `pytest`
- optional `mypy`

#### Numerical / ML libraries

| Layer | Preferred tool |
|---|---|
| first implementation | PyTorch |
| Mac GPU acceleration | PyTorch MPS |
| Mac-native optimization | MLX |
| array/data processing | NumPy |
| tables/analysis | pandas |
| plotting | matplotlib |
| tokenizer | SentencePiece or Hugging Face tokenizers |
| CLI | Typer or argparse |
| progress/logging | tqdm, rich |
| experiment logs | CSV/JSONL first; W&B optional |

#### Why PyTorch first

PyTorch is more familiar, more documented, and better for debugging the first implementation.

#### Why MLX later

MLX is designed for Apple Silicon and should be explored once the model and training logic are understood.

---

### 3.3 Repository Structure

Suggested structure:

```text
mac-local-llm-from-scratch/
  README.md
  NORTH_STAR.md
  pyproject.toml

  configs/
    micro_char.yaml
    tiny_bpe.yaml
    kgpt_30m.yaml
    kgpt_100m.yaml
    sft_small.yaml

  data/
    raw/
    processed/
    tokenized/
    README.md

  tokenizer/
    train_tokenizer.py
    inspect_tokenizer.py
    tokenizer_report.py

  kgpt/
    __init__.py
    model.py
    attention.py
    blocks.py
    config.py
    sampling.py
    checkpoint.py

  train/
    train_pretrain.py
    train_sft.py
    schedule.py
    dataloader.py

  eval/
    eval_loss.py
    eval_generate.py
    eval_tasks.py
    eval_report.py

  inference/
    generate.py
    chat.py
    bench_inference.py
    mlx_generate.py

  experiments/
    runs/
    reports/

  tests/
    test_attention_mask.py
    test_shapes.py
    test_tokenizer_roundtrip.py
    test_checkpoint.py
```

---

### 3.4 Model Family

The project should define a small model family rather than a single model.

Approximate model ladder:

| Name | Params | Layers | d_model | Heads | Vocab | Context | Purpose |
|---|---:|---:|---:|---:|---:|---:|---|
| `micro-char` | <1M | 2–4 | 64–128 | 2–4 | char | 128–256 | debugging |
| `tiny-bpe` | 5M–10M | 6 | 256 | 4 | 8k–16k | 256–512 | first real token model |
| `small-bpe` | 20M–40M | 8–10 | 384 | 6 | 16k–32k | 512–1024 | meaningful local run |
| `kgpt-50m` | 40M–70M | 10–12 | 512 | 8 | 32k | 1024 | main educational model |
| `kgpt-100m` | 90M–120M | 12 | 768 | 12 | 32k | 1024 | stretch goal |

These are approximate targets. Parameter count should be computed programmatically for each config.

---

### 3.5 Core Architecture

The main architecture should be a decoder-only causal Transformer.

#### Base architecture

- token embedding;
- positional encoding;
- stacked Transformer blocks;
- causal self-attention;
- feed-forward MLP;
- residual connections;
- normalization;
- final language-modeling head.

#### Recommended first version

| Component | First implementation |
|---|---|
| position | learned absolute positional embeddings |
| norm | LayerNorm |
| MLP | GELU |
| attention | multi-head self-attention |
| output head | tied embedding/head weights |
| dropout | small dropout for regularization |
| context | 256–1024 tokens |

#### Recommended serious version

| Component | Upgrade |
|---|---|
| position | RoPE |
| norm | RMSNorm |
| MLP | SwiGLU |
| attention | MHA first; GQA optional |
| inference | KV cache |
| precision | mixed precision where stable |
| context | 1024+ tokens if feasible |

The project should implement the simple version first. Then upgrade one component at a time.

---

### 3.6 Training Objective

Primary objective:

```text
next-token prediction
```

Given tokens:

```text
x_0, x_1, ..., x_n
```

The model predicts:

```text
x_1, x_2, ..., x_{n+1}
```

Loss:

```text
cross-entropy over vocabulary
```

Important training details:

- AdamW optimizer;
- learning-rate warmup;
- cosine decay;
- gradient clipping;
- gradient accumulation;
- periodic validation;
- checkpointing by validation loss;
- deterministic seeds;
- sample generation every N steps.

---

### 3.7 Tokenizer Strategy

Tokenizer should be treated as a first-class part of the project.

#### Initial path

1. character-level tokenizer for Phase 1;
2. byte-level/BPE tokenizer for Phase 2;
3. bilingual Japanese/English tokenizer for main model.

#### Tokenizer report should include

- vocabulary size;
- average tokens per English sentence;
- average tokens per Japanese sentence;
- unknown-token behavior;
- byte fallback behavior;
- examples of bad segmentation;
- compression ratio;
- effect on context length.

#### Candidate vocabulary sizes

| Vocab size | Use |
|---:|---|
| char | debugging |
| 8k | tiny experiments |
| 16k | small bilingual model |
| 32k | main bilingual model |
| 50k+ | likely too expensive for small models unless justified |

For small models, an overly large vocabulary wastes parameters in the embedding matrix. The tokenizer should be matched to the model size.

---

### 3.8 Dataset Strategy

This project should prioritize clean, inspectable data over raw scale.

#### Dataset principles

- start tiny;
- keep licensing clean;
- deduplicate aggressively;
- separate train and validation early;
- document every source;
- preserve reproducibility;
- avoid contaminated benchmark data where possible;
- keep Japanese and English proportions explicit.

#### Suggested stages

| Stage | Data size | Purpose |
|---|---:|---|
| tiny | <10MB | debugging and overfit tests |
| small | 100MB–500MB | first meaningful pretraining |
| medium | 1GB–5GB | stretch local pretraining |
| practical | larger only if storage/time allow | long run |

#### Data mixture examples

| Mix | English | Japanese | Code | Use |
|---|---:|---:|---:|---|
| English debug | 100% | 0% | 0% | simplest baseline |
| bilingual balanced | 50% | 50% | 0% | main educational target |
| bilingual technical | 40% | 40% | 20% | practical assistant flavor |
| Japanese-heavy | 20% | 80% | 0% | Japanese tokenizer stress test |

---

### 3.9 Training Runs

Each training run should produce a run directory:

```text
experiments/runs/{timestamp}_{run_name}/
  config.yaml
  metrics.jsonl
  samples.txt
  checkpoint_last.pt
  checkpoint_best.pt
  tokenizer_info.json
  eval_report.md
```

Minimum logged metrics:

- step;
- tokens seen;
- train loss;
- validation loss;
- perplexity;
- learning rate;
- gradient norm;
- tokens/sec;
- memory usage if available;
- generated sample.

---

### 3.10 Evaluation Design

Evaluation should be built from the beginning.

#### Quantitative

- validation loss;
- perplexity;
- tokens/sec;
- memory use;
- repetition metrics;
- exact-match toy tasks.

#### Qualitative

A fixed prompt set should be sampled across checkpoints.

Example prompt categories:

| Category | Example |
|---|---|
| English continuation | “The history of artificial intelligence began” |
| Japanese continuation | “人工知能の歴史は” |
| technical explanation | “A transformer model uses attention because” |
| instruction | “Explain gradient descent in simple terms.” |
| bilingual | “Translate this short sentence into Japanese:” |
| self-consistency | same prompt at multiple temperatures |

#### Failure taxonomy

Every serious run should classify failures:

- pure gibberish;
- syntax without semantics;
- repetition loops;
- mode collapse;
- English/Japanese mixing;
- memorized fragments;
- bad token boundaries;
- instruction ignored;
- false factual confidence.

---

### 3.11 Inference Design

Inference should be implemented separately from training.

Required features:

- greedy decoding;
- temperature sampling;
- top-k sampling;
- top-p sampling;
- repetition penalty optional;
- stop token handling;
- max new tokens;
- KV cache for serious version.

Interfaces:

```text
generate.py  # completion mode
chat.py      # instruction/chat format
bench.py     # latency/throughput measurement
```

Mac-specific inference path:

- PyTorch MPS baseline;
- MLX implementation or export;
- optional quantized inference;
- optional local UI later.

---

### 3.12 Agentic Coding Workflow

Agentic coding should accelerate implementation, but not replace project ownership.

Recommended agent roles:

| Agent role | Responsibility |
|---|---|
| implementer | writes initial module |
| reviewer | checks shape logic, masks, training bugs |
| test writer | adds tests for invariants |
| experiment analyst | reads logs and writes reports |
| refactor agent | improves structure without changing behavior |

Human-owned decisions:

- architecture;
- data choice;
- tokenizer choice;
- phase gates;
- interpretation of loss curves;
- whether a result is real or misleading.

Suggested workflow:

1. Write phase issue.
2. Ask coding agent for implementation plan.
3. Implement minimal version.
4. Run tests.
5. Ask reviewer agent to inspect.
6. Run small experiment.
7. Write short report.
8. Only then scale.

---

### 3.13 Testing Strategy

Minimum tests:

- tokenizer encode/decode roundtrip;
- batch shape correctness;
- attention mask prevents future-token access;
- logits shape is `[batch, time, vocab]`;
- loss decreases on tiny overfit dataset;
- checkpoint save/load exactness;
- generation does not crash;
- config loads correctly.

Important invariant:

> A model must be able to overfit a tiny dataset before it is trusted on a larger dataset.

---

### 3.14 Risks and Mitigations

| Risk | Mitigation |
|---|---|
| MPS unsupported operation | fallback to CPU for that op, simplify implementation, or move to MLX |
| training too slow | reduce context, batch, layers, or data size |
| model never learns | overfit tiny dataset first |
| model only memorizes | improve split/dedup and reduce repeated data |
| Japanese tokenization poor | inspect tokenization and retrain tokenizer |
| false sense of progress | fixed eval prompts and validation loss |
| messy repo | phase gates and tests |
| data licensing issues | document sources and use permissive/public datasets |
| too much time spent optimizing | PyTorch first, MLX later |
| too much scope creep | keep practical fine-tuning branch separate |

---

### 3.15 Compute Expectations

This project should be calibrated honestly.

Likely feasible:

- character models;
- 5M–20M token-level models;
- 30M–70M models with patience;
- 100M-ish stretch run;
- inference for much larger open-weight models as comparison.

Likely not feasible locally:

- training a strong 1B model from scratch;
- reproducing GPT-2 124M at full OpenWebText scale quickly;
- frontier-style instruction tuning;
- RLHF at serious scale;
- distributed training.

The useful educational target is not maximum parameter count. It is the full causal chain from text data to trained weights to evaluated behavior.

---

## 4. Proposed Milestone Table

| Milestone | Description | Output |
|---|---|---|
| M0 | repo foundation | tests, configs, logging |
| M1 | character model | tiny generation demo |
| M2 | tokenizer pipeline | tokenizer report |
| M3 | Transformer core | trainable GPT-like model |
| M4 | tiny pretraining | 5M–20M checkpoint |
| M5 | small pretraining | 30M–70M checkpoint |
| M6 | stretch pretraining | 100M attempt |
| M7 | instruction tuning | SFT checkpoint |
| M8 | evaluation suite | comparable model reports |
| M9 | Mac inference | CLI + MLX path |
| M10 | final write-up | portfolio-quality technical report |

---

## 5. Definition of Done

The project reaches “done” when the following are true:

- `scratch-core` contains a self-implemented decoder-only Transformer.
- The model can be trained from random initialization.
- Tokenizer and data pipeline are documented.
- At least one 30M+ model is trained.
- At least one instruction-tuned model is produced.
- Evaluation reports exist for each serious checkpoint.
- Inference runs locally on the Mac.
- The final write-up explains the architecture, experiments, and limitations.
- The repository is understandable to an engineer who already knows LLMs.

---

## 6. Final Positioning

This project should sit between:

```text
toy summer homework
```

and

```text
production LLM company
```

The intended position is:

```text
serious educational research-engineering artifact
```

The right taste is:

- small enough to finish;
- deep enough to be nontrivial;
- honest about compute;
- Mac-native;
- architecture-forward;
- evaluation-aware;
- useful as a portfolio artifact.

The core statement:

> This project is a full-stack reconstruction of a small decoder-only LLM on Apple Silicon, built from scratch for understanding rather than benchmark chasing.

# Final Project

*Instructor: Chris Larson | Georgetown University | ANLY-5800 | Fall '25*

---

## Overview

The final project is a 3‑week, group-based assignment in which you will build, finetune, or analyze an NLP system involving transformers, finetuning, evaluation, scaling, agents, etc.

Projects may follow one of several structured tracks or a student-defined variant that fits within this framework. Across all tracks, you are expected to:

- Implement or adapt a non-trivial model or system
- Run meaningful experiments (with baselines and at least one ablation/variant)
- Use sound evaluation with appropriate metrics
- Communicate your work clearly in code, a written report, and a presentation

---

## Project guidelines

### Group size

- Recommended: 2–4 students
- Solo projects only by exception (must be explicitly approved and appropriately scoped)

### Deliverables

- GitHub repository
  - Clean, runnable code
  - `README.md` with setup, how to run experiments, and how to reproduce main results
- Project report (3-5 pages, PDF)
  - IMRaD-style: Introduction, Related Work, Methods, Experiments, Results/Discussion, Limitations/Future Work (markdown formatted!).
- Final presentation and demo
  - 15 minutes talk + 5 minutes Q&A
  - Live demo strongly encouraged

---

## Three‑week timeline & milestones

This timeline is relative (Week 1, 2, 3). Exact calendar dates will be specified on the syllabus.

### Week 1 – Problem, data, and baseline

- Checkpoint 1 (mid‑Week 1): One‑pager proposal
  - Use `project/proposal-template.md` as a guide.
  - Includes:
    - Problem statement and motivation
    - Track choice and scope
    - Dataset(s): source, size, preprocessing plan
    - Baseline idea (simplest thing that could work)

- Checkpoint 2 (end of Week 1): Baseline & plan
  - Data preprocessing scripts in the repo
  - A running baseline
  - Initial baseline metrics
  - A short progress note (1–2 pages in the repo) describing:
    - What you built
    - Baseline performance
    - Two concrete improvements you plan to implement in Weeks 2–3

### Week 2 – Core system and main experiments

- Implement the core model or system for your chosen track (see Tracks below).
- Run at least one full experimental comparison (baseline vs. improved system).
- By the end of Week 2:
  - End-to-end pipeline running on your task
  - At least one complete set of results with metrics and plots
  - Draft experimental section for your report (including setup and preliminary analysis)

### Week 3 – Refinement, ablations, and communication

- Complete remaining ablations/variants.
- Strengthen evaluation:
  - Quantitative: metrics, curves, tables
  - Qualitative: representative successes and failures, error analysis
- Finalize:
  - Repository structure, documentation, scripts
  - Final report
  - Demo

Final report and presentation are due at the end of Week 3; exact date/time will be posted on Canvas.

---

## Tracks (project options)

You may choose any of the following tracks or propose your own (Track E). All tracks should include:

- A clearly defined NLP task and dataset
- At least one baseline and one improved system
- At least one ablation or variant (e.g., architecture change, data size, hyperparameter)

### Track A – Tiny Transformer language model (from scratch)

Objective: Build and train a small transformer-based language model from scratch to understand the mechanics of modern LMs.

Core requirements

- Implement (or heavily adapt, with understanding) a minimal Transformer:
  - Tokenization (character‑level or BPE)
  - Positional encodings
  - Multi-head self-attention
  - Feed-forward blocks
  - Layer normalization
  - Causal masking for autoregressive modeling
- Train on a small corpus of your choice (e.g., domain text, lyrics, code, class notes).
- Evaluate using:
  - Perplexity or next-token prediction accuracy
  - Qualitative generations

Experiments

- At least two variants, e.g.:
  - Different model sizes (layers/hidden dim)
  - Different context lengths
  - Different dataset sizes (e.g., 10%, 50%, 100%)

Suggested resources

- [`nanoGPT`](https://github.com/karpathy/nanoGPT) for reference (do not just clone + run; you must understand and adapt)
- [Llama GitHub repository](https://github.com/meta-llama)

---

### Track B – LoRA finetuning for a downstream task

Objective: Efficiently adapt a pretrained LLM to a specific NLP task using Low‑Rank Adaptation (LoRA) and analyze trade‑offs.

Core requirements

- Choose:
  - A pretrained LLM (e.g., LLaMA variant, Mistral, etc.)
  - A task: text classification, summarization, QA, dialogue, instruction following, etc.
- Implement or configure LoRA:
  - Understand where adapters are injected (e.g., attention projections, feed-forward layers)
  - Control LoRA rank and scaling
- Train on:
  - A public dataset (GLUE, SST‑2, SQuAD, etc.) or
  - A custom dataset you collect/curate

Experiments

- Compare:
  - LoRA vs. a simple baseline (e.g., prompt‑only or frozen encoder + linear head)
  - At least one LoRA hyperparameter variant (e.g., rank or learning rate)
- Evaluate with task-appropriate metrics:
  - Accuracy/F1 for classification
  - BLEU/ROUGE for generation/summarization
  - Exact match / F1 for QA

Suggested resources

- [LoRA paper](https://arxiv.org/abs/2106.09685)
- PEFT / Hugging Face ecosystem for practical patterns (if used, you still need to explain the math)

---

### Track C – LLM agent with dynamic tool usage and/or code execution

Objective: Build an LLM-based agent that can decide when to call tools and/or execute code to solve complex queries.

Core requirements

- Implement an agent loop:
  - Observe user query and current context
  - Decide whether to:
    - Answer directly, or
    - Call a tool (e.g., retrieval, calculator, web API, code-exec)
  - Incorporate tool results into subsequent reasoning
- Support at least two tools, such as:
  - Calculator / symbolic math
  - Code execution (Python sandbox)
  - Vector search over documents
  - External API
- Build a small evaluation suite of multi-step questions/tasks where tools are necessary.

Experiments

- Compare:
  - Agent with tools vs. baseline LLM without tools
  - Possibly different orchestration strategies (e.g., ReAct-style vs. simple tool-calling heuristics)
- Measure:
  - Task success rate / accuracy
  - Latency / number of tool calls

Important constraint

- LLM and tool orchestration must be implemented by you, not simply delegated to a black-box managed service. You may use libraries (e.g., LangChain) as building blocks, but your solution cannot simply be a wrapper around existing tools.

Suggested resources

- [LangChain GitHub repository](https://github.com/langchain-ai/langchain)
- [Llama-stack GitHub repository](https://github.com/meta-llama/llama-stack)
- [ReAct paper](https://arxiv.org/pdf/2210.03629)

---

### Track D – Analysis, evaluation, or scaling study

Objective: Answer a focused empirical question about language models using careful experimental design and analysis.

Example questions

- How does performance scale with:
  - Model size?
  - Training data size?
  - Context length?
- How do different finetuning strategies compare?
  - SFT vs. instruction tuning vs. DPO (on a small setup)
- How robust is a model to:
  - Prompt perturbations or adversarial prompts?
  - Domain shift (train vs. test domain)?

Core requirements

- Clearly stated research question and hypotheses.
- Implementation of training/fine-tuning or evaluation pipeline.
- Systematic experiments with multiple settings, not just one run.
- Plots/tables that answer your question (e.g., scaling curves, robustness curves).

---

### Track E – Student-defined project

Students may propose their own project, provided it:

- Involves a significant component of NLP research or application (novel model, task, dataset, or analysis).
- Ties clearly to course content:
  - Language models, transformers, etc.
- Includes:
  - Clear problem statement and motivation
  - Methods grounded in course material
  - At least one baseline and one improved system/analysis
  - A realistic but ambitious 3‑week plan

Use the proposal template (`project/proposal-template.md`) and discuss with the instructor for approval.

---

## Jetstream2 compute resources

Google Colab is an excellent choice of most project work. If you need additional resources, we can grant access to free compute resources via a Jupyter Lab notebook hosted on the Jetstream2 supercomputing cluster with Nvidia A100 (40GB) GPUs.

*See [jetstream2.md](project/jetstream2.md) for environment, connection, and storage details.*

---

## Evaluation criteria (updated rubric)

| Category                          | Weight | Description                                                                 |
|--------------------------------------|------------|---------------------------------------------------------------------------------|
| Technical depth & correctness    | 30%        | Sound implementation; appropriate modeling choices; code runs & is reproducible |
| Experimental design & analysis   | 25%        | Quality of baselines, metrics, ablations, and interpretation of results        |
| Ambition & scope                 | 15%        | Project difficulty/novelty relative to group size and 3‑week timeframe         |
| Communication (report)  | 20%        | Clarity, organization, use of figures/tables, and explanation of concepts      |
| Code quality & documentation     | 10%        | Repo structure, README, comments, and ease of reproducing key experiments      |

Important: You will not be graded on *how good* your model is in absolute terms, but on how well you design, execute, and analyze your project within the constraints.

**Importantly, negative results (e.g., an idea that doesn’t improve performance) are completely acceptable if the experimentation and analysis are careful and well explained**.

---

## Checklist

- Week 1
  - [ ] Proposal one‑pager (problem, dataset, baseline, plan)
  - [ ] Running baseline + initial metrics
- Week 2
  - [ ] Core model/system implemented
  - [ ] At least one full experimental comparison (baseline vs improved)
  - [ ] Draft experimental section for report
- Week 3
  - [ ] Ablations/variants completed
  - [ ] Error analysis + qualitative examples
  - [ ] Repo cleaned; README updated
  - [ ] Final report
  - [ ] Demo


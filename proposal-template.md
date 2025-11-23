# ANLY-5800 Final Project – Proposal Template

Use this as a guide for your **2-page max** project proposal (one per group). You can write it directly in this file or a separate PDF in your repo.

---

## 1. Team

- **Project title**:
- **Team members**: Names & NetIDs
- **Preferred track**: (A) Tiny LM, (B) LoRA finetuning, (C) Agent, (D) Analysis, (E) Student-defined

---

## 2. Problem statement & motivation

- **Task**: What are you trying to do? (e.g., sentiment classification, QA, code generation, tool-using agent)
  - Sentiment classification using a recurrent Transformer architecture for better parameter efficiency.
- **Why it matters**: Briefly explain why this problem is interesting or important (scientifically or practically).
  - Sentiment signals enable downstream industrial pipelines, such as education grading and evaluation workflows.
  - A recurrent Transformer can generalize better while supporting reasoning-style, stepwise conditioning in a compact model.
- **Desired outcome**: What will success look like in 3 weeks?
  - Benchmark against an encoder baseline, demonstrating measurable performance gains while keeping the recurrent model smaller and within a comparable training budget.
---

## 3. Datasets

- **Primary dataset(s)**:
  - *SST-2 (Stanford Sentiment Treebank v2)* – [GLUE benchmark](https://huggingface.co/datasets/glue/viewer/sst2/train) with 67k English movie-review sentences labeled positive/negative.
  - *Yelp Review Polarity* (optional scale-up) – [Hugging Face](https://huggingface.co/datasets/yelp_polarity) release with 560k English business reviews (binary sentiment).
- **Preprocessing**:
  - **Normalization**: Standardize text by lowercasing and normalizing whitespace. Truncate sequences to 256 tokens to optimize GPU memory usage.
  - **Tokenization**: Use Hugging Face `AutoTokenizer` (initialized from the baseline checkpoint) or the `tokenizers` library/SentencePiece to map text to token IDs.
  - **Embeddings**: Freeze the tokenizer vocabulary but keep the embedding layer trainable to adapt representations to the sentiment task.
  - **Label Mapping (Yelp)**: If using Yelp, map 4–5 stars → positive and 1–2 stars → negative; filter out reviews >512 tokens.
- **Train/val/test split**:
  - SST-2: follow GLUE splits (≈67k train, 872 dev, 1,821 test) and carve out 5% of the train set as an internal validation set for hyperparameters.
  - Yelp: use official 560k train / 38k test split and reserve 10k samples from the train portion as validation when needed.

---

## 4. Baseline

- **Baseline model/system**: What is the simplest reasonable model you will implement in Week 1?
  - Examples: TF-IDF + logistic regression, zero-shot LLM, off-the-shelf checkpoint without finetuning, a tiny RNN.
- **Baseline metrics**: What metric(s) will you report (accuracy, F1, BLEU/ROUGE, perplexity, etc.)?

---

## 5. Approach (beyond baseline)

Describe your **core idea(s)** for improving over the baseline, tied to the course content.

Examples:

- Track A: Modify Transformer depth/width, context length, or data size and study effects.
- Track B: LoRA finetuning with different ranks, target modules, or objectives.
- Track C: Tool-using agent with ReAct-style reasoning, or multiple tools.
- Track D: Scaling study, robustness analysis, or comparison of finetuning strategies.
- Track E: Any combination of the above, but clearly grounded in course topics.

You don’t need every detail now, but you should have **at least two concrete improvements or experiments** planned beyond the baseline.

---

## 6. Compute & resources

- **Will you use Jetstream2?** (yes/no)
- **Rough plan**: Expected model sizes, batch sizes, and approximate training time.
- **Other resources** (if any): local GPUs, other cloud providers, external APIs.

---

## 7. Risks & scope

- **What could go wrong?** (e.g., data too noisy, model too big to train, evaluation too hard)
- **Plan B**: If your original idea is too ambitious, what scaled-down version will you execute?

---

## 8. Milestones

Very briefly, list what you plan to achieve by:

- **End of Week 1**:
- **End of Week 2**:
- **End of Week 3**:

These should align with the course-wide milestones in `project/README.md`.

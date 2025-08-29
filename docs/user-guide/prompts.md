# Prompt Engineering for GPT_AITIS

This guide explains the prompts available in GPT_AITIS for insurance policy analysis. Each prompt has been designed and tested to maximize accuracy while minimizing hallucination.

## Available Prompts

### Basic Prompts

**standard**
```bash
python src/main.py --prompt standard
```
The original prompt with basic instructions for Yes/No determination and quote requirements.

---

**detailed**
```bash
python src/main.py --prompt detailed
```
Enhanced version that identifies affected persons and handles multi-party coverage scenarios.

### Precise Series

**precise**
```bash
python src/main.py --prompt precise
```
First major improvement with strict quote requirements and sanity checks.

---

**precise_v2**
```bash
python src/main.py --prompt precise_v2
```
Better handling of conditions and clearer eligibility rules.

---

**precise_v3**
```bash
python src/main.py --prompt precise_v3
```
Major accuracy improvements with strict verbatim copying and anti-hallucination measures.

---

**precise_v4**
```bash
python src/main.py --prompt precise_v4
```
Enhanced condition checking with temporal/territorial awareness.

---

**precise_v5**
```bash
python src/main.py --prompt precise_v5
```
Latest version optimized for Phi-4 with better pattern matching and ALL RISKS handling.

### Model-Specific Prompts

**precise_v2_qwen**
```bash
python src/main.py --model hf --model-name Qwen/Qwen2.5-32B --prompt precise_v2_qwen
```
Compliance-focused prompt for Qwen models with legal defensibility.

---

**precise_v3_qwen**
```bash
python src/main.py --model hf --model-name Qwen/Qwen2.5-32B --prompt precise_v3_qwen
```
Compact, deterministic prompt optimized for Qwen models.

---

**precise_v4_qwen**
```bash
python src/main.py --model hf --model-name Qwen/Qwen2.5-32B --prompt precise_v4_qwen
```
Extreme strictness version for Qwen with machine-only function approach.

---

**precise_v5_qwen**
```bash
python src/main.py --model hf --model-name Qwen/Qwen2.5-32B --prompt precise_v5_qwen
```
Enhanced Qwen prompt that reduces false negatives.

---

**precise_v3_phi-4_v2**
```bash
python src/main.py --model hf --model-name microsoft/phi-4 --prompt precise_v3_phi-4_v2
```
Ultra-strict extractor for Phi-4 that forbids generative language.

### Relevance Filter Prompts

**relevance_filter_v1**
```bash
python src/main.py --filter-irrelevant --prompt-relevant relevance_filter_v1
```
Basic filtering with high-level category matching.

---

**relevance_filter_v2**
```bash
python src/main.py --filter-irrelevant --prompt-relevant relevance_filter_v2
```
Enhanced filtering with better category definitions and examples.

## Recommended Usage

### For Phi-4 Models
```bash
python src/main.py \
    --model hf \
    --model-name microsoft/phi-4 \
    --prompt precise_v5 \
    --filter-irrelevant \
    --prompt-relevant relevance_filter_v2
```

### For Qwen Models
```bash
python src/main.py \
    --model hf \
    --model-name Qwen/Qwen2.5-32B \
    --prompt precise_v4_qwen \
    --filter-irrelevant \
    --prompt-relevant relevance_filter_v2
```

### For OpenRouter API
```bash
python src/main.py \
    --model openrouter \
    --model-name qwen/qwen-2.5-72b-instruct \
    --prompt precise_v4_qwen
```

## Key Components of Effective Prompts

### 1. Role Definition
Clear statement of what the model should do:

- Text scanner (cannot create content)
- Insurance expert (can analyze)

### 2. Task Instructions
Numbered steps for clarity:

1. Find relevant policy text
2. Decide eligibility category
3. Quote supporting text
4. Output in JSON format

### 3. Anti-Hallucination Rules
Explicit prohibitions:

- No creating sentences
- No paraphrasing
- No ellipsis or [...]
- Only copy existing text

### 4. Output Format
Strict JSON structure with three fields:

- eligibility: One of three exact values
- outcome_justification: Verbatim quote or empty string
- payment_justification: Amount quote or null

## Creating Custom Prompts

To add a custom prompt:

1. Add to `src/prompts/insurance_prompts.py`:
```python
@classmethod
def custom_prompt(cls) -> str:
    return """Your prompt here"""
```

2. Register in the prompt_map:
```python
prompt_map = {
    # existing prompts...
    "custom": cls.custom_prompt(),
}
```

3. Use it:
```bash
python src/main.py --prompt custom
```

## Common Issues and Solutions

**Hallucination**: Model creates non-existent policy text

- Solution: Use stricter prompts with explicit copy-only rules

**Wrong Category**: Incorrectly classifies coverage

- Solution: Add explicit condition checking rules

**Truncated Quotes**: Model uses [...] or shortens text

- Solution: Forbid ellipsis and require complete sentences

**Format Violations**: Model adds explanations outside JSON

- Solution: Specify first character must be '{' and stop at '}'
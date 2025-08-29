# Model Selection Guide

This guide helps you choose the right model for your insurance policy analysis needs, covering performance, cost, accuracy, and technical requirements.

## Available Models Overview

### Model Categories

| Provider | Models | Context Window | Speed | Cost          |
|----------|--------|----------------|-------|---------------|
| **OpenAI** | GPT-4o, GPT-4 | 128k, 8k | Medium | High          |
| **Microsoft** | Phi-4 | 100k | Fast | Free          |
| **Qwen** | 7B, 14B, 32B, 72B | 32k-128k | Varies | Free/Low      |
| **OpenRouter** | Various | Varies | Fast | Free versions |

## Detailed Model Specifications

### OpenAI Models

#### GPT-4o (Recommended for Production)
```bash
python src/main.py --model openai --model-name gpt-4o
```

#### GPT-4
```bash
python src/main.py --model openai --model-name gpt-4
```

**Differences from GPT-4o:**

- Smaller context window (8k tokens)
- Slightly lower speed
- Same accuracy level
- Higher cost per token

### Open Source Models

#### Microsoft Phi-4 (Best Value)
```bash
python src/main.py --model hf --model-name microsoft/phi-4
```

#### Qwen 2.5 Series

##### Qwen2.5-7B (Lightweight)
```bash
python src/main.py --model hf --model-name Qwen/Qwen2.5-7B
```

##### Qwen2.5-32B (Balanced)
```bash
python src/main.py --model hf --model-name Qwen/Qwen2.5-32B
```


##### Qwen2.5-72B (Via OpenRouter)
```bash
python src/main.py --model openrouter --model-name qwen/qwen-2.5-72b-instruct
```

### OpenRouter Cloud Models

Access various models through OpenRouter API:

```bash
# Qwen models
python src/main.py --model openrouter --model-name qwen/qwen-2.5-72b-instruct

# Claude models
python src/main.py --model openrouter --model-name anthropic/claude-3.5-sonnet

# Open source models
python src/main.py --model openrouter --model-name meta-llama/llama-3.1-70b-instruct
```

**Advantages:**

- No GPU required
- Access to latest models
- No cost, using the free versions
- Automatic scaling

## Advanced Model Configurations

### Multi-Model Pipeline
Run critical questions through multiple models:

```bash
#!/bin/bash
# High-stakes questions through multiple models

QUESTION_ID="18"  # Critical question

# Tier 1: Fast screening
python src/main.py --model hf --model-name microsoft/phi-4 \
  --questions $QUESTION_ID --output-dir results/tier1

# Tier 2: Accuracy check  
python src/main.py --model hf --model-name Qwen/Qwen2.5-32B \
  --questions $QUESTION_ID --output-dir results/tier2

# Tier 3: Final verification
python src/main.py --model openai --model-name gpt-4o \
  --questions $QUESTION_ID --verifier --output-dir results/tier3
```

## Next Steps

- Configure your chosen model: [Model Configuration](../models/configuration.md)
- Optimize for your use case: [RAG Strategies](rag-strategies.md)
- Set up prompts: [Prompt Engineering](prompts.md)
- Run evaluation: [Model Comparison](../scripts/compare-models.md)
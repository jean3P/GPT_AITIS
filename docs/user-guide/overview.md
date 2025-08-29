# User Guide Overview
The system helps answer questions like:

- "Is my stolen luggage covered by this policy?"
- "What's the maximum reimbursement for medical expenses abroad?"
- "Are trip cancellations due to illness covered?"

## Key Features

### 🤖 Multiple Model Support
- **OpenAI Models**: GPT-4, GPT-4o
- **Open Source Models**: Microsoft Phi-4, Qwen family (7B to 72B)
- **API Services**: OpenRouter for accessing various cloud models
- **Custom Models**: Easy integration of new models

### 📚 Advanced RAG Capabilities
- **Multiple Chunking Strategies**: Simple, semantic ans by sections.
- **Smart Retrieval**: Context-aware chunk selection
- **Complete Policy Mode**: Process entire documents when needed

### 🎯 Intelligent Analysis
- **Multi-language Support**: Handles English and Italian policies
- **Verification System**: Double-checks and corrects responses
- **Relevance Filtering**: Skips unrelated queries efficiently

### 📊 Comprehensive Evaluation
- **Multiple Metrics**: Accuracy, precision, recall, F1 scores
- **Model Comparison**: Side-by-side performance analysis

## Core Workflow

### 1. Input Processing
The system takes:

- **Questions**: User queries about insurance coverage
- **Policies**: PDF or text documents containing policy terms
- **Configuration**: Model selection, RAG strategy, prompts

### 2. Document Processing
Depending on the mode:

- **RAG Mode**: Chunks documents intelligently and indexes them
- **Complete Policy Mode**: Loads entire documents for models with large context windows

### 3. Query Analysis

- Extracts persona information (who's claiming, where the event occurred)
- Checks query relevance to avoid processing unrelated questions
- Retrieves the most relevant policy sections

### 4. Coverage Determination
The LLM analyzes the context and determines:

- **Eligibility**: Yes / No - Unrelated event / No - condition(s) not met
- **Justification**: Exact policy text supporting the decision
- **Payment Amount**: Specific coverage limits if applicable

### 5. Verification (Optional)
A second pass that:

- Reviews the initial determination
- Checks for hallucinations or errors
- Corrects mistakes in eligibility or quotes

### 6. Output Generation
Results are formatted as structured JSON:
```json
{
  "policy_id": "18",
  "questions": [{
    "request_id": "1",
    "question": "My luggage was lost at the airport...",
    "outcome": "Yes",
    "outcome_justification": "In the event that the air carrier fails to deliver...",
    "payment_justification": "Option 1 € 150,00 Option 2 € 350,00"
  }]
}
```

## Operating Modes

### RAG Mode (Default)
Best for:

- Standard analysis with good performance
- Limited GPU memory
- Quick processing
- Specific question answering

Configuration:
```bash
python src/main.py --model hf --k 3 --rag-strategy semantic
```

### Complete Policy Mode
Best for:

- Complex multi-reference queries
- Models with large context windows (100k+ tokens)
- Maximum accuracy requirements
- Cross-document reasoning

Configuration:
```bash
python src/main.py --model hf --complete-policy
```

### Batch Processing
Processes multiple policies efficiently:
```bash
python src/main.py --batch --questions "1,2,3,4,5"
```

## Key Concepts

### Chunking Strategies
Different approaches to splitting documents:

| Strategy | Best For | Characteristics |
|----------|----------|-----------------|
| **Simple** | Quick testing | Paragraph-based splitting |
| **Semantic** | Coherent content | Groups related sentences |
| **Section** | Structured docs | Preserves legal sections |
| **Smart Size** | Adaptive chunking | Adjusts based on content importance |

### Persona Extraction
Identifies key information about the claim:

- **Policy User**: Who owns the policy
- **Affected Person**: Who experienced the event
- **Location**: Where the event occurred
- **Relationships**: How people are connected
- **Coverage Status**: Whether affected persons are covered

### Prompt Templates
Pre-configured prompts optimized for different models:

- `standard`: Basic coverage determination
- `precise_v3`: Enhanced accuracy with strict formatting
- `precise_v4`: Latest version with better edge case handling
- Model-specific variants (e.g., `precise_v3_qwen`)

## Typical Use Cases

### 1. Single Policy Analysis
Analyze one policy with specific questions:
```bash
python src/main.py --policy-id 18 --questions "1,5,10"
```

### 2. Comparative Model Testing
Compare different models on the same dataset:
```bash
# Run with Phi-4
python src/main.py --model hf --model-name microsoft/phi-4

# Run with Qwen
python src/main.py --model hf --model-name Qwen/Qwen2.5-32B
```

### 3. Production Pipeline
Full pipeline with verification:
```bash
python src/main.py \
  --batch \
  --model openai \
  --model-name gpt-4o \
  --rag-strategy semantic \
  --k 5 \
  --verifier \
  --verifier-iterations 1
```

### 4. Evaluation Workflow
After running analysis:
```bash
# Evaluate results
python scripts/evaluate_results.py results/model_output.json
```

## Best Practices

### 1. Model Selection
- **For accuracy**: GPT-4o or Qwen-72B
- **For speed**: Phi-4 or Qwen-7B
- **For cost efficiency**: Open source models via HuggingFace

### 2. RAG Configuration
- Start with `k=3` chunks
- Use semantic chunking for general queries
- Switch to section chunking for structured policies
- Enable complete policy mode for complex cases

### 3. Quality Assurance
- Always run evaluation after analysis
- Use verification for critical applications
- Monitor for hallucinations in quotes
- Check edge cases (e.g., family claims, exclusions)

### 4. Performance Optimization
- Batch process when analyzing multiple policies
- Cache embeddings for repeated analyses
- Use appropriate GPU resources
- Consider OpenRouter for cloud scaling

## Next Steps

- **[Running Analysis](running-analysis.md)**: Detailed guide on executing analyses
- **[Model Selection](model-selection.md)**: Choosing the right model for your needs
- **[RAG Strategies](rag-strategies.md)**: Deep dive into chunking approaches
- **[Prompt Engineering](prompts.md)**: Customizing prompts for better results
- **[Result Verification](verification.md)**: Ensuring accuracy with verification

## Getting Help

- Check the [FAQ](../troubleshooting/faq.md) for common questions
- Review [Troubleshooting](../troubleshooting/common.md) for issues
- See [Examples](../examples/basic.md) for practical workflows
- Consult [API Reference](../api/core.md) for technical details
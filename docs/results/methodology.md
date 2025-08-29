# GPT_AITIS Evaluation Results and Methodology

This document presents comprehensive evaluation results and methodology from various experiments with the GPT_AITIS system, including comparisons between different prompt versions, models, and RAG strategies. The evaluation experiments were conducted on policies 18 and 20 initially, then expanded to include policies 10 and 19, with different configurations to assess the system's performance in insurance coverage determination tasks.

## Summary

The evaluation followed a systematic progression through four phases:

1. **Prompt Engineering Phase** (Policies 18 & 20, Phi-4 with Simple RAG):
    - V1 Baseline: Established 75% accuracy benchmark
    - V2 Optimization: Improved to 80% accuracy 
    - V3 Testing: Alternative formulation (77.5%)
    - V4 Validation: Added deterministic improvements (80%)

2. **Parameter Optimization**:
    - k=3 vs k=5: Confirmed k=3 optimal (60% drop with k=5)
    - Large-scale testing: 72 questions achieved 84.72% accuracy

3. **Relevance Filtering Analysis**:
    - Baseline without filter: 86.96% accuracy (best)
    - With relevance filter: 85.51% accuracy
    - Modified prompt with filter: 63.77% accuracy

4. **Multi-Model & RAG Strategy Comparison** (Policies 10, 18, 19, 20):
    - Tested 9 RAG strategies on Phi-4 and Qwen3-235b-a22b
    - Complete Policy processing achieved best results
    - Qwen3-235b-a22b: 79.6% accuracy, Phi-4: 75.9% accuracy

## Evaluation Phases Overview

The evaluation of GPT_AITIS progressed through four distinct phases, each building upon the insights from the previous phase to systematically improve the system's performance in insurance coverage determination.

### Phase 1: Prompt Engineering (Policies 18 & 20)

This foundational phase focused on iterative prompt refinement using a limited dataset of 20-40 questions across two policies. All experiments used the Phi-4 model with simple RAG strategy (k=3).

**Key Activities:**

- Established baseline performance with Prompt V1
- Iteratively refined prompts through V2, V3, and V4
- Tested on Nobis Baggage Loss (Policy 18) and Travel Cancellation (Policy 20)

**Results:**

- **V1 Baseline**: 75% accuracy - established initial benchmark
- **V2 Optimization**: 80% accuracy - significant 5% improvement through better prompt structure
- **V3 Testing**: 77.5% accuracy - alternative minimal format showed no advantage
- **V4 Validation**: 80% accuracy - confirmed V2's effectiveness with added validation

**Finding:** Prompt V2 emerged as the optimal formulation, balancing clarity with effectiveness.

### Phase 2: Parameter Optimization

With the optimal prompt identified, this phase explored key parameter variations to understand their impact on performance.

**Key Activities:**

- Compared different k values (number of retrieved chunks)
- Expanded testing to 72 questions for statistical robustness
- Maintained focus on policies 18 and 20

**Results:**

- **k=3 vs k=5**: Performance dropped from 80% to 60% with k=5
- **Large-scale testing (72 questions)**: Achieved 84.72% accuracy with k=3

**Finding:** More context (higher k) actually degraded performance, suggesting that focused, relevant chunks are more valuable than broader context. The system showed improved performance with larger test sets.

### Phase 3: Relevance Filtering Analysis

This phase investigated whether pre-filtering irrelevant chunks could improve the system's accuracy by reducing noise in the retrieval process.

**Key Activities:**

- Tested three configurations on 69 questions
- Compared baseline, standard filtering, and modified prompt with filtering
- Continued using policies 18 and 20

**Results:**

- **Baseline without filter**: 86.96% accuracy (best overall result)
- **With relevance filter**: 85.51% accuracy - slight degradation
- **Modified prompt with filter**: 63.77% accuracy - significant drop

**Finding:** The model inherently handles irrelevant content well without explicit filtering. Adding filtering mechanisms actually decreased performance, possibly by removing marginally relevant context.

### Phase 4: Multi-Model & RAG Strategy Comparison

The final phase expanded evaluation to the mature system, testing multiple models and RAG strategies across a broader set of policies.

**Key Activities:**

- Expanded to 4 policies (10, 18, 19, 20) covering different insurance types
- Tested 9 different RAG strategies
- Compared Phi-4 vs Qwen3-235b-a22b models
- Used 27 carefully selected questions (excluding "Maybe" outcomes)

**Results:**

- **Complete Policy processing**: Best for both models
  - Phi-4: 75.9% accuracy, 0.6 IoU
  - Qwen3-235b-a22b: 79.6% accuracy, 0.7 IoU
- **Best chunking strategies**:
  - Phi-4: Section-based (75% accuracy)
  - Qwen3-235b-a22b: Semantic with k=3 (77.8% accuracy)

**Finding:** Complete policy processing achieves best results despite computational costs. When chunking is necessary, model-specific strategies optimize performance. Qwen3-235b-a22b demonstrated superior outcome prediction while Phi-4 excelled at justification quality.

---

These four phases represent a systematic approach to optimizing the GPT_AITIS system, moving from basic prompt engineering to comprehensive strategy evaluation. The progression from 75% to 86.96% accuracy (and ultimately to 79.6% with the best model configuration) demonstrates the value of methodical experimentation and iterative refinement in developing AI systems for complex domain-specific tasks such as insurance policy analysis.

## Experimental Timeline and Evolution

The evaluation of GPT_AITIS involved a series of progressive experiments that evolved from initial prompt engineering tests on two policies (18 and 20) with limited ground truth data to comprehensive RAG strategy comparisons using four policies with the mature system. 

**Important Note:** All experiments from Prompt V1 through Relevance Filtering were conducted using the **Phi-4 model with simple RAG strategy (k=3)** unless otherwise specified.

### Summary of All Experiments

| Experiment | Policies | Questions | Best Accuracy | Best IoU | Key Finding |
|------------|----------|-----------|---------------|----------|-------------|
| **Prompt V1 Baseline** | 18, 20 | 20 | 75.00% | 0.802 | Initial baseline established |
| **Prompt V2 Tests** | 18, 20 | 40 | 80.00% | 0.843 | Significant improvement over V1 |
| **Prompt V3 Test** | 18, 20 | 40 | 77.50% | 0.837 | No advantage over V2 |
| **k Parameter Test** | 18, 20 | 40 | 60.00% | 0.629 | k=5 degrades performance |
| **Prompt V4 Test** | 18, 20 | 40 | 80.00% | 0.837 | Matches V2 performance |
| **Large-Scale V2** | 18, 20 | 72 | 84.72% | 0.866 | Best overall performance |
| **Relevance Filtering** | 18, 20 | 69 | 86.96% | 0.876 | No filter performs best |
| **RAG Strategies** | 10, 18, 19, 20 | 27† | 79.60% | 0.7 | Complete policy wins |

† Questions 2,3,4,5,6,7,8,9,10,11,12,14,16,17,18,19,20,21,22,23,24,25,26,29,30,31,33 (excluded questions with "Maybe" outcomes)

## Prompt Version Comparisons (Phi-4 Model with Simple RAG)

**Note:** All experiments in this section excluded questions with "Maybe" outcomes from the ground truth to ensure clear binary classification. The actual datasets contained questions with "Yes", "No - Unrelated event", "No - condition(s) not met", and "Maybe" outcomes, but "Maybe" cases were filtered out for evaluation clarity.

### Prompt Precise V1 - Policies 18 & 20 (Baseline)

This initial experiment tested the baseline Prompt Precise V1 on two policies (18: Nobis Baggage Loss and 20: Nobis Travel Cancellation) with 20 questions to establish performance benchmarks.

#### Best Result: 20 Questions

| Metric | Result |
|--------|--------|
| **Predicted Outcome Accuracy** | **75.00%** |
| Justification Outcome (IoU) | 0.8024 |

**Confusion Matrix:**

| Actual \ Predicted | Yes | No - Unrelated | No - Conditions | Total |
|-------------------|-----|----------------|-----------------|-------|
| Yes | 2 | 0 | 3 | 5 |
| No - Unrelated event | 2 | 27 | 0 | 29 |
| No - conditions not met | 2 | 2 | 1 | 5 |
| **Total** | 6 | 29 | 4 | 39 |

**Performance by Category:**

| Category | Precision | Recall | F1-Score |
|----------|-----------|---------|----------|
| Yes | 0.333 | 0.400 | 0.364 |
| No - Unrelated event | 0.900 | 0.931 | 0.915 |
| No - conditions not met | 0.250 | 0.200 | 0.222 |
| **Weighted Average** | 0.744 | 0.769 | 0.756 |

The confusion matrix revealed particular weakness in distinguishing between "Yes" cases (precision 0.333) and "No - conditions not met" cases (precision 0.250), demonstrating the need for more precise prompt engineering to improve classification accuracy.

### Prompt Precise V2 - Policies 18 & 20 (Optimization)

Multiple experiments with Prompt Precise V2 explored different configurations to improve upon V1's baseline, continuing with the same two policies.

#### Best Result: 40 Questions

| Metric | Result |
|--------|--------|
| **Predicted Outcome Accuracy** | **80.00%** |
| Justification Outcome (IoU) | 0.8434 |

**Performance by Category:**

| Category | Precision | Recall | F1-Score |
|----------|-----------|---------|----------|
| Yes | 0.500 | 0.800 | 0.615 |
| No - Unrelated event | 0.900 | 0.931 | 0.915 |
| No - conditions not met | 0.500 | 0.167 | 0.250 |
| **Weighted Average** | 0.790 | 0.800 | 0.778 |

The performance metrics demonstrated better handling of positive cases, with recall improving from 0.40 to 0.80 for "Yes" outcomes, while maintaining strong performance on "No - Unrelated event" classifications.

#### Experiment 3: 40 Questions with k=5

| Metric | Result |
|--------|--------|
| **Predicted Outcome Accuracy** | **60.00%** |
| Justification Outcome (IoU) | 0.6293 |

> **⚠️ Performance Drop with k=5**  
> Increasing the number of retrieved chunks (k) from 3 to 5 resulted in a significant performance drop, suggesting that more context doesn't always improve results. This counterintuitive finding emphasized the importance of quality over quantity in RAG systems.

### Prompt Precise V3 - Policies 18 & 20

Prompt Precise V3 was tested with the same two policies and 40 questions to explore alternative prompt formulations for reduced hallucination.

#### 40 Questions

| Metric | Result |
|--------|--------|
| **Predicted Outcome Accuracy** | **77.50%** |
| Justification Outcome (IoU) | 0.8367 |

The performance patterns suggested that while V3's minimal format approach was valid, it did not provide significant advantages over the optimized V2 prompt in this limited testing scenario.

### Prompt Precise V4 - Policies 18 & 20 (Validation)

Prompt Precise V4 introduced deterministic improvements and timing validation checks, tested on policies 18 and 20 with 40 questions.

#### 40 Questions

| Metric | Result |
|--------|--------|
| **Predicted Outcome Accuracy** | **80.00%** |
| Justification Outcome (IoU) | 0.8367 |

The results matched V2's best performance while adding enhanced validation mechanisms.

### Large-Scale Evaluation: 72 Questions

Scaling up to 72 questions while maintaining the same two policies (18 and 20) provided a more robust evaluation of Prompt Precise V2's performance on a larger dataset.

#### Prompt Precise V2

| Metric | Result |
|--------|--------|
| **Predicted Outcome Accuracy** | **84.72%** |
| Justification Outcome (IoU) | 0.8662 |

**Performance by Category:**

| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|---------|----------|---------|
| Yes | 0.500 | 0.857 | 0.632 | 7 |
| No - Unrelated event | 0.930 | 0.946 | 0.938 | 56 |
| No - conditions not met | 0.667 | 0.222 | 0.333 | 9 |
| **Weighted Average** | 0.855 | 0.847 | 0.833 | 72 |

The system achieved its best overall accuracy, validating the prompt's effectiveness at scale despite persistent challenges with "No - conditions not met" classifications.

## Relevance Filter Impact Analysis

Three experiments compared relevance filtering approaches on 69 questions using policies 18 and 20 to assess pre-filtering effectiveness.

### Comparison of Relevance Filtering Approaches (69 Questions)

| Configuration | Accuracy | IoU |
|--------------|----------|-----|
| **Pv2 (No Filter)** | **86.96%** | **0.8757** |
| Pv2 + Relevance Filter | 85.51% | 0.8660 |
| Modified Pv2 + Relevance Filter | 63.77% | 0.6626 |

> **📝 Key Finding**  
> The baseline Pv2 without relevance filtering achieved the best performance. Adding a relevance filter slightly decreased performance, while modifying the prompt structure with the filter led to significant performance degradation. The model already handled irrelevant content well without explicit filtering.

### Detailed Results: Pv2 with Relevance Filter v2

| Metric | Result |
|--------|--------|
| **Predicted Outcome Accuracy** | **85.51%** |
| Justification Outcome (IoU) | 0.8660 |

### Detailed Results: Pv2 without Relevance Filter

| Metric | Result |
|--------|--------|
| **Predicted Outcome Accuracy** | **86.96%** |
| Justification Outcome (IoU) | 0.8757 |

## Policy-Level Performance Analysis

### IoU Scores by Policy

| Policy ID | With Cleanup | Without Cleanup |
|-----------|--------------|-----------------|
| 18 | 0.92 | 0.92 |
| 20 | 0.79 | 0.80 |

**Note:** The "cleanup" refers to a post-processing script that normalizes the extracted justification and payment text from the model outputs to better match the formatting of the original PDF text. This cleaning process removes extra whitespace, standardizes punctuation, and aligns formatting differences between the model's quoted text and the source document.

> **ℹ️ Observation**  
> Policy 18 consistently shows higher IoU scores compared to Policy 20, suggesting it may have clearer or more structured content that's easier for the system to process. The minimal difference between cleaned and uncleaned scores indicates that the model is already extracting text that closely matches the original PDF format.

## Model Comparison: Phi-4 vs Qwen3-235b

The final comprehensive evaluation expanded testing to four policies and compared nine different RAG strategies across two models using the mature system.

### Tested Policies
- **Policy 10**: AXA 20220316 DIP AGGIUNTIVO ALI@TOP
- **Policy 18**: Nobis Baggage Loss
- **Policy 19**: Nobis Flight Delay
- **Policy 20**: Nobis Travel Cancellation (Hospital)

### Overall Performance (27 Questions)

Testing was conducted on 27 carefully selected questions (2,3,4,5,6,7,8,9,10,11,12,14,16,17,18,19,20,21,22,23,24,25,26,29,30,31,33) that excluded any ground truth cases with "Maybe" outcomes to ensure clear binary classification.

| Metric | Phi-4 | Qwen3-235b-a22b |
|--------|-------|-----------------|
| **Predicted Outcome Accuracy** | 60.19% | **68.52%** |
| **Justification Outcome (IoU)** | **0.5995** | 0.0648 |

> **⚠️ Trade-off Alert**  
> While Qwen3-235b-a22b achieved higher accuracy in outcome prediction, Phi-4 demonstrated significantly better performance in justification quality (IoU), suggesting different strengths for each model.

### Performance by Category

| Outcome Category | Phi-4 (P/R/F1) | Qwen3-235b-a22b (P/R/F1) |
|-----------------|----------------|--------------------------|
| Yes | 0.500/0.250/0.333 | 0.571/0.333/0.421 |
| No - Unrelated | 0.737/0.824/0.778 | 0.726/0.897/0.803 |
| No - Conditions | 0.300/0.188/0.231 | 0.500/0.312/0.385 |

## RAG Strategy Comparison

### Performance Across Different RAG Strategies

| RAG Strategy | Phi-4 (Acc/IoU) | Qwen3-235b-a22b (Acc/IoU) |
|--------------|-----------------|--------------------------|
| Simple (k=3) | 60.2% / 0.6 | 68.5% / 0.1 |
| By Section (k=3) | 75.0% / 0.6 | 68.5% / 0.5 |
| Smart Size (k=3) | 74.1% / 0.6 | 75.0% / 0.6 |
| Smart Size (k=5) | 71.3% / 0.6 | 72.2% / 0.6 |
| Smart Size (k=7) | 71.2% / 0.6 | 73.2% / 0.6 |
| Semantic (k=3) | 71.3% / 0.6 | 77.8% / 0.7 |
| Semantic (k=5) | 69.5% / 0.5 | 74.1% / 0.7 |
| Semantic (k=7) | 68.5% / 0.6 | 74.1% / 0.7 |
| **Complete Policy** | **75.9% / 0.6** | **79.6% / 0.7** |

> **✅ Best Performers**  
> - **Phi-4**: Complete Policy (75.9% accuracy) and By Section (75.0% accuracy)
> - **Qwen3-235b-a22b**: Complete Policy (79.6% accuracy) and Semantic k=3 (77.8% accuracy)

Complete Policy processing achieved the best results for both models, with Qwen3-235b-a22b reaching 79.6% accuracy and 0.7 IoU. Among chunking strategies, Section-based chunking performed best for Phi-4 (75.0% accuracy), while Semantic chunking with k=3 showed optimal results for Qwen3-235b-a22b (77.8% accuracy, 0.7 IoU).

## Key Insights and Conclusions

1. **Prompt Evolution**: Progressive improvements from V1 to V4, with V2 showing the most consistent performance. The mature system performs best with Prompt Precise V2.

2. **Optimal k Value**: k=3 consistently outperforms higher values (k=5, k=7), suggesting that focused context is more valuable than broader context.

3. **Relevance Filtering**: Contrary to expectations, adding relevance filtering slightly decreased performance, indicating the model already handles irrelevant information well.

4. **Model Trade-offs**:
    - Phi-4: Better justification quality (higher IoU)
    - Qwen3-235b: Better outcome prediction accuracy
    - Qwen3-235b-a22b with Complete Policy processing provides the best overall results (79.6% accuracy)

5. **RAG Strategy Impact**: Complete policy processing achieves best results for both models, but comes with higher computational costs. When chunking is necessary, model-specific strategies work best.

6. **System Maturity**: Performance improved from 75% (2 policies) to 86.96% (2 policies, larger dataset) as the system matured and ground truth expanded.

7. **Policy Coverage**: The system successfully handles diverse insurance types including travel cancellation, baggage loss, flight delay, and comprehensive travel insurance.

> **📊 Metrics Explanation**

 - **IoU**: Intersection over Union - measures word-level overlap between predicted and ground truth justifications
 - **Accuracy**: Percentage of correct outcome predictions
 - **Precision**: True positives / (True positives + False positives)
 - **Recall**: True positives / (True positives + False negatives)
 - **F1-Score**: Harmonic mean of precision and recall
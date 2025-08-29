# Evaluation Results

This page presents comprehensive evaluation results from various experiments with the GPT_AITIS system, including comparisons between different prompt versions, models, and RAG strategies.

## Overview

The evaluation experiments were conducted on policies 18 and 20 with different configurations to assess the system's performance in insurance coverage determination tasks.

## Prompt Version Comparisons

### Prompt Precise V1 - Policies 18 & 20

#### Experiment 1: 20 Questions (14-05-2025)

| Metric | Result |
|--------|--------|
| **Predicted Outcome Accuracy** | **75.00%** |
| Justification Outcome (IoU) | 0.7799 |

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

#### Experiment 2: 20 Questions (15-05-2025, 08:40)

| Metric | Result |
|--------|--------|
| **Predicted Outcome Accuracy** | **75.00%** |
| Justification Outcome (IoU) | 0.8024 |

### Prompt Precise V2 - Policies 18 & 20

#### Experiment 1: 40 Questions (15-05-2025, 11:13)

| Metric | Result |
|--------|--------|
| **Predicted Outcome Accuracy** | **77.50%** |
| Justification Outcome (IoU) | 0.8352 |

**Performance by Category:**

| Category | Precision | Recall | F1-Score |
|----------|-----------|---------|----------|
| Yes | 0.429 | 0.600 | 0.500 |
| No - Unrelated event | 0.900 | 0.931 | 0.915 |
| No - conditions not met | 0.333 | 0.167 | 0.222 |
| **Weighted Average** | 0.756 | 0.775 | 0.759 |

#### Experiment 2: 40 Questions (15-05-2025, 12:52) - Best V2 Result

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

#### Experiment 3: 40 Questions with k=5 (15-05-2025, 15:31)

| Metric | Result |
|--------|--------|
| **Predicted Outcome Accuracy** | **60.00%** |
| Justification Outcome (IoU) | 0.6293 |

!!! warning "Performance Drop with k=5"
    Increasing the number of retrieved chunks (k) from 3 to 5 resulted in a significant performance drop, suggesting that more context doesn't always improve results.

### Prompt Precise V3 - Policies 18 & 20

#### 40 Questions (15-05-2025, 12:46)

| Metric | Result |
|--------|--------|
| **Predicted Outcome Accuracy** | **77.50%** |
| Justification Outcome (IoU) | 0.8367 |

### Prompt Precise V4 - Policies 18 & 20

#### 40 Questions (21-05-2025, 16:19)

| Metric | Result |
|--------|--------|
| **Predicted Outcome Accuracy** | **80.00%** |
| Justification Outcome (IoU) | 0.8367 |

### Large-Scale Evaluation: 72 Questions

#### Prompt Precise V2 (22-05-2025, 17:30)

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

## Relevance Filter Impact Analysis

### Comparison of Relevance Filtering Approaches (69 Questions)

| Configuration | Accuracy | IoU |
|--------------|----------|-----|
| **Pv2 (No Filter)** | **86.96%** | **0.8757** |
| Pv2 + Relevance Filter | 85.51% | 0.8660 |
| Modified Pv2 + Relevance Filter | 63.77% | 0.6626 |

!!! note "Key Finding"
    The baseline Pv2 without relevance filtering achieved the best performance. Adding a relevance filter slightly decreased performance, while modifying the prompt structure with the filter led to significant performance degradation.

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

## IoU Scores by Policy

### Policy-Level Performance Comparison

| Policy ID | With Cleanup | Without Cleanup |
|-----------|--------------|-----------------|
| 18 | 0.92 | 0.92 |
| 20 | 0.79 | 0.80 |

!!! info "Observation"
    Policy 18 consistently shows higher IoU scores compared to Policy 20, suggesting it may have clearer or more structured content that's easier for the system to process.

## Model Comparison: Phi-4 vs Qwen3-235b

### Overall Performance (27 Questions)

| Metric | Phi-4 | Qwen3-235b-a22b |
|--------|-------|-----------------|
| **Predicted Outcome Accuracy** | 60.19% | **68.52%** |
| **Justification Outcome (IoU)** | **0.5995** | 0.0648 |

!!! warning "Trade-off Alert"
    While Qwen3-235b achieved higher accuracy in outcome prediction, Phi-4 demonstrated significantly better performance in justification quality (IoU), suggesting different strengths for each model.

### Performance by Category

| Outcome Category | Phi-4 (P/R/F1) | Qwen3-235b (P/R/F1) |
|-----------------|----------------|---------------------|
| Yes | 0.500/0.250/0.333 | 0.571/0.333/0.421 |
| No - Unrelated | 0.737/0.824/0.778 | 0.726/0.897/0.803 |
| No - Conditions | 0.300/0.188/0.231 | 0.500/0.312/0.385 |

## RAG Strategy Comparison

### Performance Across Different RAG Strategies

| RAG Strategy | Phi-4 (Acc/IoU) | Qwen3-235b (Acc/IoU) |
|--------------|-----------------|----------------------|
| Simple (k=3) | 60.2% / 0.6 | 68.5% / 0.1 |
| By Section (k=3) | 75.0% / 0.6 | 68.5% / 0.5 |
| Smart Size (k=3) | 74.1% / 0.6 | 75.0% / 0.6 |
| Smart Size (k=5) | 71.3% / 0.6 | 72.2% / 0.6 |
| Smart Size (k=7) | 71.2% / 0.6 | 73.2% / 0.6 |
| Semantic (k=3) | 71.3% / 0.6 | 77.8% / 0.7 |
| Semantic (k=5) | 69.5% / 0.5 | 74.1% / 0.7 |
| Semantic (k=7) | 68.5% / 0.6 | 74.1% / 0.7 |
| **Complete Policy** | **75.9% / 0.6** | **79.6% / 0.7** |

!!! success "Best Performers"
    - **Phi-4**: Complete Policy (75.9% accuracy) and By Section (75.0% accuracy)
    - **Qwen3-235b**: Complete Policy (79.6% accuracy) and Semantic k=3 (77.8% accuracy)

## Key Insights

1. **Prompt Evolution**: Progressive improvements from V1 to V4, with V2 showing the most consistent performance.

2. **Optimal k Value**: k=3 consistently outperforms higher values (k=5, k=7), suggesting that focused context is more valuable than broader context.

3. **Relevance Filtering**: Contrary to expectations, adding relevance filtering slightly decreased performance, indicating the model already handles irrelevant information well.

4. **Model Trade-offs**: 
   - Phi-4: Better justification quality (higher IoU)
   - Qwen3-235b: Better outcome prediction accuracy

5. **RAG Strategy Impact**: Complete policy processing achieves best results for both models, but comes with higher computational costs.

!!! note "Metrics Explanation"
    - **IoU**: Intersection over Union - measures word-level overlap between predicted and ground truth justifications
    - **Accuracy**: Percentage of correct outcome predictions
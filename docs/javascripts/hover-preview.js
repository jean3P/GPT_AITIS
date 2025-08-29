document.addEventListener('DOMContentLoaded', function() {
  // Define preview data for all experiments
  const previews = {
    // Prompt V1 Experiments
    'prompt-precise-v1---policies-18--20': {
      title: 'Prompt Precise V1 Results Summary',
      content: `
        <table class="preview-table">
          <tr><td colspan="2"><b>Two Experiments on Policies 18 & 20</b></td></tr>
          <tr><td><b>Exp 1 (May 14)</b></td><td>75.00% Acc, 0.7799 IoU</td></tr>
          <tr><td><b>Exp 2 (May 15)</b></td><td>75.00% Acc, 0.8024 IoU</td></tr>
          <tr><td colspan="2" class="weak-point">⚠️ Weak on "Yes" (0.333 precision)</td></tr>
        </table>
      `
    },
    'experiment-1-20-questions-14-05-2025': {
      title: 'Prompt V1 - Experiment 1 Details',
      content: `
        <table class="preview-table">
          <tr><td><b>Accuracy</b></td><td>75.00%</td></tr>
          <tr><td><b>IoU</b></td><td>0.7799</td></tr>
          <tr><td colspan="2"><b>Confusion Matrix</b></td></tr>
          <tr><td>Yes</td><td>2/6 predicted (33.3%)</td></tr>
          <tr><td>No-Unrelated</td><td>27/29 correct (93.1%)</td></tr>
          <tr><td>No-Conditions</td><td>1/4 predicted (25.0%)</td></tr>
          <tr><td><b>F1 Weighted</b></td><td>0.7557</td></tr>
        </table>
      `
    },

    // Prompt V2 Experiments
    'prompt-precise-v2---policies-18--20': {
      title: 'Prompt Precise V2 Results Summary',
      content: `
        <table class="preview-table">
          <tr><td colspan="2"><b>Three Experiments on Policies 18 & 20</b></td></tr>
          <tr><td><b>Exp 1 (40 Q)</b></td><td>77.50% Acc, 0.8352 IoU</td></tr>
          <tr><td><b>Exp 2 (40 Q)</b></td><td class="best">80.00% Acc, 0.8434 IoU ✓</td></tr>
          <tr><td><b>Exp 3 (k=5)</b></td><td class="poor">60.00% Acc, 0.6293 IoU ⚠️</td></tr>
          <tr><td colspan="2" class="improvement">📈 Yes recall: 0.40→0.80</td></tr>
        </table>
      `
    },
    'experiment-2-40-questions-15-05-2025-1252---best-v2-result': {
      title: 'Prompt V2 - Best Result Details',
      content: `
        <table class="preview-table">
          <tr><td><b>Accuracy</b></td><td class="best">80.00%</td></tr>
          <tr><td><b>IoU</b></td><td class="best">0.8434</td></tr>
          <tr><td colspan="2"><b>Performance by Category</b></td></tr>
          <tr><td>Yes</td><td>P: 0.500, R: 0.800, F1: 0.615</td></tr>
          <tr><td>No-Unrelated</td><td>P: 0.900, R: 0.931, F1: 0.915</td></tr>
          <tr><td>No-Conditions</td><td>P: 0.500, R: 0.167, F1: 0.250</td></tr>
          <tr><td><b>F1 Weighted</b></td><td>0.7780</td></tr>
        </table>
      `
    },

    // Prompt V3
    'prompt-precise-v3---policies-18--20': {
      title: 'Prompt Precise V3 Results',
      content: `
        <table class="preview-table">
          <tr><td><b>Accuracy</b></td><td>77.50%</td></tr>
          <tr><td><b>IoU</b></td><td>0.8367</td></tr>
          <tr><td><b>Questions</b></td><td>40</td></tr>
          <tr><td><b>F1 Weighted</b></td><td>0.7594</td></tr>
          <tr><td colspan="2" class="note">No advantage over V2</td></tr>
        </table>
      `
    },

    // k=5 Experiment
    'experiment-3-40-questions-with-k5-15-05-2025-1531': {
      title: 'Impact of k=5 (vs k=3)',
      content: `
        <table class="preview-table">
          <tr><td><b>Accuracy</b></td><td class="poor">60.00% (-20%)</td></tr>
          <tr><td><b>IoU</b></td><td class="poor">0.6293 (-0.21)</td></tr>
          <tr><td colspan="2"><b>Performance Drop</b></td></tr>
          <tr><td>Yes precision</td><td>0.250 (vs 0.500)</td></tr>
          <tr><td>No-Unrelated recall</td><td>0.690 (vs 0.931)</td></tr>
          <tr><td colspan="2" class="warning">⚠️ More context ≠ Better results</td></tr>
        </table>
      `
    },

    // Prompt V4
    'prompt-precise-v4---policies-18--20': {
      title: 'Prompt Precise V4 Results',
      content: `
        <table class="preview-table">
          <tr><td><b>Accuracy</b></td><td>80.00%</td></tr>
          <tr><td><b>IoU</b></td><td>0.8367</td></tr>
          <tr><td><b>Questions</b></td><td>40</td></tr>
          <tr><td colspan="2"><b>Policy-Level IoU</b></td></tr>
          <tr><td>Policy 18</td><td>0.92</td></tr>
          <tr><td>Policy 20</td><td>0.79</td></tr>
          <tr><td colspan="2" class="note">Matches V2 performance</td></tr>
        </table>
      `
    },

    // Large-Scale Evaluation
    'large-scale-evaluation-72-questions': {
      title: 'Large-Scale Evaluation (72 Questions)',
      content: `
        <table class="preview-table">
          <tr><td><b>Accuracy</b></td><td class="best">84.72%</td></tr>
          <tr><td><b>IoU</b></td><td class="best">0.8662</td></tr>
          <tr><td><b>F1 Weighted</b></td><td>0.8327</td></tr>
          <tr><td colspan="2"><b>Strong Performance</b></td></tr>
          <tr><td>No-Unrelated</td><td>F1: 0.938 (P: 0.930, R: 0.946)</td></tr>
          <tr><td>Yes</td><td>F1: 0.632 (P: 0.500, R: 0.857)</td></tr>
          <tr><td colspan="2" class="weak-point">⚠️ No-Conditions: F1 0.333</td></tr>
        </table>
      `
    },

    // Relevance Filter Analysis
    'relevance-filter-impact-analysis': {
      title: 'Relevance Filter Comparison (69 Q)',
      content: `
        <table class="preview-table">
          <tr><td><b>Configuration</b></td><td><b>Accuracy / IoU</b></td></tr>
          <tr><td>V2 No Filter</td><td class="best">86.96% / 0.8757 ✓</td></tr>
          <tr><td>V2 + Filter</td><td>85.51% / 0.8660</td></tr>
          <tr><td>Modified V2 + Filter</td><td class="poor">63.77% / 0.6626</td></tr>
          <tr><td colspan="2" class="finding">💡 Model handles irrelevance well without filter</td></tr>
        </table>
      `
    },

    // RAG Strategy Comparison
    'rag-strategy-comparison': {
      title: 'RAG Strategy Performance',
      content: `
        <table class="preview-table">
          <tr><td><b>Strategy</b></td><td><b>Phi-4</b></td><td><b>Qwen3</b></td></tr>
          <tr><td>Simple (k=3)</td><td>60.2% / 0.6</td><td>68.5% / 0.1</td></tr>
          <tr><td>By Section</td><td class="good">75.0% / 0.6</td><td>68.5% / 0.5</td></tr>
          <tr><td>Smart Size</td><td>74.1% / 0.6</td><td>75.0% / 0.6</td></tr>
          <tr><td>Semantic (k=3)</td><td>71.3% / 0.6</td><td class="good">77.8% / 0.7</td></tr>
          <tr><td>Complete Policy</td><td class="best">75.9% / 0.6</td><td class="best">79.6% / 0.7 ✓</td></tr>
          <tr><td colspan="3" class="note">Format: Accuracy / IoU</td></tr>
        </table>
      `
    },

    // Model Comparison
    'model-comparison-phi-4-vs-qwen3-235b': {
      title: 'Model Trade-offs (27 Questions)',
      content: `
        <table class="preview-table">
          <tr><td><b>Metric</b></td><td><b>Phi-4</b></td><td><b>Qwen3-235b</b></td></tr>
          <tr><td>Accuracy</td><td>60.19%</td><td class="better">68.52%</td></tr>
          <tr><td>IoU</td><td class="better">0.5995</td><td class="poor">0.0648</td></tr>
          <tr><td colspan="3"><b>F1-Scores by Category</b></td></tr>
          <tr><td>Yes</td><td>0.333</td><td>0.421</td></tr>
          <tr><td>No-Unrelated</td><td>0.778</td><td>0.803</td></tr>
          <tr><td>No-Conditions</td><td>0.231</td><td>0.385</td></tr>
          <tr><td colspan="3" class="finding">📊 Qwen3: Better accuracy, Phi-4: Better justifications</td></tr>
        </table>
      `
    }
  };

  // Function to create and position preview
  function showPreview(link, previewData) {
    // Create preview element
    const previewDiv = document.createElement('div');
    previewDiv.className = 'hover-preview';
    previewDiv.innerHTML = `
      <div class="hover-preview-title">${previewData.title}</div>
      <div class="hover-preview-content">${previewData.content}</div>
    `;

    document.body.appendChild(previewDiv);

    // Position the preview
    const rect = link.getBoundingClientRect();
    const previewRect = previewDiv.getBoundingClientRect();

    // Calculate position
    let top = rect.bottom + window.scrollY + 5;
    let left = rect.left + window.scrollX;

    // Adjust if preview goes off-screen
    if (left + previewRect.width > window.innerWidth) {
      left = window.innerWidth - previewRect.width - 10;
    }

    if (top + previewRect.height > window.innerHeight + window.scrollY) {
      top = rect.top + window.scrollY - previewRect.height - 5;
    }

    previewDiv.style.top = top + 'px';
    previewDiv.style.left = left + 'px';

    return previewDiv;
  }

  // Add hover handlers to all evaluation links
  let currentPreview = null;

  document.addEventListener('mouseover', function(e) {
    // Check if hovering over a link to evaluation.md
    const link = e.target.closest('a[href*="evaluation.md"]');
    if (!link || currentPreview) return;

    const href = link.getAttribute('href');
    const anchor = href.split('#')[1];

    if (anchor && previews[anchor]) {
      currentPreview = showPreview(link, previews[anchor]);
    }
  });

  document.addEventListener('mouseout', function(e) {
    // Remove preview when mouse leaves link or preview
    if (currentPreview && !e.relatedTarget?.closest('.hover-preview') &&
        !e.relatedTarget?.closest('a[href*="evaluation.md"]')) {
      currentPreview.remove();
      currentPreview = null;
    }
  });

  // Remove preview when clicking anywhere
  document.addEventListener('click', function() {
    if (currentPreview) {
      currentPreview.remove();
      currentPreview = null;
    }
  });
});

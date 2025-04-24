# Task 3: Training Considerations & Transfer Learning Strategy

## Freezing Scenarios

### 1. Entire Network Frozen
- **Use case:** If the sentence embeddings from the pre-trained model are rich enough and we only need a quick classification layer on top (e.g., for small or static datasets).
- **Implication:** 
  - Fast training, fewer parameters.
  - Limited task-specific learning capacity.
  - Not ideal for our multi-task case — both Task A and Task B benefit from representation tuning.

### 2. Freeze Only Transformer Backbone
- **Use case:** When the backbone is pre-trained on a large corpus and we're dealing with relatively simple tasks (e.g., sentiment analysis).
- **In our case:** This would still allow the task-specific heads to learn efficiently while preserving the general language understanding of the encoder.
- **Benefit:** Reduces overfitting and speeds up training.

### 3. Freeze One Task Head
- **Use case:** Suppose Task A is already well-performing, and we only want to improve Task B without affecting Task A.
- **Implication:** Encourages specialization in a sequential multi-task setup.

---

## Transfer Learning Strategy

### Pretrained Model
We used `distilbert-base-uncased`, a lighter version of BERT ideal for fast experimentation. In a production setting, something like `all-MiniLM-L6-v2` or `roberta-base` could offer better results.

### Freezing Strategy
- **Frozen:** Early transformer layers (which capture syntactic structure and general semantics)
- **Trainable:** Last few layers and task-specific heads (to allow task adaptation)

This approach maintains the general understanding of language while adapting the upper layers to the specific nuances of sentence classification and sentiment tasks.

---

## Final Thoughts

- Our model showed clear learning across tasks with decreasing loss values, confirming the effectiveness of this setup.
- Multi-task learning proved beneficial even with synthetic data — Task B had strong signal propagation from Task A's shared embeddings.
- The architecture is modular, easily extendable to more tasks or domains.

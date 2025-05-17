# Common Loss Functions in Machine Learning

## 1. MSE (Mean Squared Error)
- Used for regression problems
- Calculates the average squared difference between predicted and actual values
- Formula: MSE = (1/n) * Σ(y_pred - y_true)²
- Penalizes larger errors more heavily due to squaring
- Differentiable and convex, making optimization easier

## 2. BCE (Binary Cross-Entropy)
- Used for binary classification problems
- Measures the performance of a model whose output is a probability value between 0 and 1
- Formula: BCE = -(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
- Penalizes confident incorrect predictions more severely

## 3. BCEWithLogits (Binary Cross-Entropy With Logits)
- Combines sigmoid activation and BCE loss in one operation
- More numerically stable than separate sigmoid + BCE
- Takes raw logits (unbounded outputs) rather than probabilities
- Formula: Same as BCE but with sigmoid applied internally to logits
- More efficient and prevents potential underflow/overflow issues

## 4. Softmax Cross-Entropy
- Used for multi-class classification problems
- Softmax converts raw scores to probabilities, then cross-entropy measures the difference from true labels
- Formula: -Σ(y_true * log(softmax(y_pred)))
- Encourages the model to assign high probability to the correct class
- Typically implemented as a single operation for numerical stability

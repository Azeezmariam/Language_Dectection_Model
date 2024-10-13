
# Language Detection Model

## Overview

This project involves the development of a machine-learning model for language detection. The model focuses on detecting four languages: English, French, Kinyarwanda, and Kiswahili. The project is designed to help improve language classification accuracy, especially for the African region.

## Project Objectives

- Build a classification model capable of detecting and classifying four languages.
- Evaluate the model using various metrics such as Precision, Recall, F1-Score, and Specificity.

## Dataset

The dataset used for this project contains samples of text in the following four languages:
- English
- French
- Kinyarwanda
- Kiswahili

The dataset was split into training and testing sets to evaluate the model's performance. A total of 400 samples were used in the evaluation phase.

## Key Findings

### Detailed Discussion of Optimization Techniques

In this project, various optimization techniques were employed to improve the performance of the language detection model. The model was optimized using combinations of regularization (L1 and L2), optimizers (Adam, RMSprop), learning rate tuning, dropout, and early stopping. These techniques were chosen to enhance accuracy, prevent overfitting, and ensure that the model generalizes well to unseen data.

### 1. **L2 and L1 Regularization**

**Principle**:  
Both L1 (Lasso) and L2 (Ridge) regularization are methods that add a penalty to the model’s loss function to prevent overfitting. L2 regularization penalizes the square of the weights, while L1 regularization adds an absolute value of the weights as a penalty. L2 tends to shrink weights evenly, while L1 encourages sparsity by driving some weights to zero, effectively selecting features.

**Relevance to the Project**:  
In language detection, the model could overfit due to the small sample size and diverse language features. Regularization ensures that the model does not focus too heavily on individual features, instead learning more generalized patterns. The choice between L1 and L2 regularization depends on the characteristics of the dataset and the desired model behavior (sparsity vs. even weight distribution).

**Parameter Tuning**:  
The regularization strength (lambda, λ) was tuned to control the trade-off between bias and variance. For L2, a moderate value of λ was chosen to balance the penalty on large weights while allowing the model to maintain flexibility. For L1, a similar strategy was applied, but with the goal of promoting feature sparsity in certain cases. Both L1 and L2 were tested with λ values between 0.01 and 0.1, with the optimal value selected based on model performance.

### 2. **Adam Optimizer**

**Principle**:  
Adam (Adaptive Moment Estimation) is a widely used optimization algorithm that combines the advantages of two methods: momentum and RMSprop. It adjusts the learning rates of individual parameters based on their first and second moments (mean and variance of gradients). Adam's key benefit is that it automatically adjusts the learning rate throughout training, leading to faster and more efficient convergence.

**Relevance to the Project**:  
Adam’s adaptive nature makes it ideal for the language detection task, where the model must handle varying feature importance across different languages. Since text data can be sparse and noisy, Adam ensures more stable learning by dynamically adjusting the learning rate for each parameter.

**Parameter Tuning**:  
Adam’s primary parameters include the learning rate (`α`), β1 (decay rate for the first moment), and β2 (decay rate for the second moment). The default values for β1 (0.9) and β2 (0.999) were used, while the learning rate was tuned. A learning rate of **0.001** was chosen after experimenting with values between 0.0001 and 0.01, as it provided a good balance between fast convergence and stability.

### 3. **RMSprop Optimizer**

**Principle**:  
RMSprop is an adaptive learning rate optimizer that divides the learning rate by a moving average of the magnitude of recent gradients. This ensures that the learning rate is smaller for parameters with large gradients, allowing for more stable and efficient updates, particularly in tasks where gradient magnitudes can vary significantly.

**Relevance to the Project**:  
In language detection, RMSprop was particularly useful in handling large gradients that could occur when certain features (words, phrases) dominate. RMSprop helps balance learning by preventing drastic updates that could lead to unstable learning and poor convergence.

**Parameter Tuning**:  
For RMSprop, the learning rate was the key parameter to tune. After experimentation, a learning rate of **0.001** was chosen as the optimal value. The decay rate was kept at the default value of 0.9, as it provided a good balance between maintaining sensitivity to recent gradients and ensuring stable updates over time.

### 4. **Learning Rate Adjustment**

**Principle**:  
The learning rate determines the size of the steps taken by the optimizer during gradient descent. In many cases, a fixed learning rate is not ideal, as large steps can cause overshooting in the early stages, while small steps in the later stages can slow convergence. Learning rate adjustment (decay) ensures that the learning rate is high during the early stages of training and gradually decreases as the model approaches an optimal solution.

**Relevance to the Project**:  
Language detection is sensitive to the learning rate, as large initial steps can help the model learn quickly, but small steps are needed later to fine-tune the model. A decaying learning rate allowed the model to quickly learn the basic patterns in the language data and then fine-tune without overshooting as it neared optimal performance.

**Parameter Tuning**:  
The learning rate was initially set at **0.001** and decayed by a factor of **0.1** every **10 epochs**. This schedule ensured that early learning was fast, but the learning rate reduced over time to avoid instability.

### 5. **Dropout**

**Principle**:  
Dropout is a regularization technique where, during training, a fraction of the neurons in a neural network are randomly “dropped out” or deactivated. This prevents the model from becoming too reliant on any single neuron and forces the model to learn more distributed representations, improving generalization and reducing overfitting.

**Relevance to the Project**:  
In this language detection model, dropout was essential for preventing overfitting due to the small size of the dataset. By randomly deactivating neurons, dropout ensured that the model did not memorize the training data but rather learned generalizable features that could apply to unseen samples.

**Parameter Tuning**:  
The dropout rate controls the fraction of neurons dropped during training. A dropout rate of **0.5** was selected after testing rates from 0.2 to 0.6. This rate was found to provide a good balance between preventing overfitting and maintaining enough capacity for the model to learn meaningful patterns.

### 6. **Early Stopping**

**Principle**:  
Early stopping is a technique that monitors the model's performance on the validation set during training and halts the training process if the performance stops improving. This prevents the model from overfitting, as it stops learning once it starts to memorize the training data instead of improving generalization.

**Relevance to the Project**:  
Given the limited dataset size, the model could easily start to overfit after several epochs. Early stopping ensured that the model training was halted as soon as the validation loss stopped decreasing, preventing unnecessary overfitting while maintaining strong generalization performance.

**Parameter Tuning**:  
Early stopping was set with a patience of **5 epochs**, meaning that training would stop if the validation loss did not improve for 5 consecutive epochs. This patience level was chosen after testing different values to ensure the model had enough time to improve without halting too early.

---

### Model Combinations and Performance

Each combination of techniques was carefully selected and tuned for optimal performance:

- **Combination 1**: L2 regularization, Adam optimizer, learning rate adjustment, dropout  
  - **Accuracy**: 0.9825  
  - **Validation Loss**: 0.0856  
  - This combination performed the best, with L2 regularization preventing overfitting, Adam providing efficient updates, and dropout further enhancing generalization.

- **Combination 2**: L1 regularization, RMSprop optimizer, dropout  
  - **Accuracy**: 0.9619  
  - **Validation Loss**: 0.1140  
  - L1 regularization helped promote feature sparsity, but RMSprop's handling of large gradients was less effective in this configuration compared to Adam.

- **Combination 3**: L2 regularization, RMSprop optimizer, learning rate adjustment, early stopping  
  - **Accuracy**: 0.9381  
  - **Validation Loss**: 0.1793  
  - While this combination provided good results, RMSprop and the early stopping criteria led to a lower overall accuracy and higher validation loss.

- **Combination 4**: L1 regularization, Adam optimizer, dropout, early stopping  
  - **Accuracy**: 0.9381  
  - **Validation Loss**: 0.1646  
  - This combination also produced good results, though the choice of L1 regularization and early stopping led to a slightly higher validation loss.

---

### Justification of Parameter Selections

- **L1 vs. L2 Regularization**: L2 was chosen for most combinations to prevent overfitting in a balanced way, while L1 was used in some cases to encourage feature sparsity.
- **Optimizers**: Adam provided more stable learning for this dataset, while RMSprop was effective but slightly less stable in combination with other techniques.
- **Learning Rate Adjustment**: A decaying learning rate ensured faster convergence initially, with fine-tuning as the model neared optimal performance.
- **Dropout**: A dropout rate of 0.5 was chosen to prevent overfitting while maintaining model complexity.
- **Early Stopping**: The patience parameter allowed the model enough time to improve without overfitting, ensuring optimal stopping during training.

These optimizations ensured that the final model achieved high accuracy while generalizing well to unseen data.

### Summary

After experimenting with various combinations of optimization techniques, **Combination 1**—which included **L2 regularization**, the **Adam optimizer**, **learning rate adjustment**, and **dropout**—was chosen as the final model configuration. This combination provided the highest accuracy of **0.9825** and a validation loss of **0.0856**. The balanced approach of L2 regularization for generalization, Adam’s adaptive learning rates, gradual learning rate decay, and dropout to prevent overfitting resulted in the best performance. This model effectively addressed the challenges of language detection, ensuring high accuracy and robust generalization.

**ERROR ANALYSIS**
- **Accuracy**: The model achieved an overall accuracy of **98%**.
- **F1 Score (Macro Average)**: The model achieved an F1 score of **0.9829**, demonstrating high classification performance across the four languages.
- **Specificity**:
  - English: 0.00
  - French: 1.00
  - Kinyarwanda: 1.00
  - Kiswahili: 0.50
  
The model performed exceptionally well, particularly in distinguishing French and Kinyarwanda.

## Evaluation Metrics

- **Precision**: The model achieved high precision across all languages.
- **Recall**: The recall scores were similarly high, indicating that the model correctly identified the most true positives.
- **F1-Score**: The F1-scores balanced precision and recall.
- **Specificity**: This metric helped measure the model's ability to identify negatives correctly.


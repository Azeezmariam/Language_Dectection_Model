
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


# Sentiment Analysis with Deep Learning using BERT

## Overview
This project implements sentiment analysis using BERT (Bidirectional Encoder Representations from Transformers), a powerful transformer-based language model. We will explore the SMILE Twitter dataset to classify sentiments effectively.

## Prerequisites
- Intermediate-level knowledge of Python 3 (familiarity with NumPy and Pandas is preferred, but not required).
- Exposure to PyTorch usage.
- Basic understanding of Deep Learning and Language Models, specifically BERT.

## Project Outline
1. **Introduction**
   - What is BERT?
   - Overview of the BERT architecture and its applications.
   - [Original BERT Paper](https://arxiv.org/abs/1810.04805)
   - [HuggingFace Documentation](https://huggingface.co/transformers/model_doc/bert.html)

2. **Exploratory Data Analysis and Preprocessing**
   - Introduction to the SMILE Twitter dataset.
   - Data cleaning and preprocessing steps.

3. **Training/Validation Split**
   - Splitting the dataset into training and validation sets.

4. **Loading Tokenizer and Encoding our Data**
   - Using the BERT tokenizer to preprocess text data.

5. **Setting up BERT Pretrained Model**
   - Loading a pretrained BERT model for sentiment analysis.

6. **Creating Data Loaders**
   - Implementing PyTorch DataLoaders for batching and shuffling data.

7. **Setting Up Optimizer and Scheduler**
   - Configuring the optimizer and learning rate scheduler for training.

8. **Defining our Performance Metrics**
   - Implementing accuracy and F1 score metrics to evaluate model performance.

9. **Creating our Training Loop**
   - Writing the training loop for model training with logging.

10. **Loading and Evaluating our Model**
    - Evaluating the model on the validation dataset and generating predictions.


## Getting Started
To get started with this project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/prabal-17/Sentiment-Analysis-with-Deep-Learning-using-BERT.
pip install -r requirements.txt

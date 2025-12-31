<h1 align="center">Linguistic Signals of Mental Health</h1>

Mental health conditions affect 1 in 8 people globally. Yet, despite its significance, early detection remains challenging.
This repository implements a BERT-based NLP model to infer users’ mental health conditions from their online posts and achieves an accuracy of 80.95% across seven mental health classes.

## Dataset

This project uses the [Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health) dataset from Kaggle.
The data is sourced from diverse platforms, including social media posts, Reddit, and Twitter, and is created by aggregating and cleaning multiple publicly available datasets on Kaggle.

Each statement of the dataset is labeled with one of these seven mental health conditions:
| Class               | Original Samples |
|---------------------|------------------|
| Normal              | **16,343**       |
| Depression          | 15,404           |
| Suicidal            | 10,652           |
| Anxiety             | 3,841            |
| Bipolar             | 2,777            |
| Stress              | 2,587            |
| **Personality disorder** | **1,077**   |

The severe imbalance in this dataset poses a major challenge. To address this issue, both downsampling and upsampling strategies were applied after the train/validation/test split.

### Downsampling on Majority Classes
KNN applied to TF–IDF representations were used to select representative samples

### Upsampling (Augmentation) on Minority Classes
Minority classes were augmented using text-level strategies, including:
  * mixing chunks of texts from the same class
  * shuffling sentences within a post
  * randomly dropping sentences

This resulted in a total of 23,724 balanced training samples.

## Text Preprocessing
Tokenization is performed using a BERT tokenizer. Due to long post lengths, each input consists of **the first 128 tokens (introduction) and the last 128 tokens (conclusion)** to preserve both the contextual setup and the concluding content of each post.

## Model Architecture
A BERT model is used as the encoder, followed by a linear classification layer operating on the BERT [CLS] token embedding. The first seven BERT layers are frozen, while the remaining layers are fine-tuned.

`BERT-base-uncased (Layers 0–7 frozen)`<br>
&emsp;&emsp;↓<br>
`Dropout (p = 0.3)`<br>
&emsp;&emsp;↓<br>
`Linear Classification Head (768 → 7)`


To demonstrate the effect of fine-tuning, PCA visualizations of the test set embeddings are provided. As observed, embeddings of statements belonging to the same class become more localized, and class separation becomes more distinct after fine-tuning.

## Training and Evaluation
The model is trained for 15 epochs using AdamW optimizer and balanced CrossEntropy loss, and is evaluated on a held-out test set.

### Test Performance
Loss: 0.5883 | Accuracy: 80.95% | Macro-F1: 79.46% | Weighted-F1: 81.27%

| Class                | Precision | Recall | F1-Score | Support |
|----------------------|-----------|--------|----------|---------|
| Anxiety              | 0.862     | 0.875  | 0.868    | 192     |
| Bipolar              | 0.683     | 0.928  | 0.787    | 139     |
| Depression           | 0.806     | 0.729  | 0.766    | 771     |
| Normal               | 0.979     | 0.873  | 0.923    | 817     |
| Personality disorder | 0.686     | 0.889  | 0.774    | 54      |
| Stress               | 0.593     | 0.915  | 0.720    | 129     |
| Suicidal             | 0.709     | 0.741  | 0.725    | 533     |

### Confusion Matrix

### Key Observations
1. The relatively small gap between Macro F1 and Weighted F1 scores suggests that performance is not dominated by the majority class, indicating that class imbalance has been largely mitigated.
2. Depression and Suicidal classes are frequently confused due to shared vocabulary and emotional tone, which is also clearly evident from their word clouds.
3. Personality Disorder, the smallest minority class, exhibits lower precision due to limited data. However, its recall score is strong, indicating that the model still successfully learns to recognize minority-class samples and is sensitive to at-risk language.

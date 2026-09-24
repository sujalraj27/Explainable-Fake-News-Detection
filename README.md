# Explainable Fake News Detection

### Using Machine Learning, Large Language Models, and Knowledge Graphs

## 📌 Project Overview

The rapid growth of online news platforms and social media has
increased the spread of fake news and misinformation.

This project presents an Explainable Fake News Detection System
that classifies news articles as Fake or Real using Machine
Learning and provides explanations to improve transparency
and user understanding.

The framework integrates Machine Learning, Explainable AI (XAI),
Knowledge Graphs, and Large Language Models (LLMs) into a
unified system.

## 🎯 Objectives

- Detect fake and real news articles automatically.
- Improve the transparency of machine learning predictions.
- Identify important words influencing classification.
- Visualize relationships among entities using Knowledge Graphs.
- Generate human-readable explanations for predictions.
- Provide understandable results through an interactive application.

## 🚀 Key Features

1. Fake News Classification
   - TF-IDF feature extraction.
   - Logistic Regression classifier.
   - Fake and Real news prediction.

2. Explainable AI
   - LIME-based local explanations.
   - Identification of influential words.

3. Knowledge Graph Visualization
   - Extraction of important entities.
   - Visualization of relationships between entities.

4. LLM-Based Explanation
   - Human-readable explanations of prediction results.
   - Gemini API integration as described in the research.

5. Interactive Application
   - FastAPI backend.
   - Streamlit frontend.

6. Automated Reporting
   - PDF report generation.
   - Prediction results and explanation summaries.

## 🛠️ Technologies Used

- Python
- Scikit-learn
- TF-IDF
- Logistic Regression
- Naive Bayes
- Support Vector Machine (SVM)
- LIME
- spaCy
- NetworkX
- PyVis
- FastAPI
- Streamlit
- ReportLab
- Large Language Models (LLMs)

## 📊 Dataset

The project uses the WELFake dataset for fake news
classification.

The dataset contains news articles labeled as Fake or Real.

The dataset was divided into training and testing subsets
using an 80:20 ratio.

## 📈 Experimental Results

The following classification results are reported in the
research paper:

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 94.55% | 93.71% | 94.02% | 93.87% |
| Naive Bayes | 84.83% | 81.58% | 84.99% | 83.25% |
| SVM | 94.99% | 94.18% | 94.54% | 94.36% |

These values are reported experimental results and may depend
on the dataset split and experimental configuration.

## 📁 Model Artifacts

The repository includes the following model and evaluation
artifacts:

- `logistic_model.pkl` — Trained Logistic Regression model.
- `tfidf_vectorizer.pkl` — TF-IDF vectorizer.
- `logistic_metrics.json` — Classification evaluation metrics.
- `logistic_confusion_matrix.png` — Confusion matrix visualization.
- `performance_report.pdf` — Model performance report.

## 🔮 Future Enhancements

- Integration of transformer-based models such as BERT,
  RoBERTa, and DeBERTa.
- Advanced Knowledge Graph construction.
- Graph Neural Networks (GNNs).
- Multilingual fake news detection.
- Real-time social media monitoring.
- Multimodal analysis involving text, images, and videos.

## 👩‍💻 Author

Sujal Raj

Department of Computer Science and Engineering


## 📄 Research

This project is based on the research work titled:

"Explainable Fake News Detection using Machine Learning,
Large Language Models, and Knowledge Graphs."

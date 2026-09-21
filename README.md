# 📊 Student Exam Performance Prediction Project

An End-to-End Machine Learning project designed to predict a student's **Exam Score** based on academic, lifestyle, motivational, and environmental factors.

The project goes beyond simple score prediction by performing **model-based what-if analysis**. It tests realistic changes in relevant input factors using the trained Machine Learning model and identifies the changes with the highest potential positive impact on the predicted exam score.

The application follows a modular Machine Learning architecture and provides predictions through a responsive Flask web interface.

---

## 🎯 Project Overview

The objective of this project is to build a Machine Learning system that can:

- Predict a student's expected exam score.
- Process numerical and categorical student-related features.
- Train and compare multiple regression models.
- Select the best-performing model.
- Evaluate the final model using regression metrics.
- Provide predictions through a Flask web application.
- Identify potential improvement opportunities using model-based what-if analysis.

---

## ✨ Key Features

### 🔹 Exam Score Prediction

The trained Machine Learning model predicts the expected exam score based on student-related input features.

### 🔹 Model-Based Improvement Analysis

The application performs model-based what-if analysis instead of relying on simple hardcoded suggestions.

The system:

1. Generates the original prediction.
2. Modifies one relevant factor within a realistic range.
3. Sends the modified input through the same preprocessing pipeline.
4. Generates a new prediction using the trained model.
5. Calculates the difference between the predictions.
6. Repeats the process for multiple relevant factors.
7. Filters meaningful positive improvements.
8. Ranks the opportunities according to their predicted impact.

The system focuses the improvement analysis on factors that can reasonably be considered adjustable, such as study habits, attendance, lifestyle, learning environment, and other relevant student-related factors.

These recommendations represent **model-based what-if analysis** and should not be interpreted as guaranteed causal improvements in actual exam performance.


### 🔹 Responsive Web Interface

The Flask application provides:

- Organized input sections.
- Responsive form layout.
- Smooth transitions and hover effects.
- Predicted exam score.
- Performance category.
- Model-based improvement opportunities.
- Current value → recommended value comparison.
- Predicted impact based on model what-if analysis.

---

## 🔑 Input Features

The model uses **16 input features**.

### 📚 Academic and Study Factors

- Hours Studied
- Attendance
- Previous Scores
- Tutoring Sessions

### 🌙 Lifestyle Factors

- Sleep Hours
- Physical Activity

### 🧠 Motivation and Learning Environment

- Motivation Level
- Access to Resources
- Parental Involvement
- Internet Access
- Teacher Quality
- Extracurricular Activities
- Peer Influence

### 👨‍👩‍👧 Family and Background Factors

- Parental Education Level
- School Type
- Family Income

---

## 🤖 Machine Learning Architecture

```text
Raw Student Data
        │
        ▼
Data Ingestion
        │
        ▼
Data Transformation
        │
        ├── Numerical Feature Processing
        │
        └── Categorical Feature Processing
        │
        ▼
Train-Test Split
        │
        ▼
Multiple Model Training
        │
        ▼
Model Evaluation
        │
        ├── R² Score
        ├── MAE
        └── RMSE
        │
        ▼
Best Model Selection
        │
        ▼
Saved Model + Preprocessor
        │
        ▼
Flask Prediction Pipeline
        │
        ▼
Exam Score Prediction
        │
        ▼
Model-Based What-If Analysis
        │
        ▼
Improvement Opportunities

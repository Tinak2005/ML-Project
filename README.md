# 📊 Student Exam Performance Prediction Project

A comprehensive, modular End-to-End Machine Learning pipeline that automates the entire lifecycle from data ingestion to model deployment. Built using software engineering best practices, this project handles raw student data, processes it through custom transformation pipelines, trains multiple regression algorithms to select the best performer, and serves math score predictions via a responsive Flask web interface hosted on Render.

## 🎯 Project Overview & Objective
The goal of this project is to predict the **Math Score** of students based on various demographic and exam-related features. By analyzing these factors, the model helps identify key variables that significantly impact a student's academic performance in exams.

### 🔑 Key Features Handled:
* **Gender:** Male / Female
* **Race/Ethnicity:** Groups (A, B, C, D, E)
* **Parental Level of Education:** High School, Associate's Degree, Bachelor's, Master's, etc.
* **Lunch Type:** Standard vs. Free/Reduced (acts as an economic indicator)
* **Test Preparation Course:** Completed vs. None
* **Reading & Writing Scores:** Used to analyze correlation with mathematical performance.

---

## 📂 Project Structure & Architecture

The project follows a production-grade modular architecture split into independent, reusable components:

```text
├── artifacts/               # Generated data splits and serialized pickle files
├── logs/                    # Automated runtime execution logs for debugging
├── notebook/                # Jupyter Notebooks for EDA and initial model prototyping
├── src/                     # Core application source code
│   ├── components/          # Pipeline building blocks
│   │   ├── __init__.py
│   │   ├── data_ingestion.py      # Script to load and split raw data
│   │   ├── data_transformation.py # Custom preprocessors, pipelines, and scaling
│   │   └── model_trainer.py       # Model training, evaluation, and tuning
│   ├── pipeline/            # Execution workflows
│   │   ├── __init__.py
│   │   ├── predict_pipeline.py    # Custom pipeline for real-time user input prediction
│   │   └── train_pipeline.py      # Triggers the complete training sequence
│   ├── exception.py         # Custom robust exception handling framework
│   ├── logger.py            # Custom logging setup to track execution flow
│   └── utils.py             # Global utility functions (e.g., model saving/loading)
├── templates/               # HTML layout files for the Flask frontend
├── app.py                   # Flask application entry point
├── Procfile                 # Process file defining production WSGI server for Render
├── requirements.txt         # Project dependencies and libraries
└── README.md                # Project documentation

---

## 🚀 How to Run Locally

1. Clone the repository:
   ```bash
   git clone [https://github.com/Tinak2005/ML-Project.git](https://github.com/Tinak2005/ML-Project.git)

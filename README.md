# Fake News Detection using Machine Learning and Streamlit

## Project Overview

In this project, we implement a **Fake News Detection system** using **Natural Language Processing (NLP)** and **Machine Learning** techniques.

Fake news detection is a **text classification problem** where the goal is to determine whether a news article is **reliable (real)** or **unreliable (fake)**.

Each news article will be assigned a label from the predefined categories:

| Label | Meaning         |
| ----- | --------------- |
| 0     | Reliable News   |
| 1     | Unreliable News |

Example:

| ID   | News Content                                  | Label      |
| ---- | --------------------------------------------- | ---------- |
| 1245 | Government confirms new economic policy...    | Reliable   |
| 3271 | Celebrity secretly controls world politics... | Unreliable |

This project includes:

* **Data preprocessing**
* **Exploratory Data Analysis**
* **Text vectorization using TF-IDF**
* **Machine Learning model training**
* **A Streamlit web application for real-time fake news detection**

---

## Fake News Detection

Fake news detection is an important NLP task used to automatically identify misleading or fabricated information.

Applications include:

* Social media monitoring
* News verification
* Misinformation prevention
* Journalism support tools

This project uses **Machine Learning with TF-IDF text encoding** to classify news articles.

---

## Jupyter Notebook

All analysis and experimentation are performed inside the notebook:

```
FakeNewsAnalysis.ipynb
```

The notebook includes:

* Dataset loading
* Data cleaning
* Text preprocessing
* Feature engineering
* Model training
* Model evaluation

---

## Streamlit Application

The project includes a **Streamlit web app** that allows users to input news content and detect whether it is **fake or reliable**.

Run the application using:

```bash
streamlit run app.py
```

Below is the deployed application interface.

![Fake news detection app](imgs/final_app.png)

---


## Project Structure

```
fake_news_detection
│
├── FakeNewsAnalysis.ipynb
├── app.py
│
├── models
│   ├── final_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── dataset
│   └── news_dataset.csv
│
└── README.md
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* NLTK
* Scikit-learn
* TF-IDF
* Joblib
* Streamlit

---

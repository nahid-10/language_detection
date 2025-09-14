# Language Detection using NLP 🌍🔤

This project demonstrates how to build a **language detection system** using Natural Language Processing (NLP) and Machine Learning. The model can detect the language of a given text input by analyzing character and word patterns.

---

## 📌 Project Overview

The goal of this project is to:  

- Preprocess and clean **multilingual text data**  
- Convert text into numerical features using **CountVectorizer**  
- Train a **machine learning classification model** to predict the language  
- Evaluate model performance using accuracy and classification metrics  
- Provide a simple interface to test the model with user input  

---

## 🧰 Tools & Technologies

| Tool / Library       | Purpose                                       |
|----------------------|-----------------------------------------------|
| `Python`             | Core programming language                     |
| `scikit-learn`       | Machine learning algorithms & evaluation      |
| `pandas`             | Data preprocessing & dataset handling         |
| `numpy`              | Numerical operations                          |
| `CountVectorizer`    | Feature extraction from text (bag-of-words)   |
| `NLP` techniques     | Text preprocessing and analysis               |

---

## 📂 Files Included

- `language_detection.ipynb` – Jupyter notebook with full implementation  
- `data.csv` – Dataset containing text samples in multiple languages  
- `model.pkl` – Trained model saved using pickle (optional)  
- `requirements.txt` – Python dependencies  
- `README.md` – Project documentation  

---

## 📊 Sample Usage

Some example inputs and predictions:

- Input: `"Bonjour, comment ça va ?"` → Output: **French**  
- Input: `"Hello, how are you?"` → Output: **English**  
- Input: `"¿Dónde está la biblioteca?"` → Output: **Spanish**  
- Input: `"আমি কম্পিউটার সায়েন্স পড়ি"` → Output: **Bengali**  

---

## ⚙️ Technical Details

- **Data Preprocessing:**  
  - Removed special characters and punctuation  
  - Lowercased text for uniformity  
  - Tokenized and vectorized text using **CountVectorizer**  

- **Model Training:**  
  - Algorithms tested: Naive Bayes, Logistic Regression, SVM  
  - Final model chosen based on highest accuracy score  

- **Evaluation Metrics:**  
  - Accuracy Score  
  - Confusion Matrix  
  - Classification Report  

---

## 🔧 Optimization Features

- Supports **multiple languages** (depending on dataset)  
- Model can be **re-trained with new data**  
- Lightweight and fast for real-time predictions  
- Can be integrated into a **Flask/Django web app** or API  

---

## 💡 Future Improvements

- Use **TF-IDF** instead of CountVectorizer for better accuracy  
- Train with **deep learning models** (e.g., LSTMs, Transformers)  
- Expand dataset to cover **more languages**  
- Deploy model with a **REST API** for production use  

---

## 📄 License

This project is **open-source** and free to use for educational purposes.

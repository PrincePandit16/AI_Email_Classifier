# AI Email Classifier 📧🤖

**Live Demo:** https://aiemailclassifier.streamlit.app/

AI Email Classifier is a **machine learning powered email classification system** built with Python and Streamlit. It leverages a trained Support Vector Machine (SVM) model to categorize emails into meaningful folders such as **complaint, feedback, support, spam, and other** — providing confidence scores, urgency estimation, and category-wise analytics.

This project includes both the **training notebook** and the **Streamlit app code**, and is deployed live for instant use.

---

## 🚀 Features

### 🧠 Intelligent Email Categorization
- Classifies emails into 5 meaningful categories:
  - **Complaint**
  - **Feedback**
  - **Support**
  - **Spam**
  - **Other**
- Uses **TF-IDF vectorization + calibrated SVM** for high accuracy.

### 📊 Interactive UI (Streamlit)
- Clean dashboard with **real-time insights**.
- Email classification with confidence & urgency metrics.
- **Analytics**: Category distribution, confidence histogram, keyword influence.
- Quick load of **example emails** for testing.

### 📈 Model Transparency & Insights
- Confidence scores for each prediction.
- Extracts **decision-impact keywords** to explain classifications.
- Displays **historical trends** of emails by category.
- Includes model comparison and evaluation metrics.

---

## 🛠 Tech Stack

| Component                  | Technology     |
|---------------------------|----------------|
| Frontend / UI             | Streamlit      |
| Backend / Model           | Python         |
| ML Algorithm              | Linear SVM     |
| Vectorization             | TF-IDF         |
| Deployment                | Streamlit Cloud |

---

## 📦 Repository Contents

| File / Folder     | Description |
|-------------------|-------------|
| `app.py`          | Main Streamlit application code |
| `svm_model.pkl`   | Pre-trained model used by app |
| `Emails.ipynb`    | Notebook for data preprocessing, model training & evaluation |
| `merged_data.csv` | Dataset used for model training |
| `requirements.txt`| Python dependencies |

---

## 🧩 How It Works

1. **Preprocessing**
   - Emails are cleaned & normalized.
   - Text converted into numerical features using TF-IDF.
2. **Training**
   - A Linear Support Vector Machine (SVM) model is trained with stratified sampling.
   - The SVM is wrapped in **CalibratedClassifierCV** for probability estimates.
3. **Prediction**
   - The model predicts category + confidence for any new email text.
   - Urgency is calculated based on keywords and category base score.

---

## 📂 Folder Structure
```
AI_Email_Classifier/
│
├── app.py                # Streamlit application logic
├── Emails.ipynb          # Model training & experimentation
├── merged_data.csv       # Labeled dataset
├── svm_model.pkl         # Serialized trained model
├── requirements.txt      # Dependencies
└── README.md             # Documentation
```

---

## 🔧 Setup / Local Installation

> ⚠️ Make sure you have Python 3.8+ installed.


# 1. Clone this repository
```bash
git clone https://github.com/PrincePandit16/AI_Email_Classifier.git
cd AI_Email_Classifier
```

# 2. Install dependencies
```bash
pip install -r requirements.txt
```


# 3. Place the trained model
```bash
Ensure `svm_model.pkl` is present in the project root
```


# 4. Run the Streamlit app
```bash
streamlit run app.py
```

## 📈 What This Project Demonstrates
- ✔ End-to-End Machine Learning Pipeline
- ✔ NLP Feature Engineering (TF-IDF, N-grams)
- ✔ Model Calibration with Probability Estimation
- ✔ High-Accuracy Multi-Class Classification (~99%)
- ✔ Production-Ready Deployment (Streamlit Cloud)
- ✔ Clean UI + ML Integration
- ✔ Scalable & Modular Design Architecture

## 🔮 Future Enhancements
- 🔍 SHAP / LIME based model explainability
- 🌐 REST API version using FastAPI
- 🐳 Docker containerization
- 🔄 CI/CD pipeline integration
- 🗄 Database logging & analytics storage
- 🌍 Multi-language email classification support

## 🤝 Contributing
# Contributions are welcome and appreciated!

If you would like to improve this project:
1. 🍴 Fork the repository
2. 🌱 Create a new feature branch
```bash
git checkout -b feature-name
```
3. 💾 Commit your changes
```bash
git commit -m "Add new feature"
```
4. 🚀 Push to your branch
```bash
git push origin feature-name
```
5. 🔁 Open a Pull Request

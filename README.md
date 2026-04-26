# 🧠 Brain Stroke Detection System

A machine learning-based web application that detects brain stroke conditions from medical images using deep learning and ensemble models. The system provides automated prediction through a user-friendly interface.

---

## 🚀 Features

* 🧠 Brain stroke detection using trained ML/DL models
* ⚡ Automatic model download from Hugging Face
* 💾 Local caching for faster subsequent runs
* 🌐 Flask-based web interface
* 📊 Ensemble learning (AdaBoost, CatBoost, LightGBM, Decision Tree)
* 🖼️ Image-based prediction support

---

## 🏗️ Project Structure

```
Brain_Stroke_Detection/
│
├── app.py                     # Main Flask application
├── stroke_prediction.py      # Prediction logic
├── download_model.py         # Downloads models from Hugging Face
├── requirements.txt          # Dependencies
├── static/                   # Static assets (CSS, JS, images)
├── README.md
```

---

## 🤖 Model Handling

The trained models are hosted on Hugging Face and are **not stored in this repository** to keep it lightweight.

* Models are automatically downloaded at runtime using `huggingface_hub`
* Downloaded models are cached locally (`model_cache/`)
* No manual setup required

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```
git clone https://github.com/your-username/Brain_Stroke_Detection.git
cd Brain_Stroke_Detection
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the application

```
python app.py
```

---

## 🌐 Usage

1. Open the application in your browser
2. Upload a brain CT scan image
3. Get instant prediction results

---

## 🧪 Technologies Used

* Python
* Flask
* TensorFlow / Keras
* Scikit-learn
* LightGBM
* CatBoost
* Adabost
* Hugging Face Hub
* OpenCV

---

## 📌 Key Highlights

* Clean separation of concerns (model loading vs prediction logic)
* Efficient model caching mechanism
* Scalable and deployment-ready architecture

---

## ⚠️ Note

* Dataset is not included due to size constraints
* Models are fetched dynamically from Hugging Face

---

## 👨‍💻 Author

**Saish Sagvekar**

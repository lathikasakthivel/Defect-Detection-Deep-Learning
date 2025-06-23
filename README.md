# 🛠️ Defect Detection CNN (Casting Product)

A Convolutional Neural Network (CNN) implemented in PyTorch for detecting manufacturing defects in casting products through image classification.

## 📸 Sample Outputs
- ✅ Good Product
- ⚠️ Defect Detected — Review Required

## 📊 Features
- Custom CNN model
- Training/Validation Split
- Training Loss Plot
- Confusion Matrix
- Classification Report
- Visualized Predictions with readable labels

## 📂 Directory Structure
  defect_detection_cnn/
  ├── data/
  │ └── casting_data/
  │ ├── def_front/
  │ │ └── *.png
  │ └── ok_front/
  │ └── *.png
  ├── defect_detection_cnn.py
  ├── requirements.txt
  ├── README.md
  └── .gitignore

## 📦 Install Dependencies
```bash
pip install -r requirements.txt
```

🚀 Run the Project
```bash
python defect_detection_cnn.py
```
📈 Results
  
  Final accuracy on test set
  Confusion Matrix plot
  Classification report (precision, recall, F1-score)
  5 sample predictions displayed with correctness labels

📌 Tech Stack
  
  Python
  PyTorch
  Matplotlib
  scikit-learn


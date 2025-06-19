# 🧍‍♂️ Human Action Detection using Deep Learning

This project uses deep learning to detect human actions from video clips. The model is trained on the [KTH Human Motion Dataset](https://www.kaggle.com/datasets/beosup/kth-human-motion), which includes actions like walking, jogging, boxing, and more.

---

## 📁 Dataset

- **Source**: [KTH Human Motion Dataset](https://www.kaggle.com/datasets/beosup/kth-human-motion)
- **Actions**:
  - Boxing
  - Hand Clapping
  - Hand Waving
  - Jogging
  - Running
  - Walking
- **Total Clips**: 600+
- **Format**: `.avi` videos for each action class

---

## 📊 Exploratory Data Analysis (EDA)

> Before training the model, we analyzed the dataset to understand its distribution and class balance.

### ✅ Action Distribution

![image](https://github.com/user-attachments/assets/c50b6cbc-8bbb-4ae0-8454-5ffa0c65ff14)

### ✅ Sample Frame from Each Class

![image](https://github.com/user-attachments/assets/5a2fe98a-e183-46a6-90d1-d577cbd874de)


### ✅ Clip Length Distribution

![image](https://github.com/user-attachments/assets/bb7d7303-245c-477a-a59d-d704181b2cd9)


![image](https://github.com/user-attachments/assets/a3a4381f-8009-4407-a763-722689018839)


![image](https://github.com/user-attachments/assets/6e5fd88b-9b69-4148-9be9-830952f53073)

### Confusion Matrix


![image](https://github.com/user-attachments/assets/fc1e4ec4-8ad0-4a80-ba2e-bfb42f4084c8)

### Frontend UI Screenshot

![Screenshot 2025-06-19 120425](https://github.com/user-attachments/assets/f116ef18-3d25-42b5-8022-369c43902d93)

---

## 🧠 Model Architecture

A custom **CNN-LSTM** architecture was used to handle spatial and temporal patterns in video clips.

### 🧩 Architecture

- `TimeDistributed(Conv2D + MaxPooling + BatchNorm)`
- `LSTM (64 units)`
- `Dense + Dropout + Softmax`

### ✅ Model Summary
```text
Input: (30 frames, 64, 64, 1)
Output: 6 class softmax
🏋️ Model Training
✅ Optimizer: Adam

✅ Loss: Categorical Crossentropy

✅ Epochs: 15

✅ Validation Split: 20%

📈 Training Curve

📌 Evaluation Metrics
✅ Final Evaluation
text
Copy
Edit
Accuracy: 70%
Precision, Recall, F1 Score (Macro Average): ~0.70
✅ Classification Report
markdown
Copy
Edit
               precision    recall  f1-score   support

      boxing       0.95      0.95      0.95        20
handclapping       0.81      0.85      0.83        20
  handwaving       0.79      0.75      0.77        20
     jogging       0.48      0.55      0.51        20
     running       0.56      0.50      0.53        20
     walking       0.63      0.60      0.62        20

Overall Accuracy: 0.70
🚀 Streamlit Frontend
🎬 Features
### Check the Demo video

Automatically extracts up to 30 frames

Predicts the action using a pre-trained CNN-LSTM model

Displays top 3 class predictions with confidence scores

📂 Required Files
text
Copy
Edit
📦 Human Action Detection/
│
├── human_action_model.h5          # Trained model
├── label_encoder.pkl              # Encodes class labels
├── human_action_app.py            # Streamlit frontend
▶️ Run the App
bash
Copy
Edit
streamlit run "D:/ML PROJECTS/Human Action Detection/human_action_app.py"
🛠️ Technologies Used
Python 🐍

TensorFlow / Keras

OpenCV

NumPy / Pandas

Streamlit

Scikit-learn

📦 Future Improvements
Real-time webcam integration for live predictions

Add audio-based human activity fusion

Use 3D CNNs or Transformer-based models for longer video sequences




✨ Author
V. Yuvan Krishnan

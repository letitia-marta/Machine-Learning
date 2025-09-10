# Traffic Sign Classifier 🚦

This project implements a **Machine Learning model** trained to recognize and classify traffic signs, using the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset available on Kaggle:

🔗 [GTSRB – German Traffic Sign Dataset on Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

## 📁 Project Structure

```
Machine-Learning/
│
├── datasets/
│   └── GTSRB_dataset/
│       ├── Meta/
│       ├── Train/
│       ├── Test/
│       ├── Meta.csv
│       ├── Train.csv
│       └── Test.csv
│
└── src/
    └── Sigmoid.py
```


### Dataset Description
The GTSRB dataset is organized into three parts:
- `Meta/`: Metadata for traffic sign classes.
- `Train/`: Training images and labels.
- `Test/`: Testing images and ground truth labels.
- `*.csv` files contain relevant metadata and labels for each split.

## 🧠 Model Overview

- The project implements a **LeNet-5 architecture** adapted for traffic sign classification.  

- Three main implementations are provided, each using a different activation function for comparison purposes:  
  - `Sigmoid.py` → Sigmoid activation  
  - `ReLU.py` → ReLU activation  
  - `Tanh.py` → Tanh activation  

- These variants allow benchmarking and understanding how different activation functions impact model performance.  

- Data preprocessing and feature extraction are assumed prior to feeding into the models.  

- Model performance is evaluated and results are visualized using:  
  - ✅ **Confusion Matrix** – to analyze class-wise accuracy.  
  - 📈 **ROC Curve** – to assess overall classification performance.  


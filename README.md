# Traffic Sign Classifier 🚦

This project implements a **Machine Learning model** trained to recognize and classify traffic signs using the [GTSRB (German Traffic Sign Recognition Benchmark)](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) dataset.
---

## 📁 Project Structure

Machine-Learning/

│

├── datasets/

│ └── GTSRB_dataset/

│ ├── Meta/

│ ├── Train/

│ ├── Test/

│ ├── Meta.csv

│ ├── Train.csv

│ └── Test.csv

│

└── src/

└── Sigmoid.py


### Dataset Description
The GTSRB dataset is organized into three parts:
- `Meta/`: Metadata for traffic sign classes.
- `Train/`: Training images and labels.
- `Test/`: Testing images and ground truth labels.
- `*.csv` files contain relevant metadata and labels for each split.

---

## 🧠 Model Overview

- The model uses a **Sigmoid activation function** as demonstrated in `src/Sigmoid.py`.
- Data preprocessing and feature extraction are assumed prior to feeding into the model.
- The current implementation focuses on classification using numerical features derived from image data.

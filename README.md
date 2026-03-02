# KU Ingolstadt Building Classifier 🏫

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange)
![Keras](https://img.shields.io/badge/Keras-3.13.2-red)
![License](https://img.shields.io/badge/Code_License-MIT-green)
![Data License](https://img.shields.io/badge/Data_License-CC_BY_4.0-lightgrey)

## 📌 Project Overview
This project is a custom Convolutional Neural Network (CNN) built to classify images of campus buildings (and interior) located near the Catholic University of Eichstätt-Ingolstadt (KU) in Bayern, Germany.

The primary motivation for this project was to apply concepts learned in our **Deep Learning** course taught by Professor Dr. Felix Voigtlaender and Dr. Hannes Matt. Our long-term vision is to simulate a platform where users can scan a building with their phone camera to instantly retrieve its name and information.

## 💾 The Dataset (Custom Data Collection)
**Note:** Due to the large size of the dataset (23+ GB), the raw images and videos are hosted externally on Kaggle.
* 🔗 **Download the dataset here:** https://www.kaggle.com/datasets/shahbazkhalilli/ku-phase-2

**How we built the dataset:**
* We manually collected over **5000 raw images** around the MIDS building and KU campus.
* To make the model robust for real-world application, we mimicked the behavior of a person walking along standard pathways and scanning buildings with their phone from various angles and distances.
* We further expanded our dataset by recording walking videos and programmatically extracting frames (2 frames per second) to maximize our training data, as a result of all, we got totally ~8500 image dataset.

### 🏛️ Target Classes (9 Categories)
After an initial phase where we separated buildings by specific angles (21 classes), we refined our approach to a robust 9-class system representing core structures:
. Basement room in the Georgianum
. The church on the way to the Mensa
. Entrance door of the Georgianum
. Georgianum
. The Kreuztor
. The main entrance of the KU (x80 bus-stop)
. The pink building in front of the Georgianum
. Seminar room (201-203) in the Georgianum
. The Main Building (Hauptbau) / WFI building

## 🧠 Model Architecture & Training
We built a custom CNN using Keras `Sequential` API. 
* **Input Shape:** 224 x 168 x 3 (RGB images)
* **Architecture:** 3 Convolutional Blocks (32 -> 64 -> 128 filters) with Max Pooling, followed by a Flatten layer, a Dense layer of 64 neurons, and a final Softmax output.
* **Optimization Strategy:** We initially tested SGD but observed poor convergence. Switching to the **Adam Optimizer** with a low learning rate (10^-5) yielded stable and superior results.
* **Callbacks:** We utilized `EarlyStopping` (restoring best weights), `ReduceLROnPlateau`, and custom `ModelCheckpoint` callbacks to monitor validation loss.

## 🏆 Final Model Performance & Progression
Throughout this project, we iteratively improved our classifier by expanding our dataset, refining our architecture, and ultimately leveraging transfer learning. These steps successfully increased our model's accuracy on unseen test data from **42.55% to a highly robust 93.62%**. 

| Development Phase | Dataset Size | Architecture Approach | Test Accuracy |
| :--- | :--- | :--- | :---: |
| **Phase 2** | ~3,500 images | Custom CNN (Trained from scratch) | 42.55% |
| **Phase 3 (Base)** | Expanded (~8,500) | Updated Custom Architecture | 87.23% |
| **Phase 3 (Fine-Tuned)** | Expanded (~8,500) | MobileNetV2 (Transfer Learning) | **93.62%** |

## 📂 Repository Structure
```text
📦 KU-BUILDINGS-CLASSIFIER
 ┣ 📂 data_raw/                           # (Ignored) Original raw images (9 categories)
 ┣ 📂 dataset/                            # (Ignored) 80/20 Stratified Train/Val split
 ┣ 📂 logs/                               # Training history and epoch metrics
 ┣ 📂 models/                             # Saved models (.keras, .weights.h5) & class_mapping.json
 ┣ 📂 test_images/                        # (Ignored) Completely unseen real-world test images
 ┣ 📂 videos/                             # (Ignored) Source videos for frame extraction
 ┣ 📜 image_extractor_from_videos.ipynb   # Frame extraction pipeline (2 FPS)
 ┣ 📜 split_data.ipynb                    # 80/20 stratified data splitting script
 ┣ 📜 ku_building_cnn_phase1.ipynb        # Phase 1: 21-class baseline architecture
 ┣ 📜 ku_building_cnn_phase2.ipynb        # Phase 2: Final 9-class custom CNN architecture
 ┣ 📜 ku_building_cnn_phase3.ipynb        # Phase 3: Transfer learning (MobileNetV2) & Fine-tuning
 ┣ 📊 detailed_model_outputs.xlsx         # Batch prediction results for all test images
 ┣ 📊 model_accuracy_analysis.xlsx        # Final accuracy leaderboard comparing Phase 2 & 3
 ┣ 📜 requirements.txt                    # Python environment dependencies
 ┣ 📜 .gitignore                          # Git tracking rules
 ┗ 📜 README.md                           # Project documentation
```

## 🚀 How to Run Locally

**1. Clone the repository**

```bash
git clone https://github.com/sahbaz00/ku-buildings-classifier.git
cd ku-buildings-classifier

```

**2. Install Dependencies**

```bash
pip install -r requirements.txt

```

**3. Setup the Data**
Download the dataset from Kaggle and extract it so that your `dataset` folder is populated.

**4. Run Inference on a New Image**
You can use our pre-trained model to classify your own photos! Ensure your paths inside `implementation.py` point to your local directories, then run:

```bash
by loading uploaded models from "models" folder, and predict() method
(The script will load `fine_tuned.keras` and `class_mapping.json` to process images from the `test_images` directory).

```

## 👥 Contributors

* **Shahbaz Khalilli** https://www.linkedin.com/in/shahbaz-khalilli0/ * **Anar Jafarli** https://www.linkedin.com/in/anar-jafarli-97a311203/ ---

*Code licensed under MIT. Dataset licensed under CC BY 4.0.*

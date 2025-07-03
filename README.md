# OncoVue 🔬
**AI-Enhanced Radiology Image Analysis for Multicancer Detection**

OncoVue is a deep learning-powered system that analyzes radiological CT scan images for early and accurate multicancer detection. It automates the diagnostic process using a Convolutional Neural Network (CNN) and advanced computer vision techniques, aiming to assist radiologists with precision, efficiency, and standardization.

---

## 🚀 Project Overview

Early diagnosis is key to successful cancer treatment. However, manual interpretation of CT scans can be time-consuming and prone to human error. OncoVue addresses this issue by automating the image analysis process using AI.

The system follows a modular pipeline:
1. **Data Acquisition**
2. **Image Preprocessing**
3. **ROI Segmentation**
4. **Feature Extraction**
5. **Classification (CNN)**
6. **Result Visualization**

---

## 🧠 Features

- **Deep Learning (CNN - ResNet-50)** based classification
- **Edge detection & thresholding** for ROI segmentation
- **Multi-cancer detection** from CT scans
- **DICOM integration** for CT scanner communication
- **Secure & scalable architecture**
- **User-friendly GUI** for radiologists
- **Transfer learning** to reduce training time and increase accuracy

---

## 🛠️ Technology Stack

| Component         | Tools/Frameworks                                  |
|------------------|----------------------------------------------------|
| Programming       | Python                                             |
| Deep Learning     | TensorFlow / PyTorch (ResNet-50)                  |
| Image Processing  | OpenCV                                            |
| Data Format       | DICOM                                             |
| GUI               | Tkinter / Streamlit / Flask (if web-based)        |
| Deployment        | Local Machine / Cloud (optional future support)   |

----------------------------------------------------------------------------
## ⚙️ Installation

Follow these steps to set up **OncoVue** locally:

### Prerequisites

- Python 3.8+
- pip
- Git
- (Optional) Virtualenv

### Download Model

You need to download the trained model file from [Google Drive](https://drive.google.com/file/d/1_-ZmD-7rEn2S0KlETGmczkvxfpq2V7o3/view?usp=sharing).


### 1. Clone the Repository

```bash
git clone https://github.com/your-username/oncovue.git
cd oncovue
```

### 2. Create and Activate Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

> Example dependencies:
> `torch`, `torchvision`, `opencv-python`, `pydicom`, `numpy`, `streamlit`, `scikit-learn`

### 4. Add CT Scan Images

Place your DICOM images inside the `data/` folder.

### 5. Run the Application

If using Streamlit:

```bash
streamlit run gui/app.py
```

If using Tkinter or CLI:

```bash
python main.py
```

---------------------------------------------------------------------------

## 📊 Methodology

### 🖼️ 1. Data Acquisition
- Collect CT scan images from hospitals or open datasets.
- Ensure anonymization and compliance with data privacy.

### 🧹 2. Pre-processing
- Noise reduction, artifact removal
- Image intensity standardization
- Augmentation (rotation, flipping)

### 🔍 3. Segmentation
- Identify Regions of Interest (ROIs) using edge detection and thresholding
- Advanced options: U-Net for precise segmentation

### 🧬 4. Feature Extraction
- Geometric features: size, shape, texture, intensity
- Spatial relationships between ROIs

### 🧠 5. Classification
- Use of **ResNet-50** with transfer learning for cancer type prediction
- Fine-tuned for binary (cancerous / non-cancerous) or multi-class (cancer types) classification

### 📈 6. Evaluation Metrics
- Accuracy
- Sensitivity & Specificity
- Area Under ROC Curve (AUC)

---

## 🔧 Modules

| Module                | Description |
|-----------------------|-------------|
| `data_acquisition`    | Fetches and formats DICOM images |
| `preprocessing`       | Denoising, normalization, augmentation |
| `segmentation`        | Detects ROIs in the scan |
| `feature_extraction`  | Extracts features used by CNN |
| `classification`      | Deep learning model prediction |
| `visualization`       | GUI / visual output overlaying CT scans |

---

## 📦 Deployment

- Designed to integrate with **hospital CT systems**
- Complies with **DICOM** standards
- Supports **batch processing**
- Scalable and **privacy-compliant**

---

## 🎯 Goals

- Reduce diagnostic delays
- Assist radiologists with AI second opinion
- Improve early-stage cancer detection accuracy
- Adaptable system for multiple cancer types

---

## 📚 References

- [ResNet: Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Medical Imaging with Deep Learning](https://www.midl.io/)

---

## ✨ Future Work

- Expand to MRI and PET scan support
- Integration with hospital PACS/RIS
- Use of Explainable AI (XAI) for clinical interpretability
- Real-time analysis with cloud deployment

---

## 📎 License

This project is for academic use. For commercial or clinical deployment, please consult with the developers and adhere to medical device regulations.

---

✍️ **Prepared by:** Ajith Mathew  
📧 **Contact:** [ajithm804@gmail.com](mailto:ajithm804@gmail.com)


---




# Automated Weapon Detection in X-ray Luggage Scans  
*AI-driven Data Analytic Track*

---

## 🚀 Project Overview
Security screenings rely on X-ray scans to catch prohibited items in luggage. Our project builds a deep-learning pipeline inspired by “Occluded Prohibited Items Detection: An X-ray Security Inspection Benchmark and De-occlusion Attention Module” to automatically flag weapons under varying levels of occlusion, material differences, and low-frequency occurrences.

**Key Challenges**  
- **Concealment**: Overlapping objects hide weapons.  
- **Material Variance**: X-ray appearance changes across metals, plastics, etc.  
- **Class Imbalance**: Prohibited items are rare in the dataset.  
- **Human Fatigue**: Long hours → higher error rates.

**Our Approach**  
1. **Dataset Preparation**  
   - OPIXray & EDS collections, augmented with occlusion simulation.  
   - Versioned via DVC for reproducibility.  
2. **Model Architecture**  
   - DOAM backbone + focal-loss integration.  
   - Attack simulation (X-Adv) and multi-level occlusion (OL1–OL3).  
3. **Evaluation & Testing**  
   - mAP, precision, recall & F1-score on adversarial and standard baselines.  
   - Grad-CAM heatmaps for interpretability.  
4. **Deployment**  
   - TensorRT optimization (<100 ms latency).  
   - Web UI with threat highlighting, validated on live airport scans.

---

## 🧑‍🤝‍🧑 Team Contributions & Task Distribution

| Module                                | Primary Owner | Shadow        | Support                                   |
|---------------------------------------|---------------|---------------|-------------------------------------------|
| **1. Dataset Prep & Preprocessing**   | Raj           | Akshar        | Manu & Dhruvil                            |
| **2. Model Architecture & Impl.**     | Dhruvil       | Raj           | Akshar & Manu                             |
| **3. Model Evaluation & Testing**     | Manu          | Dhruvil       | Raj & Akshar                              |
| **4. Deployment & Integration**       | Akshar        | Manu          | Dhruvil & Raj                             |
| **5. Env. Setup (TF/PyTorch, DVC)**   | **Collaborative** – All members share setup and focal-loss code. |
| **6. Documentation & Reporting**      | **Collaborative** – Ongoing docs, final report, and presentation. |

---
## ⚙️ Setup & Running Locally
1. **Clone the Repo**  
   ```bash
   git clone https://github.com/your-org/xray-weapon-detection.git
   cd xray-weapon-detection

2. **Run The Backend**  
   ```bash
   pip install -r requirements.txt
   python main.py

3. **Run the Frontend**  
   ```bash
   cd Yolo-predictor
   npm install
   npm start

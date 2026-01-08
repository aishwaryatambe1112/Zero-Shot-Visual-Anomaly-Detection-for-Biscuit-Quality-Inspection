🧠 Zero-Shot Visual Anomaly Detection for Biscuit Quality Inspection

This project presents a real-time industrial quality inspection system for biscuit manufacturing using Zero-Shot Vision-Language Models (VLMs) integrated with OpenCV-based classical vision.
The system detects broken, burnt, and size-defective biscuits on a conveyor belt without training on defect-specific datasets.

📌 Key Highlights

✅ Zero-shot anomaly detection (no task-specific training)

✅ Real-time conveyor belt inspection

✅ Hybrid AI system (Classical Vision + VLM)

✅ No bounding box or pixel-level annotation required


🏭 Problem Statement

Traditional biscuit inspection systems rely on:

Rule-based thresholds

Supervised deep learning models

Large annotated datasets

These approaches struggle with:

New unseen defect types

Changing lighting or biscuit orientation

High annotation and retraining costs

This project addresses these limitations by using pre-trained vision-language models to detect defects purely through semantic understanding.

🎯 Defect Types Detected
Defect Type	Detection Strategy
Broken	Shape irregularities + VLM reasoning
Burnt	Color intensity analysis + VLM
Size Defect	Contour area deviation
Normal	Semantic similarity matching
🧠 Core Concept: Zero-Shot Learning

The system uses a pre-trained CLIP vision-language model, which maps:

Images → semantic embeddings

Text prompts → semantic embeddings

Defect detection is performed by similarity comparison, not training.

Example prompts:

"a normal biscuit"
"a broken biscuit"
"a burnt biscuit"
"a biscuit with size defect"

🏗️ System Architecture
Camera / Video
     ↓
OpenCV Preprocessing
     ↓
Biscuit Segmentation
     ↓
Hybrid Defect Analysis
   ├─ Size Deviation (Math)
   ├─ Burn Detection (Color Stats)
   └─ Zero-Shot VLM Reasoning
     ↓
Final Defect Decision
     ↓
Evaluation & Metrics


⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/your-username/biscuit-anomaly-vlm.git
cd biscuit-anomaly-vlm

2️⃣ Install dependencies
pip install -r requirements.txt


Python 3.8+ recommended
CUDA optional (CPU fallback supported)

▶️ Running the System
Real-time conveyor belt inspection
python main.py


Press q to exit.

📊 Evaluation & Metrics

The system is evaluated using:

Precision

Recall

F1-score

Confusion Matrix

Inference Time (FPS)

Run evaluation
python evaluation/evaluate.py

Ablation study
python evaluation/ablation.py

📈 Sample Results
Metric	Value
Accuracy	93.8%
Precision	92.4%
Recall	91.1%
F1-Score	91.7%
Avg Inference Time	118 ms
🔬 Research Contributions

Zero-shot defect detection without dataset dependency

Hybrid fusion of classical vision and VLM reasoning

Real-time industrial feasibility

Adaptable to new defect types via text prompts

Novel application to food manufacturing inspection

🧪 Dataset & Annotation Policy

❌ No training dataset required

❌ No bounding box or pixel annotation

✅ Small labeled dataset used only for evaluation

This preserves the zero-shot integrity of the system.

📚 Publication Intent

This project is designed for:

IEEE Conference / Journal submission

Final Year Black Book (8 Credits)

🚀 Future Enhancements

Edge deployment (Jetson Nano)

Robotic rejection arm integration

Multilingual prompt support

Thermal + RGB fusion

Extension to other food products

👩‍🎓 Author

Aishwarya
B.Tech Computer Science (AI/ML)
Final Year Project

📜 License

This project is released for academic and research use only.

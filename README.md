# Edge AI Optimizer: Synthetic Architect Dashboard 🧠⚡

![UI Preview](frontend/public/UI.png) *(Note: Replace this path with the actual path to your UI screenshot if you added one to your repo!)*

An end-to-end, full-stack application designed to benchmark and visualize the real-world trade-offs of deploying Convolutional Neural Networks on edge devices. 

This project evaluates a **ResNet-18** architecture trained on **CIFAR-10**, comparing standard 32-bit floating-point (FP32) inference against **Mixed Precision (FP16)** and **Quantization-Aware Training (INT8)**. The custom "Synthetic Architect" dashboard provides real-time analytics on inference latency, confidence distribution, and memory footprint.

## 🚀 Key Features
* **Real-Time Inference Engine:** Flask backend running parallel inference across three separate PyTorch model states simultaneously.
* **Dynamic Analytics Dashboard:** React + Tailwind CSS frontend visualizing latency speedups and memory compression.
* **Massive Memory Compression:** Demonstrated a **74.6% reduction** in model footprint (42.7 MB down to 10.8 MB) using INT8 QAT.
* **Implicit Regularization:** Empirically visualized how FP16 mixed precision training reduces overfitting and improves generalization on out-of-distribution high-resolution images.

---

## 📂 Project Structure
```text
edge-ai-optimizer/
│
├── backend/                  # Python/Flask API & PyTorch Inference Engine
│   ├── app.py                # Main server script
│   ├── requirements.txt      # Python dependencies
│   └── saved_models/         # Directory for generated .pth weights (Empty by default)
│
├── frontend/                 # React/Vite UI Dashboard
│   ├── src/                  # React components and styling
│   ├── package.json          # Node.js dependencies
│   └── vite.config.js        # Vite configuration
│
├── notebooks/                # Jupyter Notebooks for Training & Quantization
│   ├── 01_baseline.ipynb     # FP32 Training Pipeline
│   ├── 02_mixed_precision.ipynb # FP16 Training with AMP
│   ├── 03_qat.ipynb          # INT8 Quantization-Aware Training
│   └── 04_evaluate.ipynb     # CLI evaluation metrics
│
└── README.md




🛠️ Prerequisites
To run this project locally, ensure you have the following installed:

Python 3.9+

Node.js 18+ & npm

Optional but recommended: A CUDA-enabled GPU for retraining the models.

💻 Installation & Local Setup
Step 1: Clone the Repository

git clone [https://github.com/dev-by-abhyansh/edge-ai-optimizer.git](https://github.com/dev-by-abhyansh/edge-ai-optimizer.git)
cd edge-ai-optimizer

Step 2: Generate the Model Weights
Because GitHub restricts files over 100MB, the compiled .pth model weights are not included in this repository. You must train/generate the models before running the backend.

1. Navigate to the notebooks folder.

2. Run 01_baseline.ipynb, 02_mixed_precision.ipynb, and 03_qat.ipynb sequentially.

3. Ensure the resulting .pth files are saved into the backend/saved_models/ directory.

Step 3: Start the Backend (Flask + PyTorch)
Open a terminal and set up your Python environment:
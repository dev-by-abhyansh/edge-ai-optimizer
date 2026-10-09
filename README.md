# Edge AI Optimizer: Synthetic Architect Dashboard 

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
│   └── saved_models/         # Pre-trained .pth weights (FP32, FP16, INT8)
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

💻 Installation & Local Setup
Step 1: Clone the Repository

git clone [https://github.com/dev-by-abhyansh/edge-ai-optimizer.git](https://github.com/dev-by-abhyansh/edge-ai-optimizer.git)
cd edge-ai-optimizer

(Note: The pre-trained PyTorch weights for FP32, FP16, and INT8 are already included in the backend/saved_models/ directory, so you can run the dashboard immediately!)

### Step 2: Start the Backend (Flask + PyTorch)
Open a terminal in the project root directory and set up the environment:

```bash
# Create the environment
conda create -p edge-ai python=3.9 -y

# Activate the environment
conda activate ./edge-ai

# Navigate to the backend and install dependencies
cd backend
pip install -r requirements.txt

# Start the server
python app.py

The Flask API will start running on http://127.0.0.1:5000.


Step 3: Start the Frontend (React + Vite)
Open a new terminal window (keep the backend running):

Bash
cd frontend

# Install Node dependencies
npm install

# Start the Vite development server
npm run dev


The React dashboard will be available at http://localhost:5173.


🧪 Methodology & Results
This project was developed as part of a research initiative at the SRM Institute of Science and Technology.

Our comparative analysis yielded the following conclusions:

Memory: QAT successfully compressed the ResNet-18 model by 74.6%, strictly satisfying the constraints required for microcontroller deployment.

Accuracy: Precision reduction did not destroy semantic understanding. The INT8 model achieved competitive accuracy (88.87%) compared to the FP32 baseline (86.23%).

Generalization: FP16 Mixed Precision introduced beneficial computational noise during backpropagation, acting as an implicit regularizer that prevented the network from overfitting to training artifacts.

👨‍💻 Authors
Kunwar Abhyansh Visen * Sarthak Bhardwaj * Kaustav Roy* Manchineella Suryanarayana

Manchineella Suryanarayana

Under the guidance of: Dr. Prince Chelladurai S, Dept. of Computational Intelligence, SRM Institute of Science and Technology.

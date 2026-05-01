# 🏭 Manufacturing Process Throughput Optimizer

An interactive web application that optimizes multi-stage manufacturing production schedules using Mathematical Programming (MILP).

## 🚀 Overview
This tool solves a time-dependent production optimization problem. It calculates the optimal number of units to process at each stage every hour to maximize total throughput while respecting complex inter-process dependencies, lead times, and capacity bottlenecks.

## ✨ Key Features
- **Mathematical Optimization**: Uses Pyomo and the CBC solver to find the globally optimal production schedule.
- **Interactive Simulation**: Adjust operating hours, process capacities, and bottleneck limits in real-time.
- **Sensitivity Analysis**: Automatically sweeps constraint parameters to identify the most impactful "pivot points" for production gains.
- **Rich Visualization**: Includes interactive process flow diagrams, stacked bar charts, and utilization heatmaps powered by Plotly.
- **Automated Commentary**: Generates insights into bottleneck hours and process performance.

## 🛠️ Technology Stack
- **Language**: Python 3.x
- **UI Framework**: Streamlit
- **Optimization**: Pyomo (Python Optimization Modeling Objects)
- **Solver**: COIN-OR CBC (Linear Programming Solver)
- **Visualization**: Plotly & Pandas

## 📦 Installation & Local Setup

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd <repo-name>
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install the Solver (CBC)
- **Mac (Homebrew)**: `brew install cbc`
- **Linux (Ubuntu)**: `sudo apt-get install coinor-libcbc-dev`
- **Windows**: Download binaries from [COIN-OR project](https://github.com/coin-or/Cbc)

### 4. Run the App
```bash
streamlit run app.py
```

## ☁️ Deployment
This project is configured for **Streamlit Community Cloud**.
- `requirements.txt`: Handles Python dependencies.
- `packages.txt`: Automatically installs the CBC solver in the cloud environment.

## 📝 Problem Formulation
The model maximizes total throughput $\sum x_{m,t}$ subject to:
- **C1**: Process supply dependencies.
- **C2**: Multi-hour lead-time constraints.
- **C3**: Line-wide capacity ceilings.
- **C4**: Rolling bottleneck constraints (lagged dependencies).
- **C5**: Individual machine output bounds.

---
*Created as part of the Udemy Optimization Course project.*

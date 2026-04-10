# 🚁 HumanDiffusion
### Vision-Based Diffusion Trajectory Planner with Human-Conditioned Goals for UAV Search & Rescue

---

## 📄 Paper
📌 Accepted at **HRI 2026 (Late Breaking Report)**  
📎 [Paper PDF](HumanDiffusion_final_v3.pdf)  
📎 [Poster](HumanDiffusion_Poster_v2.pdf)

---

## 🧠 Abstract

Reliable human–robot collaboration in emergency scenarios requires autonomous systems that can detect humans, infer navigation goals, and operate safely in dynamic environments. 

This work presents **HumanDiffusion**, a lightweight image-conditioned diffusion planner that generates **human-aware navigation trajectories directly from RGB imagery**. The system combines YOLO-based human detection with diffusion-driven trajectory generation, enabling a UAV to approach a target person and deliver assistance without relying on prior maps or computationally intensive planning pipelines.

Trajectories are predicted directly in **pixel space**, ensuring smooth motion while maintaining a consistent safety margin around humans. The system is evaluated in both simulation and real-world indoor scenarios, demonstrating strong agreement between predicted and ground-truth trajectories and robust performance in human-centered navigation tasks.

---

## ⚡ Key Contributions

- **Human-Conditioned Goal Inference**
  - Navigation goal derived directly from detected human

- **Image-Conditioned Diffusion Planning**
  - End-to-end trajectory generation from RGB images

- **Map-Free Navigation**
  - No SLAM, no occupancy grids, no predefined waypoints

- **Sim-to-Real Deployment**
  - Trained in simulation and deployed in real-world UAV scenarios

---

## 🏗️ System Architecture

![Architecture](assets/architecture.png)

The system consists of two core modules:

1. **Perception Module**
   - YOLO detects humans
   - Bounding box center → navigation goal

2. **Diffusion Planner**
   - Inputs: RGB image + start + goal
   - Outputs: smooth trajectory in pixel space

---

## 🔄 Pipeline

```text
RGB Image → Human Detection (YOLO)
        ↓
Goal Extraction (Bounding Box Center)
        ↓
Start + Goal Encoding
        ↓
Diffusion Model (UNet)
        ↓
Pixel-Space Trajectory
        ↓
3D Projection → UAV Execution
```
---

## 🧮 Diffusion Model Formulation

### Forward Process

$$
x_t = \sqrt{\alpha_t} \, x_0 + \sqrt{1 - \alpha_t} \, \epsilon
$$

---

### Reverse Process

$$
x_{t-1} = \mu_t(x_t, \hat{x}_0) + \sigma_t z
$$

---

### Training Objective

$$
\mathcal{L} = \lambda_{\text{path}} L_{\text{path}} + \lambda_{\text{endpoint}} L_{\text{endpoint}}
$$

---

### Where:

- $L_{\text{path}}$ ensures trajectory reconstruction  
- $L_{\text{endpoint}}$ enforces accurate start and goal prediction  

---

## 📊 Dataset

- **Total samples:** 9800  
- **Training:** 8000  
- **Validation:** 1500  
- **Test:** 300  

### Dataset generated using:
- Simulated environments  
- A* planner for ground-truth trajectories  

---

## 📈 Results 

### 🧪 Simulation Performance

| Metric | Value |
|-------|------|
| Trajectory RMSE | **0.14 pixels** |
| Start/Goal RMSE | **0.045 pixels** |
| Trajectory IoU | **0.49** |

![Simulation Results](assets/simulation_results.png)
---

### 🌍 Real-World Performance

#### 🚑 Experiment 01: Accident Response
![Experiment 01](assets/real_exp_01.png)

#### 🌲 Experiment 02: Search & Locate (Occlusion)
![Experiment 02](assets/real_exp_02.png)

| Scenario | Success Rate |
|---------|-------------|
| Accident Response | **90%** |
| Search & Locate (Occlusion) | **70%** |
| **Overall** | **80%** |

---

## 🎥 Demo Video

[![Watch Demo](https://img.youtube.com/vi/fHh9eRGh49c/0.jpg)](https://www.youtube.com/watch?v=fHh9eRGh49c)

---

### 🧾 Observations

- Accurate trajectory reconstruction in pixel space  
- Reliable human-conditioned navigation  
- Robust performance under partial occlusions  

---

## ▶️ Running the Project

The entire pipeline is executed using:

```bash
python main.py
```
---

## 🧩 Code Structure & Flow
To understand the project, follow this order:

1. Dataset.py        → Dataset preparation and preprocessing
2. Model.py          → High-level model wrapper
3. Unet.py           → Core UNet architecture
4. Perception.py     → Human detection and goal extraction
5. Diffusion.py      → Diffusion process implementation
6. Loss.py           → Training objectives and loss functions
7. Train.py          → Training loop
8. Dataloader.py     → Data loading pipeline
9. Main.py           → Entry point (execution script)

---

## 📁 Repository Structure

HumanDiffusion/
│── Dataset.py
│── Model.py
│── Unet.py
│── Perception.py
│── Diffusion.py
│── Loss.py
│── Train.py
│── Dataloader.py
│── Main.py
│
│── assets/
│   ├── architecture.png
│   ├── results.png
│
│── README.md

---

## Citation

@inproceedings{10.1145/3776734.3794549,
author = {Batool, Faryal and Zhura, Iana and Serpiva, Valerii and Khan, Roohan Ahmed and Valuev, Ivan and Tokmurziyev, Issatay and Tsetserukou, Dzmitry},
title = {HumanDiffusion: A Vision-Based Diffusion Trajectory Planner with Human-Conditioned Goals for Search and Rescue UAV},
year = {2026},
doi = {10.1145/3776734.3794549},
booktitle = {Companion Proceedings of the 21st ACM/IEEE International Conference on Human-Robot Interaction},
pages = {1023–1027},
series = {HRI Companion '26}
}
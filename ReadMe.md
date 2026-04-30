# HumanDiffusion
### Vision-Based Diffusion Trajectory Planner with Human-Conditioned Goals for UAV Search & Rescue

---

## Paper 📄
Accepted at **HRI 2026 (Late Breaking Report)**  
[Paper PDF](HumanDiffusion_final.pdf)  
[Poster](HumanDiffusion_Poster.pdf)

---

## Introduction 🚁

**HumanDiffusion** is a vision-based trajectory planning framework for human-centered UAV navigation in search-and-rescue scenarios. The main idea of the paper is to generate safe and smooth trajectories directly from RGB observations while conditioning the predicted path on the detected human location. 

---

## Abstract 🧠

Reliable human-robot collaboration in emergency scenarios requires autonomous systems that can detect humans, infer navigation goals, and operate safely in dynamic environments.

This work presents **HumanDiffusion**, a lightweight image-conditioned diffusion planner that generates **human-aware navigation trajectories directly from RGB imagery**. The system combines YOLO-based human detection with diffusion-driven trajectory generation, enabling a UAV to approach a target person and deliver assistance without relying on prior maps or computationally intensive planning pipelines.

Trajectories are predicted directly in **pixel space**, ensuring smooth motion while maintaining a consistent safety margin around humans. The system is evaluated in both simulation and real-world indoor scenarios, demonstrating strong agreement between predicted and ground-truth trajectories and robust performance in human-centered navigation tasks.

---

## Key Contributions ⚡

- **Human-Conditioned Goal Inference**: Navigation goal derived directly from the detected human.
- **Image-Conditioned Diffusion Planning**: End-to-end trajectory generation from RGB images.
- **Map-Free Inference**: No SLAM, no predefined waypoint sequence during deployment.
- **Sim-to-Real Deployment**: Trained in simulation and deployed in real-world UAV scenarios.

---

## System Architecture 🏗️

![Architecture](assets/architecture.png)

The system consists of two core modules:

1. **Perception Module**
   Detects the human and extracts a goal location from the image.
2. **Diffusion Planner**
   Predicts a smooth trajectory from the RGB image, start point, and human-conditioned goal.

---

## Inference Pipeline 🔄

```text
RGB Image -> Human Detection (YOLO)
        ->
Goal Extraction (Bounding Box Center)
        ->
Start + Goal Encoding
        ->
Diffusion Model (UNet)
        ->
Pixel-Space Trajectory
        ->
3D Projection -> UAV Execution
```

---

## Dataset 📊

- **Total samples:** 9800  
- **Training:** 8000  
- **Validation:** 1500  
- **Test:** 300

### Dataset generation 🧩

As described in the paper, the dataset is built from simulated indoor scenes where RGB observations are paired with traversability or occupancy information and planner-generated trajectories. Human-aware goals are defined from the perceived target location, and ground-truth paths are saved as pixel-space waypoints for diffusion training.

The repository also contains a `testing_samples/` directory with example samples that help verify the expected training-data structure and output format.

### Dataset annotation 📝

The `dataset_annotation/` folder contains two utilities used to prepare the dataset:

1. `Annotating_dataset_v2.py`
   Interactive annotation and trajectory-generation tool.
2. `CSV_to_npy.py`
   Converter that transforms saved annotations into training samples under a global `Training samples/` folder.

This annotation workflow follows the dataset preparation idea used in the paper: a start position and human-conditioned goal are paired with map information, a planner generates a safe path, and the saved trajectories are then converted into normalized training samples for diffusion learning.

### Dataset annotation workflow 🛠️

#### Step 1: Annotate trajectories ✍️

Run:

```bash
python dataset_annotation/Annotating_dataset_v2.py --root "PATH_TO_DATASET_ROOT"
```

Supported layout:

```text
<root>/1/img_0001/maps/
<root>/1/img_0002/maps/
...
```

or

```text
<root>/1/maps/
<root>/2/maps/
...
```

Inside each `maps/` directory, the tool looks for:

- `occupancy_grid.npy` or `occupancy_grid.png`
- `traversability_map.npy` as fallback
- an RGB image for visualization

Controls:

- Left click: add one or more start points
- `g`: switch to goal selection mode
- Next click: set the goal and compute path(s)
- `n`: save and continue
- `s`: skip current sample
- `c`: clear current selections

Saved files in each `maps/` folder include:

- `selections.csv`
- `astar_original_waypoint_count_start{k}.csv`
- `astar_normalized_original_waypoint_count_start{k}.csv`
- `astar_fixed_waypoint_count_start{k}.csv`
- `astar_normalized_fixed_waypoint_count_start{k}.csv`

#### Step 2: Convert annotations to training samples 📦

Run:

```bash
python dataset_annotation/CSV_to_npy.py --root "PATH_TO_DATASET_ROOT" --actions-log "PATH_TO_DATASET_ROOT/actions_log.csv"
```

This creates sample folders such as:

```text
Training samples/
  sample_000001/
  sample_000002/
  ...
```

Each sample contains:

- `rgb.png`
- `trav_map.npy` if available
- `occ_map.png` if available
- `traj_xy.npy`
- `start_xy.json`
- `end_xy.json`

---

## Diffusion Training Pipeline 🧪

Before training, each scene is transformed into a supervised sample containing an RGB observation, a start point, an endpoint, and a planner-generated reference trajectory. The annotation and conversion scripts in `dataset_annotation/` automate this pipeline and produce normalized trajectory files used by the diffusion model. Each image represents a first-person view of the indoor environment. For every scene, multiple start locations and a single goal location are selected using an annotation interface. Multiple start positions are included to reflect realistic operating conditions, since a drone may begin from any location within the environment. By providing diverse start-goal combinations for the same scene, the dataset encourages the model to learn how to generate a feasible path from a wide range of initial positions to the desired destination. This also helps the diffusion model capture both straight and curved trajectory patterns, improving its generalization across different navigation scenarios.

### Stage 1: Trajectory generation and annotation 🗺️
The first stage builds scene-level annotations by combining RGB, map information, and start-goal selections. A planner then generates a safe path using the occupancy map and start and goal point information that will serve as supervision.

![Dataset generation step 1](assets/Dataset_generation_01.jpg)

![Dataset generation step 2](assets/Dataset_generation_02.jpg)

![Dataset generation step 3](assets/Dataset_generation_03.jpg)

### Stage 2: Sample construction 🧱

The generated trajectories are converted into sample folders containing RGB input, optional occupancy and traversability data, normalized waypoints, start and endpoint metadata.

### Stage 3: Diffusion training input preparation 🎯

These processed samples are then fed into the diffusion training pipeline, where the model learns to reconstruct trajectories directly from image-conditioned context.

---

## Diffusion Model Formulation 🧮

### Forward Process ➡️

$$
x_t = \sqrt{\alpha_t} \, x_0 + \sqrt{1 - \alpha_t} \, \epsilon
$$

### Reverse Process ⬅️

$$
x_{t-1} = \mu_t(x_t, \hat{x}_0) + \sigma_t z
$$

### Training Objective 🎯

$$
\mathcal{L} = \lambda_{\text{path}} L_{\text{path}} + \lambda_{\text{endpoint}} L_{\text{endpoint}}
$$

Where:

- $L_{\text{path}}$ ensures trajectory reconstruction.
- $L_{\text{endpoint}}$ enforces accurate start and goal prediction.

---

## Results 📈

### Simulation Performance 🧪

| Metric | Value |
|-------|------|
| Trajectory RMSE | **0.14 pixels** |
| Start/Goal RMSE | **0.045 pixels** |
| Trajectory IoU | **0.49** |

![Simulation Results](assets/simulation_results.png)

---

### Real-World Performance 🌍

#### Experiment 01: Accident Response 🚑
![Experiment 01](assets/real_exp_01.png)

#### Experiment 02: Search & Locate (Occlusion) 🔍
![Experiment 02](assets/real_exp_02.png)

| Scenario | Success Rate |
|---------|-------------|
| Accident Response | **90%** |
| Search & Locate (Occlusion) | **70%** |
| **Overall** | **80%** |

---

## Demo Video 🎥

[![Watch Demo](https://img.youtube.com/vi/fHh9eRGh49c/0.jpg)](https://www.youtube.com/watch?v=fHh9eRGh49c)

---

## Observations 🧾

- Accurate trajectory reconstruction in pixel space.
- Reliable human-conditioned navigation.
- Robust performance under partial occlusions.

---

## Running the Project ▶️

The main project entrypoint is:

```bash
python main.py
```

---

## Code Structure & Flow 🧩

To understand the project, a useful reading order is:

1. `src/data_loader/data_loader.py` -> dataset loading and preprocessing
2. `src/models/model.py` -> model wrapper
3. `src/models/backbones/unet.py` -> core UNet architecture
4. `src/models/perception.py` -> human detection and goal extraction
5. `src/models/diffusion.py` -> diffusion process implementation
6. `src/loss.py` -> training objectives
7. `src/train.py` -> training loop
8. `main.py` -> project entrypoint
9. `dataset_annotation/` -> annotation and dataset conversion utilities

---

## Repository Notes 📁

- `dataset_annotation/` contains the refactored annotation and sample-conversion tools.
- `testing_samples/` contains example samples for quick validation and debugging.
- `assets/` contains architecture, training-pipeline, and evaluation figures used in this README.

---
## Acknowledgements

This work was inspired by:
https://github.com/jingGM/DTG

We reimplemented and extended the approach for our task.

---
## Citation 📚

```bibtex
@inproceedings{10.1145/3776734.3794549,
  author = {Batool, Faryal and Zhura, Iana and Serpiva, Valerii and Khan, Roohan Ahmed and Valuev, Ivan and Tokmurziyev, Issatay and Tsetserukou, Dzmitry},
  title = {HumanDiffusion: A Vision-Based Diffusion Trajectory Planner with Human-Conditioned Goals for Search and Rescue UAV},
  year = {2026},
  doi = {10.1145/3776734.3794549},
  booktitle = {Companion Proceedings of the 21st ACM/IEEE International Conference on Human-Robot Interaction},
  pages = {1023--1027},
  series = {HRI Companion '26}
}
```

# Mars Rover Project

## Overview

The Mars Rover Project aims to replicate the objectives of the Mars Sample Return (MSR) mission by using computer vision techniques to explore and map the Martian environment. Inspired by previous Mars missions, our goal is to develop a Mars Sample Return-like program that achieves significant progress in mapping and collecting data from Mars.

![Main img](MI.jpg)

## Objectives

1. **Mapping and Fidelity:**
   - Map at least 40% of the Martian environment with 60% fidelity.
   - Differentiate navigable terrain, obstacles, and rock samples on the map.

2. **Rock Collection:**
   - Locate and identify at least one rock sample for retrieval.

3. **Debugging Mode:**
   - Implement a debugging mode to visualize each step of the rover's operations.

## Installation

To get started with the Mars Rover Project:

    1. Create a Directory with command mkdir asu
    2. Traverse to the directory using the command cd asu 
    3. Clone the repository using the command git clone https://github.com/Omar-Mohamed-Ibrahim-Alsayed/MarsRoverProject.git
    4. Create a conda environment using the command conda create --name nasef --file cv1.txt 
    5. Install python-socketio version 4.6.1 using the command conda install python-socketio=4.6.1

    6. Active the environment using the command source activate nasef 
    7. Run the environment using the command python driver_rover.py 
    8. Run Roversim then click ok

https://drive.google.com/file/d/1U4j1YfNIvFSPYzK8TJsxLAewJdNrV1au
for step by step video


## Map Legend

- Red: Non-navigable terrain (obstacles)
- Blue: Navigable terrain (path)
- White: Rock samples

## Results

During simulations, the project achieved the following objectives for Phase One:

| Iterations | Mapping (%) | Time (50%) | Time (60%) | Success |
|------------|-------------|------------|------------|---------|
| 1          | 71%         | 64%        | 60%        | Succeeded |
| 2          | Stuck       | Stuck      | Stuck      | Stuck    |
| 3          | 68%         | 63%        | 60%        | Succeeded |
| ...        | ...         | ...        | ...        | ...     |
| 10         | 67%         | 70%        | 66%        | Succeeded |

## Step-by-Step Pipeline

The Mars Rover Project operates in a closed-loop feedback system with the following pipeline:

1. **Perspective Transform:** Obtain a top-down view of the Martian terrain.
2. **Color Thresholding:** Identify navigable terrain, obstacles, and rocks.
3. **Coordinate Transformations:** Convert image coordinates to rover and world coordinates.
4. **Geometric Transformations:** Apply rotations, scaling, and clipping.
5. **Mapping:** Generate and update the world map with navigable terrain, obstacles, and identified rocks.
6. **Debugging:** Visualize each step of the pipeline for debugging purposes.

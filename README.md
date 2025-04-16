# FBK-internship Repository

Welcome to my **FBK-internship** repository! This repository serves as a record of the work carried out during the internship. It primarily contains Jupyter Notebooks that document and analyze various tasks, experiments, and findings.

---

## Folder Structure and Navigation

The repository is structured into folders and subfolders, each containing Jupyter Notebooks and potentially supporting data files. Below is a detailed navigation guide:

### Top-Level Directories
- The top-level directories of this repository are organized according to the area of interest:
    - Main focus is the **GDS** (Geo Data Science) directory, it contains all my work done for my internship project;
    - RL directory is secondary work on Reinforcement Learning, not updated and not important, maybe I'll go deeper in that one day

### Subdirectories:
1. **`GDS/CR`**:
   - This folder contains notebooks that show _everything_ I've learned and tried, it is applied on datasets obtained from OSM of small cities that I'm comfortable analizing due to have lived there, so that I could have immediate feedback from my knowledge.
   -  **Notebooks**:
       - `CR.ipynb`: First weeks work, focused on processing and dealing with Geospatial Data, experiments and trials regard shortest routes and traffic simulation, at the end there is pre-processing useful for the following notebooks.
       - `CR-2.ipynb`: Probabbly main corpus of the work, implementation of Graphs ML models such as GAT and different Graph Transformers of various nature and comparisons.
       - `Attention.ipynb`: Extraction, processing and interpretation of the different kinds of attention that can be extracted from the different models.
2. **`GDS/Urbanity`**:
   - This folder contains notebooks that use datasets created with [Urbanity](https://urbanity.readthedocs.io/en/latest/).
   - **Notebook**:
     - `GAT_urbanity.ipynb`: Basic implementation of GAT based model following [Urbanity tutorial](https://urbanity.readthedocs.io/en/latest/notebooks/transductive_graph_ml.html)
     - `Paris_greenview.ipynb`: In this notebook the work done in the [GDS/CR](#Subdirectories) is reproduced with a way bigger dataset of Paris and node-level predictions.

---

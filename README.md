# Clinical Knowledge Graph and Bayesian Network for Medical Decision Support

## Overview
This project implements a full pipeline for transforming unstructured medical text into a **causal knowledge graph (KG)** and **Bayesian Network (BN)** to support **clinical decision-making**.

The pipeline extracts causal triples (e.g., *"High Blood Pressure → can lead to → Heart Failure"*) from **NHS clinical web pages** using large language models (LLMs). These triples are stored in **Neo4j** as a structured KG, then converted into a BN using `pgmpy`. A **Streamlit web app** allows interactive exploration, inference, and expert editing of probabilities.

This approach bridges **unstructured medical knowledge** and **probabilistic reasoning**, providing transparent, editable, and clinically meaningful decision support.

---

## Features
- **Knowledge Graph Construction**  
  - Extracts causal triples (risk factors, symptoms, treatments, guidance) from medical text.
  - Stores entities and relations in **Neo4j**, grouped by disease category.
  - Supports queries for symptoms, treatments, and causal relationships.

- **Bayesian Network Generation**  
  - Converts the KG into a BN with nodes, edges, and conditional probability distributions (CPDs).
  - Handles noisy-or priors, conditional dependencies, and expert overrides.
  - Supports inference queries (e.g., probability of *Heart Failure* given *Shortness of Breath* and *Obesity*).

- **Streamlit Application**  
  - Role-based UI: patients, experts, and admin modes.
  - Interactive sliders to adjust probabilities.
  - Evidence selection with real-time posterior probability updates.
  - Visualization of BN structure and inference results.
  - Neo4j integration for uploading, querying, and grouping nodes.

- **Expert Evaluation Tools**  
  - Editable CPDs for manual refinement.
  - Comparison between LLM-extracted probabilities and expert-adjusted ones.
  - Supports structured feedback loops for improving model reliability.

---

## 📂 Project Structure

```text
├── app.py                # Streamlit entrypoint
├── baysian.py            # Bayesian network builder (pgmpy)
├── neo4jUploader.py      # Neo4j uploader and query utilities
├── treatments.py         # BN inference for treatments
├── db/                   # SQLAlchemy + Alembic database models
├── models/               # Pydantic/ORM schemas
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
# Clinical Knowledge Graph to Bayesian Network Inference System

This project provides a pipeline for transforming unstructured clinical text into structured knowledge graphs and probabilistic reasoning models.  
It uses **Neo4j** for storing causal medical relationships, **pgmpy** for Bayesian Network construction, and **Streamlit** for interactive visualization and inference.  
The system supports **patients, experts, and admins** to input symptoms, run probabilistic inference, and refine model parameters for transparent medical decision support.  

---

## Features
- Extracts causal triples (symptom, treatment, condition) from NHS clinical text using LLMs.  
- Stores triples in **Neo4j** as a knowledge graph with categories (e.g., *Heart Failure*).  
- Builds **Bayesian Networks** from knowledge graphs using **pgmpy**.  
- Provides a **Streamlit interface** for:
  - Patients: input symptoms and view probabilistic predictions.  
  - Experts: edit probability tables and refine the model.  
  - Admins: manage categories and evaluations.  
- Supports **graph visualization** with highlighted target and evidence nodes.  
- Persists evaluations and expert feedback in a relational database (**SQLAlchemy + Alembic**).  

---

## Installation

1. Clone the repository:
git clone https://github.com/valentinafu/medical_llm_bn_pipeline.git
cd medical_llm_bn_pipeline
2.Create conda env
conda create -n medical_app python=3.10 -y
conda activate medical_app
Install dependencies
3.pip install requirement.txt
4.Populate .env file with NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
DATABASE_URL=sqlite:///./app.db
6.streamlit run app.py

7.Run the code :streamlit run app.py

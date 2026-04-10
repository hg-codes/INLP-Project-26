# Juris-Clarify (INLP-Project-26)

**Course Project for INLP 26** *A verifiable, multi-stage NLP pipeline for legal text simplification.*

🔗 **[View Pipeline Outputs (All Stages)](https://iiithydresearch-my.sharepoint.com/:u:/g/personal/manikya_pant_research_iiit_ac_in/IQAstkkZB-t6Q6w8OnpTkT91ASQC8y6xzgSyChYNSEdCA9k?e=8VQNnp)**

## Overview
**Juris-Clarify** translates complex legal contracts into accessible language while formally guaranteeing semantic faithfulness. Unlike standard generative LLMs, our Writer-Critic architecture prevents "semantic drift" by explicitly extracting constraints and verifying outputs using Natural Language Inference (NLI).

## Pipeline Architecture
The system is divided into five modular stages:

* **Stage 1: Hierarchical Clause Segmentation** Uses LegalBERT and a Conditional Random Field (CRF) to detect clause boundaries and structural metadata.
* **Stage 2: Constraint Extraction** Identifies core entities (Dates, Money, Parties) and legal obligations to prevent hallucination during rewriting.
* **Stage 3: Constraint-Aware Rewriting** A fine-tuned `BART` model generates simplified text while explicitly preserving the constraints extracted in Stage 2.
* **Stage 4: NLI Verification Loop** A `DeBERTa-v3` critic checks if the simplified text logically entails the original. If not, it rejects the text and triggers regeneration.
* **Stage 5: Persona Adaptation** A prompt-tuned `Flan-T5` model adapts the verified text to match the reading levels of specific audiences (Layperson, Student, or Professional).

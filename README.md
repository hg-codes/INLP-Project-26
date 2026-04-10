# Juris-Clarify (INLP-Project-26)

**Course Project for INLP 26** *A verifiable, multi-stage NLP pipeline for legal text simplification.*

🔗 **[View Pipeline Outputs (All Stages)](https://iiithydresearch-my.sharepoint.com/:f:/g/personal/manikya_pant_research_iiit_ac_in/IgApygMLEGiiQ4e1M0ajo6giAa-eLon4Dkp07YuuoTYNh4k?e=ZfXRp3)**

## Overview
**Juris-Clarify** translates complex legal contracts into accessible language while formally guaranteeing semantic faithfulness. Unlike standard generative LLMs, our Writer-Critic architecture prevents "semantic drift" by explicitly extracting constraints and verifying outputs using Natural Language Inference (NLI).

## Pipeline Architecture
The system is divided into five modular stages:

* **Stage 1: Hierarchical Clause Segmentation** Uses LegalBERT and a Conditional Random Field (CRF) to detect clause boundaries and structural metadata.
* **Stage 2: Constraint Extraction** Identifies core entities (Dates, Money, Parties) and legal obligations to prevent hallucination during rewriting.
* **Stage 3: Constraint-Aware Rewriting** A fine-tuned `BART` model generates simplified text while explicitly preserving the constraints extracted in Stage 2.
* **Stage 4: NLI Verification Loop** A `DeBERTa-v3` critic checks if the simplified text logically entails the original. If not, it rejects the text and triggers regeneration.
* **Stage 5: Persona Adaptation** A prompt-tuned `Flan-T5` model adapts the verified text to match the reading levels of specific audiences (Layperson, Student, or Professional).

## Results Folder Structure
```text
results/kaggle/working
├── exp1  (Stage 1)
│   ├── best_model
│   │   ├── config.json              # Model configuration settings
│   │   ├── model.pt
│   │   ├── tokenizer_config.json    # Tokenizer configuration settings
│   │   └── tokenizer.json           # Serialized tokenizer vocabulary
│   ├── best_threshold.json          # Optimal threshold computed for predictions
│   ├── data
│   │   ├── test.jsonl               # Stage 1 formatted test split
│   │   ├── train.jsonl              # Stage 1 formatted train split
│   │   └── val.jsonl                # Stage 1 formatted validation split
│   └── test_results.json            # Model evaluation results on the test data
├── stage1_predictions
│   ├── predicted_test.jsonl         # Model predictions on test subset
│   ├── predicted_train.jsonl        # Model predictions on train subset
│   └── predicted_val.jsonl          # Model predictions on validation subset
├── stage2_output
│   ├── checkpoints
│   │   └── best_model
│   │       ├── constraint_config.json # Stage 2 model configuration
│   │       ├── constraint_model.pt
│   │       ├── tokenizer_config.json
│   │       └── tokenizer.json
│   ├── demo_output.json             # Sample demonstration outputs from Stage 2
│   ├── eval_metrics.json            # Quantitative metrics for the Stage 2 model
│   ├── final_model
│   │   ├── constraint_config.json
│   │   ├── constraint_model.pt
│   │   ├── tokenizer_config.json
│   │   └── tokenizer.json
│   ├── test_constraints.json        # Test data with constraint pairs 
│   ├── train_constraints.json       # Training data with constraint pairs
│   ├── training_history.json        # Epoch-wise training and validation logs
│   └── val_constraints.json         # Validation data with constraint pairs
├── stage3_output
│   ├── evaluation_metrics.json      # Evaluation metrics from simplification
│   ├── simplified_test_clauses.json # Stage 3 text simplification results
│   ├── synthetic_pairs.json         # Synthetic positive/negative pairs generated for Stage 3
│   └── training_history.json        # Stage 3 epoch-wise progress logs
├── stage4_output
│   ├── human_review_queue.json          # Simplifications flagged by NLI that require human review
│   ├── stage4_evaluation_metrics.json   # Efficacy metrics post-NLI correction loop
│   └── stage4_simplified_clauses.json   # Refined simplifications fully conforming to NLI entailment
└── stage5_output
    └── persona_targets.json         # Final output mapped to user-specific personas
```

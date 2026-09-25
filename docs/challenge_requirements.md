# Amazon ML Challenge 2026 - Requirements & Guidelines

## 1. Challenge Objective
Build an ML solution that, given business records from 3 independent data sources with noisy and inconsistent fields, determines which records across sources refer to the same real-world business entity. 
Source 1 is the deduplicated reference source. The task is to find all matching records from Source 2 and Source 3 for each Source 1 entity.

## 2. Dataset Schema and Expected Files
All files are **tab-separated (`.tsv`)**. 
Expected columns:
- `entity_id`: Unique identifier (prefix S1-, S2-, or S3-).
- `business_name`: Entity name (with abbreviations, suffixes, typos).
- `business_address`: Address (partial, missing components, landmarks).
- `country`: String label (`US`, `India` in training; additionally `France` in test). Do not hardcode or filter these.

## 3. Training and Test Data Structure
- `dataset/train/train_source1.tsv`: Reference source.
- `dataset/train/train_source2.tsv`: Source 2 training records.
- `dataset/train/train_source3.tsv`: Source 3 training records.
- `dataset/test/test_source1.tsv`: Test reference source.
- `dataset/test/test_source2.tsv`: Source 2 test records.
- `dataset/test/test_source3.tsv`: Source 3 test records.

## 4. Ground-Truth Format
`dataset/train/train_ground_truth.tsv` contains two columns:
- `source1_entity_id`: Entity ID of Source 1 record.
- `matched_entity_ids`: Comma-separated list of matching entity IDs from Source 2/3. Empty if no matches.

## 5. Official Evaluation Formula
F_0.5 Score (β = 0.5) - Precision-heavy metric.
`F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)`
Computed as a macro-average per Source 1 entity, then averaged across all Source 1 entities.
Singletons (no matches) are included. Predicting an empty list correctly scores 1.0.

## 6. Submission File Requirements
The `output/` folder must contain two files:
1. `matching_results.tsv`: Final matches. Columns: `source1_entity_id`, `matched_entity_ids`.
2. `candidate_pairs.tsv`: Candidate set before final inference. Columns: `source1_entity_id`, `candidate_entity_ids`.

**Constraints:**
- Every Source 1 entity must appear exactly once.
- Empty `matched_entity_ids` for singletons.
- No duplicate IDs in lists.
- Only Source 2/3 IDs that exist in the test set.

## 7. Model and Licensing Restrictions
- Final model must be MIT/Apache 2.0 Licensed.
- Maximum size: 8 Billion parameters.

## 8. External Data and API Restrictions
- **STRICTLY PROHIBITED:** External Data Lookup (commercial APIs, government databases, geocoding, internet data augmentation).

## 9. Final Submission Package
- A single zip containing:
  - `output/matching_results.tsv` and `output/candidate_pairs.tsv`
  - `code/business_entity_resolution/src/` (Source code)
  - `code/business_entity_resolution/README.md` and `requirements.txt`
  - `Documentation_template.md` (Methodology document)

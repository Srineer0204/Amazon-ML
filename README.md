# Amazon ML Challenge 2026 — Business Entity Resolution

A machine learning system for identifying matching business records across multiple data sources, developed as part of the **Amazon ML Challenge 2026**.

## Overview

Business information collected from different sources can contain variations in:

- Business names
- Addresses
- Abbreviations
- Spelling
- Formatting
- Missing information

The same real-world business may therefore appear as different records across different sources.

This project aims to identify which records from **Source 2** and **Source 3** correspond to each business in **Source 1**, using business names, addresses, country information, and machine-learning-based similarity analysis.

## Approach
1. Data Validation
2. Data Preprocessing
3. Candidate Generation
4. Feature Engineering
5. Machine Learning
6. Evaluation

## Technology Stack
- Python Core development language
- Pandas	Data processing and TSV handling
- NumPy	Numerical operations
- RapidFuzz	Fuzzy string matching
- Scikit-learn	Feature engineering and machine learning
- CatBoost	Optional tree-based classification
- unittest	Automated testing
- Git	Version control

## Team Rudra
- Developed as a team project for the Amazon ML Challenge 2026.

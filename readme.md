# Small Molecule Bioactivity Benchmarks: Cell Count Baselines

## Overview
This repository contains code and datasets for evaluating small molecule bioactivity benchmarks using baseline models based on cell count features. The study investigates the predictive power of simple cell count features compared to high-dimensional Cell Painting profiles and gene expression data. It provides recommendations for best practices in machine learning applications for phenotypic profiling.

## Key Findings
- Many widely used bioactivity benchmarks are biased towards cytotoxicity and cell proliferation assays, which can be well-predicted using simple cell count features.
- Cell count-based logistic regression models perform comparably to complex machine learning models trained on Cell Painting and gene expression data for a large subset of bioactivity assays.
- Filtering benchmark datasets to focus on protein-target-specific assays reveals the added value of high-content phenotypic data.
- We propose best practices for benchmarking machine learning models for predicting bioactivity, including the importance of baseline cell count models.
- We introduce a curated dataset of protein-target-specific bioactivity assays with corresponding Cell Painting profiles.

## Repository Structure
```
The_Seal_Files/
│── The_Cell_Count_Files/      # Scripts for baseline cell count models
│── The_Hofmarcher_Files/      # Reanalysis of Hofmarcher et al. dataset
│── BAK_The_Moshkov_Files_v1/  # Backup Reanalysis of Moshkov et al. dataset considering all tasks
│── The_Ha_Files/              # Reanalysis of Ha et al. dataset
│── The_Sanchez-Fernandez_Files/ # Reanalysis of Sanchez-Fernandez et al.
│── The_Cross-Zamirski_Files/  # Label-free Cell Painting predictions
│── The_Comolet_Files/         # Evaluating dose-response effects
│── The_Seal_dataset/          # Curated protein-target-specific dataset
```

## Installation and Requirements
To run the analyses, ensure you have the following dependencies installed:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy jupyterlab
```

## Recommendations for the Community
1. **Ensure benchmarks do not over-represent cytotoxicity-related assays** to avoid misleading performance evaluations.
2. **Use baseline cell count models** as a reference point for assessing the value of more complex phenotypic profiles.
3. **Validate predictive models using independent test sets** with sufficient compound diversity.
4. **Analyze raw images** to confirm whether predicted bioactivity is driven by meaningful phenotypic changes or merely cytotoxicity.
5. **Incorporate concentration-response data** to improve signal detection in phenotypic assays.

## Data Availability
- Public datasets used in this study include:
  - Bray et al. (Cell Painting dataset)
  - Moshkov et al. (bioactivity benchmarks)
  - Hofmarcher et al. (target-based activity datasets)
  - Ha et al. (few-shot learning benchmark)
  - Sanchez-Fernandez et al. (contrastive learning)
- Curated protein-target dataset released in this repository.

## Citation
If you use this repository in your research, please cite:
```
Seal S., Dee W., Shah A., Zhang A., Titterton K., Cabrera Á. A., Boiko D., Beatson A., Puigvert J. C., Singh S., Spjuth O., Bender A., Carpenter A. E.
"Small Molecule Bioactivity Benchmarks are Often Well-Predicted by Counting Cells"
Upcoming Preprint, 2025.
```

## Acknowledgments
This research was supported by:
- The Cambridge Centre for Data-Driven Discovery (C2D3) Accelerate Programme
- National Institutes of Health (NIH MIRA R35 GM122547)
- Massachusetts Life Sciences Center Bits to Bytes Capital Call
- OASIS Consortium organized by HESI
- Swedish Research Council, FORMAS, Swedish Cancer Foundation, and Horizon Europe
- UKRI/BBSRC Collaborative Training Partnership in AI for Drug Discovery

For any questions, contact: **srijit@understanding.bio**


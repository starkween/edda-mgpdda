# Code for "Enhancing Drug-Disease Association Prediction Through Methodological Improvements to the MGP-DDA Framework"

Scripts that generate the ablation (5x5 repeated CV, N=25) and nested-CV results.

## Data
Download the MGP-DDA input files from the original authors (Kawichai et al., 2021) and put them in one folder
(not redistributed here): `drug-function_association_matrix.txt`, `disease-function_association_matrix.txt`,
`drug_list.csv`, `disease_list.csv`, `Supplementary_material_S1.csv` (known drug-disease pairs).

## Run
    pip install -r requirements.txt
    python run_repeated_cv.py --data-dir /path/to/data      # ablation, resumable, writes results/repeated_cv_results.json
    python summarize_stats.py                                # mean±std, paired t-test, sign-flip permutation test
    python run_nested_cv.py --data-dir /path/to/data --k 122 # nested CV

Configurations (name: SVD k, IC weighting, negatives):
baseline_k30 (30, no, random), ic_only_k30 (30, yes, random), svd_only_k107 (107, no, random),
rnss_only_k30 (30, no, RNSS), ic_svd_noRNSS (122, yes, random), full_method (122, yes, RNSS).
k=107 / k=122 are the smallest numbers of SVD components retaining >=95% of the variance of the unweighted / IC-weighted features.
Two additional configurations (ic_only, rnss_only with k=107) are kept in the results file for reference.
SVD is always fit on training data only. PU-bagging iterations: T=5 (`--T`). Seeds of the 5 repeats: 42,1,2,3,4.
About 40-100 s per fold on 2 CPUs. Results reproduce the reported values to within about +-0.0005 across library versions.

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

Configurations: baseline_k30 (k=30, no IC, random negatives), svd_only_k107, ic_only (k=107), rnss_only (k=107),
ic_svd_noRNSS (k=122), full_method (IC + k=122 + RNSS). SVD is always fit on training data only.
PU-bagging iterations: T=5 in these runs (`--T`). Seeds for the 5 repeats: 42,1,2,3,4. About 40 s per fold on 2 CPUs.

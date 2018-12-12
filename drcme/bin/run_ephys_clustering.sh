#!/usr/bin/env bash
set -e

CONFIG=$1

# Perform initial clustering
Rscript run_ephys_clustering.r $CONFIG

# Do post-clustering merges
python run_post_r_merging.py --input_json $CONFIG
Rscript run_entropy_cumul_merges.r $CONFIG

# Assess cluster stability
Rscript run_cluster_stability.r $CONFIG

# Merge unstable clusters
python run_merge_unstable_clusters.py --input_json $CONFIG

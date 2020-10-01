.. _run-ephys-clustering-sh:

run_ephys_clustering.sh
================================

This bash script runs several R and Python scripts in order to produce a set of stable
clusters from sparse PCA electrophysiology values. The script takes a JSON file with the
following parameters as an input:

===================================     ======================================================================================================================
Parameter                               Description
===================================     ======================================================================================================================
``components_file``                     Path to CSV file with sPCA components (input file)
``tau_file``                            Path to file with cluster membership probabilities (output of `run_ephys_clustering.r`)
``labels_file``                         Path to file with cluster labels (output of `run_ephys_clustering.r`)
``bic_file``                            Path to file with Bayes information criterion values for different numbers of clusters (output of `run_ephys_clustering.r`)
``merge_info_file``                     Path to JSON file with number of components after entropy-based merging (output of `drcme.bin.run_post_r_merging`)
``post_merge_tau_file``                 Path to file with cluster membership probabilities after entropy-based merging (output of `run_entropy_cumul_merges.r`)
``post_merge_labels_file``              Path to file with cluster labels after entropy-based merging (output of `run_entropy_cumul_merges.r`)
``jaccard_file``                        Path to file with Jaccard coefficient values after subset-based stability analysis (output of `run_cluster_stability.r`)
``etypes_file``                         Path to file with stable e-type assignments (output of `drcme.bin.run_merge_unstable_clusters`)
``post_merge_proba_file``               Path to file with post-merging cluster membership probabilities (output of `drcme.bin.run_merge_unstable_clusters`)
``merge_unstable_info_file``            Path to file with merge sequence information (output of `drcme.bin.run_merge_unstable_clusters`)
``entropy_piecewise_components``        Number of components (2 or 3) for piecewise linear fit of entropy scores
``outliers``                            List of outlier specimen IDs to exclude (use empty list if none)
===================================     ======================================================================================================================

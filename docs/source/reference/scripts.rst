Scripts
=======

These scripts are written to accomplish several common analysis tasks and illustrate
the use of library functions. They typically take a JSON file with a format defined by its
argschema_ parameters as their input.


Sparse PCA
----------

Reducing the dimensionality of standardized electrophysiology data sets using sparse
principal component analysis (sPCA).

.. autosummary::
	:toctree: generated/

	~drcme.bin.run_spca_fit
	~drcme.bin.run_existing_spca_on_new_data


Dual-modality clustering
------------------------

Clustering with both electrophysiology and morphology data at the same time.

.. autosummary::
	:toctree: generated/

	~drcme.bin.run_ephys_morph_clustering
	~drcme.bin.run_refine_unstable_coclusters


Electrophysiology-only clustering
---------------------------------

The electrophysiology clustering workflow to define e-types (as in
`Gouwens et al. (2019) <https://www.nature.com/articles/s41593-019-0417-0>`_) can be run
with the included :ref:`bash script <run-ephys-clustering-sh>`. This calls a set of R scripts
and Python scripts using a common input configuration file (described on the linked page).

The Python scripts included in this package are listed below:

.. autosummary::

    ~drcme.bin.run_post_r_merging
    ~drcme.bin.run_merge_unstable_clusters

.. toctree::
    :hidden:

    run_ephys_clustering.sh
    run_post_r_merging <drcme.bin.run_post_r_merging>
    run_merge_unstable_clusters <drcme.bin.run_merge_unstable_clusters>


Other utility scripts
-----------------------

Other scripts to perform various one-off tasks.

.. autosummary::
    :toctree: generated/

    ~drcme.bin.run_combined_tsne
    ~drcme.bin.run_joint_ephys_morph_tsne
    ~drcme.bin.run_rf_prediction


.. _argschema: http://argschema.readthedocs.io/


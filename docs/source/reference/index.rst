=============
API Reference
=============

This is the API reference of DRCME. If you are interested in running standard analysis
tasks, please look at the :doc:`scripts reference page <scripts>` for more information.

.. currentmodule:: drcme

.. toctree::
	:hidden:

	scripts


:mod:`drcme.ephys_morph_clustering`: Combined electrophysiology and morphology clustering
=========================================================================================

.. automodule:: drcme.ephys_morph_clustering
    :no-members:
    :no-inherited-members:

Functions
---------
.. currentmodule:: drcme

.. autosummary::
    :toctree: generated/

    ephys_morph_clustering.clustjaccard
    ephys_morph_clustering.coclust_rates
    ephys_morph_clustering.consensus_clusters
    ephys_morph_clustering.gmm_combo_cluster_calls
    ephys_morph_clustering.hc_nn_cluster_calls
    ephys_morph_clustering.hc_combo_cluster_calls
    ephys_morph_clustering.refine_assignments
    ephys_morph_clustering.spectral_combo_cluster_calls
    ephys_morph_clustering.subsample_run


:mod:`drcme.load_data`: Data handling of IPFX outputs
=====================================================

.. automodule:: drcme.load_data
    :no-members:
    :no-inherited-members:

Functions
---------
.. currentmodule:: drcme

.. autosummary::
    :toctree: generated/

    load_data.define_spca_parameters
    load_data.load_h5_data


:mod:`drcme.post_gmm_merging`: Merging Gaussian mixture model components
========================================================================

.. automodule:: drcme.post_gmm_merging
    :no-members:
    :no-inherited-members:

Functions
---------
.. currentmodule:: drcme

.. autosummary::
    :toctree: generated/

    post_gmm_merging.entropy_combi
    post_gmm_merging.entropy_specific_merges
    post_gmm_merging.fit_piecewise
    post_gmm_merging.order_new_labels


:mod:`drcme.prediction`: Type label prediction
==============================================

.. automodule:: drcme.prediction
    :no-members:
    :no-inherited-members:

Functions
---------
.. currentmodule:: drcme

.. autosummary::
    :toctree: generated/

    prediction.rf_predict


:mod:`drcme.spca`: Sparse principal component analysis
==========================================================

.. automodule:: drcme.spca
    :no-members:
    :no-inherited-members:

Functions
---------
.. currentmodule:: drcme

.. autosummary::
    :toctree: generated/

    spca.consolidate_spca
    spca.orig_mean_and_std_for_zscore
    spca.select_data_subset
    spca.spca_on_all_data
    spca.spca_transform_new_data
    spca.spca_zht


:mod:`drcme.tsne`: t-SNE helper functions
==========================================================

.. automodule:: drcme.tsne
    :no-members:
    :no-inherited-members:

Functions
---------
.. currentmodule:: drcme

.. autosummary::
    :toctree: generated/

    tsne.combined_tsne
    tsne.dual_modal_tsne
Getting Started
===============

The ``drcme`` package has several functions and scripts for the analysis and
clustering of electrophysiological and morphological data. For certain
electrophysiological analyses, it is designed to work with data processed by
the `IPFX package <https://github.com/alleninstitute/ipfx>`_.

There are several :doc:`scripts <source/reference/scripts>` provided for common
tasks that expect JSON files as inputs. These could be used as-is or customized
for your own analysis needs.

Sparse principal component analysis
-----------------------------------

This package has several functions and scripts for performing a variant of
principal component analysis that incorporates a sparseness penalty. There is a
provided script (:mod:`drcme.bin.run_spca_fit`) that is designed to work with
"feature vector" files processed by another software package, `IPFX
<https://ipfx.readthedocs.io>`_. In particular, the IPFX script
:mod:`run_feature_vector_extraction script
<ipfx:ipfx.bin.run_feature_vector_extraction>` produces an HDF5 file as its
output, which serves as the main input to :mod:`drcme.bin.run_spca_fit`.

The scripts in this package frequently use the `argschema
<http://argschema.readthedocs.io/>`_ package to define their inputs. This
allows inputs to be specified either as command line arguments or bundled into
a JSON input file. For this guide, we will use the latter method (note that you
can actually use a combination of the two methods as well).

We will assume that we processed electrophysiology data with
IPFX and have an HDF5 file already created (``my_feature_vectors.h5``).
We can look at the documentation for :mod:`drcme.bin.run_spca_fit` and see
that it can take several parameters. There are parameters for performing the
sPCA and where to save outputs, and there are other (nested) parameters for
the datasets used by the analysis. Here, we are only using a single HDF5
file as input, so we only need to use set of the latter parameters, but you
can use multiple files as sources if needed.

Many of the dataset-related parameters are options
for filtering the data from the HDF5 file so that you can perform sPCA on
a subset of cells. However, in this example, we will use all the cells, so
we not need to worry about them here. The file ends up looking like:

.. code-block:: JSON
    :caption: my_spca_input.json

    {
        "params_file": "/path/to/specific/spca_params.json",
        "output_dir": "example_output_dir",
        "output_code": "EXAMPLE",
        "datasets": [
            {
                "fv_h5_file": "/path/to/my_feature_vectors.h5"
            }
        ]
    }

The parameter ``params_file`` is the path to another JSON file that specifically
configures the sparse principal component analysis. It is often shared between
analyses of different data sets (it more relates to the form of the HDF5
file and what types of protocols you want to analyze), so it is separated out.

The expected format of that file is described in the function that loads it,
:func:`drcme.load_data.define_spca_parameters`. Here, we will use a simplified
version that just analyzes two types of data in the example HDF5 file - the
AP waveform (``first_ap_v``) and the shape of the interspike interval
(``isi_shape``). That configuration looks like this:

.. code-block:: JSON
    :caption: spca_params.json

    {
        "first_ap_v": {
            "n_components": 7,
            "nonzero_component_list": [267, 233, 267, 233, 233, 250, 233],
            "use_corr": false,
            "range": [0, 300]
        },
        "isi_shape": {
            "n_components": 4,
            "nonzero_component_list": [100, 90, 80, 80],
            "use_corr": false,
            "range": null
        }
    }

The keys in this file must match those found in the HDF5 file, so the ones used
here are a subset of those produced by the IPFX script. You can copy and modify
that script if you want to analyze other types of electrophysiological data.
For each type of data, we specify the number of components to keep for the
sparse PCA (``n_components``) as well as the number of non-zero loadings for
that component (``nonzero_component_list``). The length of
``nonzero_component_list`` should equal ``n_components``. These values will
depend on the data set and trade off things like the amount of explained
variance for sparseness; you may need to do some trial-and-error for your own
data. The option ``use_corr`` is a boolean value that specifies whether all the
columns of the data matrix should be scaled by their standard deviation -- in
this case we set it to ``false`` because all the values within each data set
are on the same scale (since they are time series of membrane voltage).
Finally, the ``range`` parameter indicates what section of the data set should
be used. For ``isi_shape``, we will use the entire thing, so we can leave
``range`` with a value of ``null``. For ``first_ap_v``, the traces produced by
IPFX are the concatenation of three 150-point-long traces of the first action
potentials evoked by a short-square current pulse, a long-square current pulse,
and a ramp current. However, the ramp current sometimes fails to elicit an
action potential (for biological reasons), so we may not want to use it here
(because some cells will have all zeroes for that value, which is a very strong
signal that could distort the analysis). Therefore, we only want the first 300
points of the data (from 0 up to but not including 300) to leave the ramp AP
waveform out. If we wanted just the first and last AP waveforms, excluding the
middle, we would give the ``range`` parameter a value of ``[0, 150, 300,
450]``.

Now that the inputs are specified, we can run the script simply as follows::

    $ python -m drcme.bin.run_spca_fit --input_json my_spca_input.json

Note that you can see messages about progress by setting the ``log_level``
parameter to ``INFO``; otherwise those messages are suppressed by default. At
the end, we will have files in ``example_output_dir`` called
``sparse_pca_components_EXAMPLE.csv`` (containing the transformed, z-scored
sPCA values), ``spca_components_use_EXAMPLE.json`` (which indicates which
components were kept), and ``spca_loadings_EXAMPLE.pkl`` which contains the
loadings as well as explained variance information. The file
``sparse_pca_components_EXAMPLE.csv`` is in a format used by other clustering
procedures in the package.


Electrophysiology and morphology clustering
-------------------------------------------

The script :mod:`drcme.bin.run_ephys_morph_clustering` performs a joint
clustering of electrophysiology and morphology data. It takes the strategy of
performing multiple variations of clustering algorithms (with several parameter
sets) and then finding consensus clusterings using all those results together.
It also performs a cluster stability analysis via subsampling.

The main things we need to specific to use the script are the electrophysiology
and morphology data files. We also need to give the script the paths for the
various output files, including the cluster labels (``cluster_labels_file``),
the specimen IDs of the cells analyzed (``specimen_id_file``, since only cells
that are found in both data files are used), the cell-wise co-clustering matrix
(``cocluster_matrix_file``), and an ordering of cells that puts cells in the
same cluster together (``ordering_file``). The cells in the
``cluster_labels_file`` and ``cocluster_matrix_file`` are in the same order as
the ``specimen_id_file``. The ``jacccards_file`` contains the Jaccard
coefficients for each cluster, which are an indication of their stability.

With that in mind, we can structure our input JSON file as follows:

.. code-block:: JSON
    :caption: my_me_clustering_input.json

    {
        "ephys_file": "/path/to/ephys_data.csv",
        "morph_file": "/path/to/morph_data.csv",
        "specimen_id_file": "/path/to/output/specimen_ids.txt",
        "cluster_labels_file": "/path/to/output/cluster_labels.csv",
        "cocluster_matrix_file": "/path/to/output/coclustering_matrix.txt",
        "ordering_file": "/path/to/output/ordering.txt",
        "jaccards_file": "/path/to/output/jaccards.txt",
    }

And then we can run the script like::

    $ python -m drcme.bin.run_ephys_morph_clustering --input_json my_me_clustering_input.json

If you do find that you have unstable clusters that perhaps should be folded
into other stable clusters, there is an additional script
:mod:`drcme.bin.run_refine_unstable_coclusters` that can be used for that.






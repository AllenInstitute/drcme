import pytest
import pandas as pd
import drcme.spca_zht as szht
from sklearn.utils.testing import assert_array_almost_equal


def test_pitprops_example(shared_datadir):
    # Expected loadings are output from original implementation
    data_filename = (shared_datadir / 'pitprops.csv')
    loadings_filename = (shared_datadir / 'pitprops_spca_loadings.csv')

    pitprops = pd.read_csv(data_filename, index_col=0)
    expected_loadings = pd.read_csv(loadings_filename, index_col=0)

    result = szht.spca_zht(pitprops.values, K=6, para=[7,4,4,1,1,1],
        type="Gram", sparse="varnum")
    assert_array_almost_equal(result["loadings"], expected_loadings.values)

"""
The :mod:`drcme.tsne` module contains wrapper functions for
applying t-SNE to the electrophysiology and morphology data.
"""
import numpy as np
import pandas as pd
from sklearn import manifold
import logging


def combined_tsne(df_1, df_2, perplexity=25, n_iter=20000):
    """Perform t-SNE on two data sets

    Parameters
    ----------
    df_1 : DataFrame
        First data set
    df_2 : DataFrame
        Second data set
    perplexity : float
        t-SNE perplexity parameter
    n_iter : int
        Maximum number of iterations for t-SNE optimization

    Returns
    -------
    DataFrame
        Contains "x" and "y" coordinates for 2D t-SNE embedding of
        combined data set
    """

    all_together = np.vstack([df_1.values, df_2.values])
    all_ids = df_1.index.tolist() + df_2.index.tolist()

    tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0,
                        verbose=2, n_iter=n_iter, perplexity=perplexity)
    Y = tsne.fit_transform(all_together)

    return pd.DataFrame({"x": Y[:, 0], "y": Y[:, 1]}, index=all_ids)


def dual_modal_tsne(ephys_df, morph_df, relative_ephys_weight=1.,
                    perplexity=25, n_iter=20000):
    """Perform t-SNE on electrophysiology and morphology data sets

    Parameters
    ----------
    ephys_df : DataFrame
        Electrophysiology data set
    morph_df : DataFrame
        Morphology data set
    relative_ephys_weight : float, optional
        Relative weighting of electrophysiology data
    perplexity : float
        t-SNE perplexity parameter
    n_iter : int
        Maximum number of iterations for t-SNE optimization

    Returns
    -------
    DataFrame
        Contains "x" and "y" coordinates for 2D t-SNE embedding for samples
        that exist in both `ephys_df` and `morph_df`
    """
    morph_ids = morph_df.index.values

    # Get ephys data for cells with morphologies
    ids_with_morph_for_ephys = [s for s in morph_ids
                                if s in ephys_df.index.tolist()]
    ephys_df_joint = ephys_df.loc[ids_with_morph_for_ephys, :]

    # Only use morphs that have ephys
    mask = [s in ephys_df_joint.index.tolist() for s in morph_ids]
    morph_df_joint = morph_df.loc[mask, :]

    logging.debug("ephys joint shape ", ephys_df_joint.shape)
    logging.debug("morph joint shape ", morph_df_joint.shape)

    elmo_data = np.hstack([morph_df_joint.values,
                           relative_ephys_weight * ephys_df_joint.values])

    tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0,
                        verbose=2, n_iter=n_iter, perplexity=perplexity)
    Y = tsne.fit_transform(elmo_data)
    return pd.DataFrame({"x": Y[:, 0], "y": Y[:, 1]}, index=ephys_df_joint.index.values)

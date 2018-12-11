import numpy as np
import pandas as pd
from sklearn import manifold
import logging


def combined_tsne(df_1, df_2, n_components=2, perplexity=25, n_iter=20000):
    all_together = np.vstack([df_1.values, df_2.values])
    all_ids = df_1.index.tolist() + df_2.index.tolist()

    tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0,
                        verbose=2, n_iter=n_iter, perplexity=perplexity)
    Y = tsne.fit_transform(all_together)

    return pd.DataFrame({"x": Y[:, 0], "y": Y[:, 1]}, index=all_ids)


def dual_modal_tsne(ephys_df, morph_df, relative_ephys_weight=1.,
                    n_components=2, perplexity=25, n_iter=20000):
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

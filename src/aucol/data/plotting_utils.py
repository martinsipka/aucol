import pandas as pd
import torch
import numpy as np
import ase.io

"""
Encode the collective variables and save the resulting Pandas dataframe.

Args:
    model (callable): Model to calculate encoded CVs
    n_cv (int): Number of CVs we are encoding
    repres (int): Size of a batch
    ase_name (str): Name of the dataset file we are calculating CVs for
    columns (list of str): Names of colums in dataframe for the pairs
    pairs (list of tuples): Pairs of indices to calculate distances for. Used for
        comparison with our CVs.
    batch_size (int): Batch size for the CV generation. We can use more batches at once.
Returns:
    df (DataFrame): Pandas DataFrame with data to plot the results.
"""
def encode_cv(model,n_cv, repres, ase_name, columns, pairs, batch_size):
    """Compute the CV over a dataloader"""
    dataset = torch.utils.data.TensorDataset(repres)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
            pin_memory=True, shuffle=False)

    cvs = []

    model.eval()

    with torch.no_grad():

        for i, rep in enumerate(loader):
            cv, var = model.encode(rep[0].to(model.device))

            cvs.append(cv.detach().cpu().numpy())

        cvs = np.concatenate(cvs)

    traj = ase.io.read(ase_name, index=':')

    if n_cv > 1:
        cv_columns = ["vae_cvs_" + str(i) for i in range(n_cv)]
    else:
        cv_columns = ["vae_cvs"]

    df = pd.DataFrame(np.zeros((len(cvs), len(columns + cv_columns))), columns=columns + cv_columns)
    for i in range(n_cv):
        df[cv_columns[i]]=cvs[:,i]

    for i, atoms in enumerate(traj):
        for j, pair in enumerate(pairs):
            df.iloc[i,j] = atoms.get_distance(pair[0], pair[1], mic=True)

    return df

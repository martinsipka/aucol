import torch

import schnetpack as spk
from schnetpack import AtomsData, ConcatAtomsData
from schnetpack.environment import AseEnvironmentProvider

import os
from os.path import exists


"""
Preload dataset and return representations.

Args:
    dataset (tensor):
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
def process_dataset(dataset, model, l_b_size, device):
    print("Starting calculation with device: ", device, " and batch size: ", l_b_size, flush=True)
    rep = []
    pos = []
    for i, data in enumerate(dataset):
        if i%10 == 0:
            print(i*l_b_size, flush=True)
        data = {k: v.to(device) for k, v in data.items()}
        repre = model(data)

        rep.append(repre.detach().cpu())
        pos.append(data['_positions'].detach().cpu())
    return torch.cat(rep), torch.cat(pos)

class PrecachedRTPDataloader:
    """
    Class containing precached representations and positions.

    Args:
        dl_r (DataLoader): DataLoader to calculate reactant representations from
        dl_sg (DataLoader): DataLoader to calculate steered dynamics representations from
        dl_p (DataLoader): DataLoader to calculate product representations from
        model (callable): Representations model
        l_b_size (int): Batch size to use when calculating representations  (Default=100)
        device (str): Device name (Default='cpu')
    """

    def __init__(self, dl_r, dl_sg, dl_p, model, l_b_size=100, device='cpu'):

        self.repre_tensor_r, self.pos_tensor_r = process_dataset(dl_r, model, l_b_size, device)
        self.repre_tensor_sg, self.pos_tensor_sg = process_dataset(dl_sg, model, l_b_size, device)
        self.repre_tensor_p, self.pos_tensor_p = process_dataset(dl_p, model, l_b_size, device)

    def __len__(self):
        return len(self.cached_sg)

class PrecachedRPDataloader:
    """
    Class containing precached representations and positions.

    Args:
        dl_r (DataLoader): DataLoader to calculate reactant representations from
        dl_p (DataLoader): DataLoader to calculate product representations from
        model (callable): Representations model
        l_b_size (int): Batch size to use when calculating representations  (Default=100)
        device (str): Device name (Default='cpu')
    """

    def __init__(self, dl_r, dl_p, model, l_b_size=100, device='cpu'):

        self.repre_tensor_r, self.pos_tensor_r = process_dataset(dl_r, model, l_b_size, device)
        self.repre_tensor_p, self.pos_tensor_p = process_dataset(dl_p, model, l_b_size, device)

    def __len__(self):
        return len(self.cached_sg)

class PrecachedDataloader:
    """
    Class containing precached representations and positions.

    Args:
        dl (DataLoader): DataLoader to calculate representations from
        model (callable): Representations model
        l_b_size (int): Batch size to use when calculating representations  (Default=100)
        device (str): Device name (Default='cpu')
    """

    def __init__(self, dl, model, l_b_size=100, device='cpu'):
        self.repre_tensor, self.pos_tensor = process_dataset(dl, model, l_b_size, device)

    def __len__(self):
        return len(self.cached_sg)

"""
Preload dataset and return representations. If the representations are precomputed
they are loaded from disc.

Args:
    react_path (list of str): data path
    precached_repre_path (list of str): Path where the representations are stored
    repre_directory (str): Directory where representations are stored
    names (list of str): Names of the representations
    device (str): Device name
    representation_model (callable): Representation model to use when calculating
        representations
    cutoff (int): Model cutoff
Returns:
    precached_data (tensor) Concatenated representation tensors
"""
def load_representation(traj_path, repre_path, repre_directory, device='cpu', representation_model=None, cutoff=6.0, l_b_size=10):

    precached_data = {}

    if exists(repre_path):
        precached_rep = torch.load(repre_path)


        print("Representations file was loaded from precached memmory! If this was not intended, delete or move precached files.", flush=True)
    else:
        if not os.path.exists(repre_directory):
            os.makedirs(repre_directory)

        print("Pre-calculated representations were not found.", flush=True)
        dataset = AtomsData(traj_path, environment_provider=AseEnvironmentProvider(cutoff))

        loader = spk.AtomsLoader(dataset, l_b_size, shuffle=False)

        precached_rep = PrecachedDataloader(loader, representation_model, l_b_size, device)

        torch.save(precached_rep, repre_path)
        print("Representations were calculated and a file was saved.", flush=True)

    return precached_rep




"""
Preload dataset and return representations. If the representations are precomputed
they are loaded from disc.

Args:
    react_path (list of str): data path
    precached_repre_path (list of str): Path where the representations are stored
    repre_directory (str): Directory where representations are stored
    names (list of str): Names of the representations
    device (str): Device name
    representation_model (callable): Representation model to use when calculating
        representations
    cutoff (int): Model cutoff
Returns:
    precached_data (tensor) Concatenated representation tensors
"""
def load_representations_list(traj_paths, repre_paths, repre_directory, names, device='cpu', representation_model=None, cutoff=6.0, l_b_size=10):

    precached_data = {}

    for t_path, r_path, name in zip(traj_paths, repre_paths, names):
        if exists(r_path):
            precached_rep = torch.load(r_path)


            print("Representations file was loaded from precached memmory! If this was not intended, delete or move precached files.", flush=True)
        else:
            if not os.path.exists(repre_directory):
                os.makedirs(repre_directory)

            print("Pre-calculated representations were not found.", flush=True)
            dataset = AtomsData(t_path, environment_provider=AseEnvironmentProvider(cutoff))

            loader = spk.AtomsLoader(dataset, l_b_size, shuffle=False)

            precached_rep = PrecachedDataloader(loader, representation_model, l_b_size, device)

            torch.save(precached_rep, r_path)
            print("Representations were calculated and a file was saved.", flush=True)

        precached_data[name] = precached_rep
        #print(precached_data)

    return precached_data


"""
Preload dataset and return representations. If the representations are precomputed
they are loaded from disc.

Args:
    react_path (list of str or str): Reactants path
    prods_path (list of str or str): Products path
    sg_path (list of str or str)): Steered dynamics path
    repre_directory (str): Directory where representations are stored
    precached_repre_path (str): Path where the representations are stored
    device (str): Device name
    representation_model (callable): Representation model to use when calculating
        representations
    cutoff (int): Model cutoff
Returns:
    precached_data (tensor) Concatenated representation tensors
"""
def load_representations(react_paths, prods_paths, sg_paths, repre_directory,  precached_repre_path, device='cpu', representation_model=None, cutoff=6.0, l_b_size=10):
    if isinstance(react_paths, list):
        react_dataset = ConcatAtomsData(datasets=[AtomsData(rp, environment_provider=AseEnvironmentProvider(cutoff)) for rp in react_paths])
    else:
        react_dataset = AtomsData(react_paths, environment_provider=AseEnvironmentProvider(cutoff))

    if isinstance(prods_paths, list):
        prods_dataset = ConcatAtomsData(datasets=[AtomsData(pp, environment_provider=AseEnvironmentProvider(cutoff)) for pp in prods_paths])
    else:
        prods_dataset = AtomsData(prods_paths, environment_provider=AseEnvironmentProvider(cutoff))

    if isinstance(sg_paths, list):
        sg_dataset = ConcatAtomsData(datasets=[AtomsData(sgp, environment_provider=AseEnvironmentProvider(cutoff)) for sgp in sg_paths])
    else:
        sg_dataset = AtomsData(sg_paths, environment_provider=AseEnvironmentProvider(cutoff))


    #If the precalculated representation exists, we can use it.
    if exists(precached_repre_path):
        precached_data = torch.load(precached_repre_path)
        print("Representations file was loaded from precached memmory! If this was not intended, delete or move precached files.", flush=True)

    else:
        if not os.path.exists(repre_directory):
            os.makedirs(repre_directory)

        print("Pre-calculated representations were not found.", flush=True)
        rl = spk.AtomsLoader(react_dataset, l_b_size, shuffle=False)
        sgl = spk.AtomsLoader(sg_dataset, l_b_size, shuffle=False)
        pl = spk.AtomsLoader(prods_dataset, l_b_size, shuffle=False)
        precached_data = PrecachedRTPDataloader(rl, sgl, pl, representation_model, l_b_size, device)

        torch.save(precached_data, precached_repre_path)
        print("Representations were calculated and a file was saved.", flush=True)

    return precached_data

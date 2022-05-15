import torch
import matplotlib.pyplot as plt

"""
Get the most variance from represenation. Helps to reduce the size by only considering the elements that are changing.

Args:
    react_tensor (tensor): tensor of reactants representations
    product_tensor (tensor): tensor of products representations
    select_k (int): Number of elements to pick
    verbose (bool): Output some information about the indices
    plot (bool): Plot the resulting representations coeficients
Returns:
    indices with the most information
"""
def get_most_variance(react_tensor, product_tensor, select_k = 300, verbose=False, plot=False):

    stds_r, means_r = torch.std_mean(react_tensor, dim=0)
    stds_p, means_p = torch.std_mean(product_tensor, dim=0)
    #stds_tot, means_tot = torch.std_mean(torch.cat([train_cached_loader.repre_tensor_r, train_cached_loader.repre_tensor_p], dim=0), dim=0)

    weight_coefs = torch.abs((means_r-means_p))/(stds_r + stds_p)
    weight_coefs = torch.nan_to_num(weight_coefs)
    weight_coefs = weight_coefs / torch.max(weight_coefs)

    coefs, indices = torch.topk(weight_coefs.ravel(), k=select_k)

    if verbose:
        print(indices/128)
        print(coefs)

    if plot:
        fig, axs = plt.subplots(2, 1, figsize=(15,10))

        axs[0].scatter(np.arange(weight_coefs.ravel().shape[0])/128,weight_coefs.ravel().cpu())
        axs[0].scatter(np.arange(weight_coefs.ravel().shape[0])/128,weight_coefs.ravel().cpu())

        plt.show()

    return indices

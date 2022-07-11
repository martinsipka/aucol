import schnetpack as spk
import torch
import numpy as np
from torch import nn
from schnetpack.data.atoms import AtomsConverter

from aucol.utils.prefilter_indices import get_most_variance
from aucol.nn.models.cv_module import CVModule
from aucol.nn.layers.basic import IndexSelectLayer


"""
# Simple KL divergence between VAE output and normal distribution N(0,1) possibly multidimensional

Args:
    a_mean (tensor): Means
    a_std (tensor): Standard deviations

Returns:
    KL divergence between distributions
"""
def simple_kl_divergence(a_mean, a_sd):
    """Method for computing KL divergence of two normal distributions."""
    a_sd_squared = a_sd ** 2

    b_mean = 0
    b_sd_squared = 1

    ratio = a_sd_squared / b_sd_squared + 1E-9

    loss = torch.mean((a_mean - b_mean) ** 2 / (2 * b_sd_squared)
            + (ratio - torch.log(ratio) - 1) / 2)
    return loss


#Initialize VAE
class SchnetBasedVAE(torch.nn.Module):
    """
    Predicts Collective variable and its derivative for training runs. The class
    is based on Schnetpack package and uses its layers and base representation.

    Args:
        representation_model (callable): Network used for Energy prediction. we used
            it to obtain representations.
        data (tuple of tensors): name of the output property (default: "cv")
        device (str): Name of the device to train on
        args (args): Args for the training.
        cutoff (float): Cutoff to use

    Returns:
        Returns CVs for the given atom inputs
    """
    def __init__(self, representation_model, data, device, args, hidden_l = [8,32,64,128,256,512], cutoff=6.0, normalize_input=True, normalize_output=False):
        super(SchnetBasedVAE, self).__init__()

        self.normalize_input = normalize_input
        self.normalize_output = normalize_output

        # Data input statistics
        train_repre_tensor = torch.cat(data)
        repre_std, repre_mean = torch.std_mean(train_repre_tensor, dim=(0,2))

        self.n_rp = train_repre_tensor.shape[2]
        self.n_atoms = train_repre_tensor.shape[1]

        # Create real model with proper statistics
        #if isinstance(representation_model, spk.representation.SymmetryFunctions):
        #    self.n_rp = representation_model.n_symfuncs
        #else:
        #    self.n_rp = representation_model.interactions[-1].dense.out_features

        rescaled_std = repre_std.unsqueeze(1).repeat(1,self.n_rp)
        rescaled_mean = repre_mean.unsqueeze(1).repeat(1,self.n_rp)

        # Extract the representation from the NNP for cases when it is needed
        self.repre = representation_model
        for param in self.repre.parameters(): param.requires_grad = False
        self.repre.eval()
        self.normalize_repre = spk.nn.base.Standardize(rescaled_mean, rescaled_std)

        # Get the most important indices
        indices = get_most_variance(data[0], data[1],select_k=min(args.n_indices, self.n_rp*self.n_atoms))

        # The number of chosen indices
        self.k_best = len(indices)
        self.index_layer = IndexSelectLayer(indices)

        # Intermediate layers
        if args.shallow_encoder:
            self.deep_encoder = nn.Sequential(torch.nn.Identity())
            mean_in = int(self.k_best)
        else:
            self.deep_encoder = nn.Sequential(
              spk.nn.Dense(in_features=self.k_best, out_features=self.k_best, activation=spk.nn.shifted_softplus),
              spk.nn.Dense(in_features=int(self.k_best), out_features=int(self.k_best/2), activation=spk.nn.shifted_softplus),
              )
            mean_in = int(self.k_best/2)


        # Output the means and variance

        #self.means = spk.nn.Dense(in_features=int(self.k_best), out_features=n_CVs)
        #self.logvars = spk.nn.Dense(in_features=int(self.k_best), out_features=n_CVs)
        self.means = spk.nn.Dense(in_features=mean_in, out_features=args.n_CVs)
        self.logvars = spk.nn.Dense(in_features=mean_in, out_features=args.n_CVs)
        self.aggregate = spk.nn.Aggregate(1)

        # The hidden layer sizes

        self.reconstruct = spk.nn.blocks.MLP(n_in=args.n_CVs, n_out=self.k_best, n_hidden=hidden_l, n_layers=len(hidden_l)+1,  activation=torch.nn.functional.relu)

        # The reconstruction loss
        self.recon_loss = nn.MSELoss()

        self.init_epoch = 0
        self.device = device
        self.n_CVs = args.n_CVs
        self.recon_w = 1

        # Atom converter setup
        self.atoms_converter = AtomsConverter(
            environment_provider=spk.environment.AseEnvironmentProvider(cutoff),
            collect_triples=False,
            device=device)

        # Training scheduler and optimizer
        self.opt = torch.optim.Adam(self.parameters(), lr=args.learning_rate, weight_decay=args.l2_reg)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=1, gamma=args.gamma)
        self.print_loss = 1
        self.standardize_cv = spk.nn.base.Standardize(torch.tensor([0.]), torch.tensor([1.]))

        self.to(device)

    """
    Predicts Collective variable and its derivative for training runs. The class
    is based on Schnetpack package and uses its layers and base representation.

    Args:
        nnp (callable): Network used for Energy prediction. we used
            it to obtain representations.

    Returns:
        NNP with CV output
    """
    def create_cv_nnp(self, nnp):
        if self.normalize_input:
            out_net = nn.Sequential(
                spk.nn.base.GetItem("representation"),
                self.normalize_repre,
                nn.Flatten(),
                self.index_layer,
                self.deep_encoder,
                self.means,
                self.standardize_cv
            )
        else:
            out_net = nn.Sequential(
                spk.nn.base.GetItem("representation"),
                nn.Flatten(),
                self.index_layer,
                self.deep_encoder,
                self.means,
                self.standardize_cv
            )
        cv_module = CVModule(out_net)
        nnp.output_modules.append(cv_module)
        return nnp

    """
    Calculate representation from inputs and save them to the input dictionary

    Args:
        inputs (dictionary): model inputs

    Returns:

    """
    def calcRepre(self, inputs):
        inputs["representation"] = self.repre(inputs)


    """
    Calculate means and logvars from the encoder

    Args:
        repre (dictionary): representation tensor

    Returns:
        means (tensor): means tensors
        logvars (tensor): logvars tensor
    """
    def encode(self, repre):
        x = repre
        if self.normalize_input:
            x = self.normalize_repre(x)
        x = x.reshape(-1, x.shape[-2]*x.shape[-1])
        x = self.index_layer(x)
        x = self.deep_encoder(x)
        means = self.means(x)
        logvars = self.logvars(x)
        return means, logvars

    """
    Decode a CV to form a representation

    Args:
        cvs (tensor): Collective variable to decode

    Returns:
        recon_repre (tensor): Reconstructed representation
    """
    def decode(self, cvs):
        recon_repre = self.reconstruct(cvs)
        return recon_repre


    """
    Find a CV from inputs

    Args:
        inputs (dictionary): Model inputs

    Returns:
        cvs (tensor): CVs tensor
    """
    def forward(self, inputs):
        self.calcRepre(inputs)
        means, _ = self.encode(inputs["representation"])
        cvs = self.standardize_cv(means)
        return cvs

    """
    Find a CV and return its gradient

    Args:
        atoms (ASE atoms): ASE Atoms object

    Returns:
        cvs (tensor): Collective variable
        cv_grad (list of tensors) Gradients of every CV with respect to coordinates
    """
    def cvs_with_grad(self, atoms):

        model_inputs = self.atoms_converter(atoms)

        model_inputs['_positions'].requires_grad=True
        # Call model
        cvs_ar, = self.forward(model_inputs)
        cvs_grad = []
        for cv in cvs_ar:
            cv_grad, = torch.autograd.grad(cv,model_inputs['_positions'], create_graph=True)
            cvs_grad.append(cv_grad.cpu().detach().numpy())

        return cvs_ar.cpu().detach().numpy(), cvs_grad

    """
    Find a CV for single input

    Args:
        atoms (ASE atoms): ASE Atoms object

    Returns:
        cvs (tensor): CVs tensor
    """
    def single_point_value(self, atoms):
        return self.forward(self.atoms_converter(atoms))


    """
    Construct standardization layer

    Args:
        all_points (tensor): Points to get the statistics from

    Returns:

    """
    def set_cv_standard(self, all_points):
        cv_maxima = np.max(all_points, 0)
        cv_minima = np.min(all_points, 0)

        scale_margin = 1.3
        scale = (cv_maxima - cv_minima) * 0.5 * scale_margin
        shift = (cv_maxima + cv_minima)/2
        scale = torch.tensor(scale)
        shift = torch.tensor(shift)
        self.standardize_cv = spk.nn.base.Standardize(shift, scale)

    """
    Process batch of training data

    Args:
        rep (tensor): Representation tensor

    Returns:
        latent_loss (tensor) latent loss tensor
        means (tensor) means of the cvs
        decoded (tensor) decoded representations
    """
    def process_batch(self, rep):
        means, logvars = self.encode(rep)

        noise = torch.clone(means).normal_(0, 1)
        z = noise * torch.exp(logvars / 2) + means
        # =================latent loss=======================
        latent_loss = self.n_CVs*simple_kl_divergence(means, torch.exp(logvars / 2))
        # =================decode=====================
        decoded = self.decode(z)

        return latent_loss, means, decoded


    """
    Reconstruction loss with index picking

    Args:
        decoded (tensor): decoded representations
        gold (tensor): gold representations
    Returns:
        recon_loss (tensor) reconstruction loss
    """
    def w_recon(self, decoded, gold):
        if self.normalize_output:
            gold = self.normalize_repre(gold)
        gold = gold.reshape(-1, gold.shape[-2]*gold.shape[-1])
        gold = self.index_layer(gold)
        return self.recon_loss(decoded, gold)


    """
    Train a batch

    Args:
        epochs (tensor): number of epochs to train
        reps (tensor): representations input
    Returns:
        recon_loss (tensor) reconstruction loss
    """
    def train_batch(self, epoch, reps):
        input_r  = reps[0].to(self.device)

        latent_loss, means, decoded_r = self.process_batch(input_r)

        r_s = input_r.shape

        recon_loss = r_s[1] * r_s[2] * self.w_recon(decoded_r, input_r)
        tot_loss = latent_loss + self.recon_w*recon_loss

        self.opt.zero_grad()
        tot_loss.backward()
        self.opt.step()

        return latent_loss, recon_loss, tot_loss


    """
    Evaluate batch of representations

    Args:
        reps (tensor): batch of representations
    Returns:
        latent_loss (tensor) latent loss tensor
        means (tensor) means of the cvs
        decoded (tensor) decoded representations
    """
    def eval_batch(self, reps):

        input_r  = reps[0].to(self.device)
        input_r.requires_grad = True
        latent_loss, means, decoded_r = self.process_batch(input_r)

        r_s = input_r.shape
        recon_loss = r_s[1] * r_s[2] * self.w_recon(decoded_r, input_r)

        tot_loss = latent_loss + self.recon_w*recon_loss
        return latent_loss, recon_loss, tot_loss


    """
    Train the variational autoencoder on the dataset

    Args:
        dataset (tensor): tensor dataset of representations
        num_epochs (int): number of epizodes of training
        batch_size (int): Size of a batch
    Returns:
    """
    def train_vae(self, dataset, num_epochs = 20, batch_size=20):
        ##@title Training
        print(len(dataset))
        train_set_size = int(len(dataset) * 0.8)
        valid_set_size = len(dataset) - train_set_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_set_size, valid_set_size])


        t_rep_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        v_rep_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        #format output
        float_formatter = lambda x: "%.6f" % x
        np.set_printoptions(formatter={'float_kind':float_formatter})

        print('[{:>3}/{:>3}] {:>10} {:>5}, {:>5}, {:>5} {:>5}'.format('ep','tot','tot_loss', 'latent_loss', 'recon_loss', 'grad_loss', 'val_loss', 'lr'), flush=True)

        # -- Training --
        for epoch in range(num_epochs):
            running_latent, running_recon, running_total, running_val = 0.0, 0.0, 0.0, 0.0
            for reps in t_rep_loader:

                l_loss, r_loss, t_loss = self.train_batch(epoch, reps)

                running_latent += l_loss
                running_recon += r_loss
                running_total += t_loss

            for reps in v_rep_loader:
                vl_loss, vr_loss,  vt_loss = self.eval_batch(reps)
                running_val += vt_loss

            tot_loss = running_total/len(t_rep_loader)*batch_size
            latent_loss = running_latent/len(t_rep_loader)*batch_size
            recon_loss = running_recon/len(t_rep_loader)*batch_size
            val_tot_loss = running_val/len(v_rep_loader)*batch_size
            #Print data
            if (epoch+1)%self.print_loss == 0:
                print('[{:3d}/{:3d}] {:10.3f} {:10.5} {:10.5} {:10.5} {:10.3}'.format
                    (self.init_epoch+epoch+1, self.init_epoch+num_epochs, tot_loss,
                     latent_loss, recon_loss, val_tot_loss, self.scheduler.get_last_lr()[0]), flush=True)
            self.scheduler.step()

        self.init_epoch += num_epochs

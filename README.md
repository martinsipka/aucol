# Collective variable project on representations

AuCol is a python package for learning collective variables and latent representations of chemical systems employing pre-trained or fixed atomic representations.

The package currently builds on top of the Schnetpack package. Two possible approaches to generate atomic representations are currently:

- Atomic centered symmetric functions (ACSF) for smaller systems with no NNPs [1]
- Pre-trained Schnet network if they are available [2]

The metadynamics package is currently linked to a [modified version of the PLUMED](https://github.com/martinsipka/plumed2) library using a modified version of the ASE PLUMED calculator.

The main uses of the package are the following:

 1. Create an automatic collective variable learning workflow, possibly employing iterative learning procedure
 2. Use collective variables defined as neural network in PLUMED library. Send CV value and gradient using PLUMED cmd interface.

Example use of the library is demonstrated on a set of examples. For some a neural network potential is needed, for the main simple case ACSF are employed. The example can be downloaded from [Google drive](https://drive.google.com/drive/folders/1I2hI5Q3RAXnJHpoNgVuJ63V-8G3yp5V9?usp=sharing)

If you use this library please cite:

    @misc{https://doi.org/10.48550/arxiv.2203.08097,
	doi = {10.48550/ARXIV.2203.08097},
	url = {https://arxiv.org/abs/2203.08097},
	author = {Šípka, Martin and Erlebach, Andreas and Grajciar, Lukáš}
	keywords = {Chemical Physics (physics.chem-ph), FOS: Physical sciences, 	FOS: Physical sciences},
	title = {Understanding chemical reactions via variational autoencoder and atomic representations},
	publisher = {arXiv},



[1] Behler, J. Atom-centered symmetry functions for construct-
ing high-dimensional neural network potentials. The
Journal of Chemical Physics, 134(7):074106, 2011. doi:
10.1063/1.3553717. URL https://doi.org/10.
1063/1.3553717.

[2] Schütt, K. T., Kessel, P., Gastegger, M., Nicoli, K. A.,
Tkatchenko, A., and Müller, K.-R. Schnetpack: A
deep learning toolbox for atomistic systems. Journal
of Chemical Theory and Computation, 15(1):448–455,

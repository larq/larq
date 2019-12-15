# Papers using Larq

One of the main focuses of Larq is to accelerate research on neural networks with extremely low precision weights and activations.

If you publish a paper using Larq please let us know and [add it to the list below](https://github.com/larq/larq/edit/master/docs/papers.md). Feel free to also add the author names, abstract and links to the paper and source code.

<h2><a class="headerlink" style="float:right; opacity: 1;" href="https://github.com/plumerai/rethinking-bnn-optimization" title="Source code"><i class="md-icon">code</i></a> <a class="headerlink" style="float:right; opacity: 1;" href="https://github.com/plumerai/rethinking-bnn-optimization/raw/master/poster.pdf" title="Poster"><i class="md-icon">collections</i></a> <a class="headerlink" style="float:right; opacity: 1;" href="https://papers.nips.cc/paper/8971-latent-weights-do-not-exist-rethinking-binarized-neural-network-optimization" title="arXiv paper"><i class="md-icon">library_books</i></a> Latent Weights Do Not Exist: Rethinking Binarized Neural Network Optimization</h2>

<div style="color: rgba(0, 0, 0, 0.54);">Koen Helwegen, James Widdicombe, Lukas Geiger, Zechun Liu, Kwang-Ting Cheng, Roeland Nusselder</div>

Optimization of Binarized Neural Networks (BNNs) currently relies on real-valued latent weights to accumulate small update steps. In this paper, we argue that these latent weights cannot be treated analogously to weights in real-valued networks. Instead their main role is to provide inertia during training. We interpret current methods in terms of inertia and provide novel insights into the optimization of BNNs. We subsequently introduce the first optimizer specifically designed for BNNs, Binary Optimizer (Bop), and demonstrate its performance on CIFAR-10 and ImageNet. Together, the redefinition of latent weights as inertia and the introduction of Bop enable a better understanding of BNN optimization and open up the way for further improvements in training methodologies for BNNs.

# -*- coding: utf-8 -*-
"""Main module."""
from typing import Callable, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import torch
from scvi import REGISTRY_KEYS
from scvi._compat import Literal
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.nn import DecoderSCVI, Encoder
from torch import logsumexp
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from ._utils import GumbelSigmoid

torch.backends.cudnn.benchmark = True


# VAE model
class SpikeSlabVAEModule(BaseModuleClass):
    """
    Variational auto-encoder model.

    This is light reimplementation of the scVI model described in [Lopez18]_
    with a spike slab prior for sparse mechanism shift modeling

    Parameters
    ----------
    n_input
        Number of input genes
    n_batch
        Number of batches, if 0, no batch correction is performed.
    n_labels
        Number of labels
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    n_continuous_cov
        Number of continuous covarites
    n_cats_per_cov
        Number of categories for each extra categorical covariate
    dropout_rate
        Dropout rate for neural networks
    latent_distribution
        One of

        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)
    encode_covariates
        Whether to concatenate covariates to expression in encoder
    deeply_inject_covariates
        Whether to concatenate covariates into output of hidden layers in encoder/decoder. This option
        only applies when `n_layers` > 1. The covariates are concatenated to the input of subsequent hidden layers.
    use_layer_norm
        Whether to use layer norm in layers
    var_activation
        Callable used to ensure positivity of the variational distributions' variance.
        When `None`, defaults to `torch.exp`.
    """

    def __init__(
        self,
        n_input: int,
        n_input_actions: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        gumbel_network_type: str = 'noNetwork',
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate: float = 0.1,
        latent_distribution: str = "normal",
        encode_covariates: bool = False,
        use_chem_prior: bool = True,
        deeply_inject_covariates: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        var_activation: Optional[Callable] = None,
        
    ):
        super().__init__()
        self.n_latent = n_latent
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.gumbel_network_type = gumbel_network_type
        self.n_input_actions = n_input_actions
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.use_chem_prior = use_chem_prior
        self.beta = 1
        self.warmup = True
        self.sparse_mask_penalty = 1
        self.px_r = torch.nn.Parameter(torch.randn(n_input))

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        n_input_encoder = n_input + n_continuous_cov * encode_covariates
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        encoder_cat_list = cat_list if encode_covariates else None
        self.z_encoder = Encoder(
            n_input_encoder,
            n_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
        )
        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = Encoder(
            n_input_encoder,
            1,
            n_layers=1,
            n_cat_list=encoder_cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
        )
        # decoder goes from n_latent-dimensional space to n_input-d data
        n_input_decoder = n_latent + n_continuous_cov
        self.decoder = DecoderSCVI(
            n_input_decoder,
            n_input,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            scale_activation="softmax",
        )


        # mu_a
        self.action_prior_mean = torch.nn.parameter.Parameter(
            torch.randn((n_labels, n_latent))
        )
        # mixture weights
        self.w = torch.nn.Parameter(torch.randn(self.action_prior_mean.shape[0]))


        # p_a
        self.action_prior_logit_weight = torch.nn.parameter.Parameter(
            1 * torch.ones((n_labels, n_latent))
        )
        # q_a
        self.gumbel_action = GumbelSigmoid(num_action=n_labels, num_latent=n_latent, 
                                           n_input_actions = n_input_actions, gumbel_network_type = gumbel_network_type) 
        self.use_global_kl = True
        self.learnable_actions = True if n_input_actions is not None else False

    def _get_inference_input(self, tensors):
        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        # y = tensors[REGISTRY_KEYS.LABELS_KEY]

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        input_dict = dict(
            x=x, batch_index=batch_index, cont_covs=cont_covs, cat_covs=cat_covs
        )
        return input_dict
    # override _get_generative_input
    def _get_generative_input(self, tensors, inference_outputs, generative_replay_tensors=None): 
        z = inference_outputs["z"]
        library = inference_outputs["library"]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        
        if generative_replay_tensors is not None:
            x_gr = generative_replay_tensors[REGISTRY_KEYS.X_KEY] #
            y = generative_replay_tensors[REGISTRY_KEYS.LABELS_KEY] #
        else:
            y = tensors[REGISTRY_KEYS.LABELS_KEY]
            x_gr = None

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        input_dict = dict(
            z=z,
            library=library,
            batch_index=batch_index,
            y=y,
            x_gr = x_gr,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
            
            
        )
        return input_dict

    @auto_move_data
    def inference(self, x, batch_index, cont_covs=None, cat_covs=None, n_samples=1):
        """
        High level inference method.

        Runs the inference (encoder) model.
        """
        x_ = x
        library = torch.log(x.sum(1)).unsqueeze(1)
        x_ = torch.log(1 + x_)

        if cont_covs is not None and self.encode_covariates:
            encoder_input = torch.cat((x_, cont_covs), dim=-1)
        else:
            encoder_input = x_
        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()
        qz, z = self.z_encoder(encoder_input, batch_index, *categorical_input)
        ql = None

        if n_samples > 1:
            untran_z = qz.sample((n_samples,))
            z = self.z_encoder.z_transformation(untran_z)
            library = library.unsqueeze(0).expand(
                (n_samples, library.size(0), library.size(1))
            )
        outputs = dict(z=z, qz=qz, ql=ql, library=library)
        return outputs

    @auto_move_data
    def generative(
        self,
        z,
        library,
        batch_index,
        cont_covs=None,
        cat_covs=None,
        y=None,
        x_gr=None, #
        transform_batch=None,
    ):
        """Runs the generative model."""
        if cont_covs is None:
            decoder_input = z
        elif z.dim() != cont_covs.dim():
            decoder_input = torch.cat(
                [z, cont_covs.unsqueeze(0).expand(z.size(0), -1, -1)], dim=-1
            )
        else:
            decoder_input = torch.cat([z, cont_covs], dim=-1)

        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        # order is px_scale, px_r, px_rate, px_dropout
#         px_scale, _, px_rate, _ = self.decoder(
#             "gene",
#             decoder_input,
#             library,
#             batch_index,
#             *categorical_input,
#             #            y, IMPORTANT TO BE TAKEN AWAY, otherwise we have leakage of y into the decoder by other means than the shift
#         )
        px_scale, _, px_rate, px_dropout = self.decoder(
            "gene",
            decoder_input,
            library,
            batch_index,
            *categorical_input,
            #            y, IMPORTANT TO BE TAKEN AWAY, otherwise we have leakage of y into the decoder by other means than the shift
        )
        px_r = torch.exp(self.px_r)
#         px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)
        px = ZeroInflatedNegativeBinomial(
                mu=px_rate,
                theta=px_r,
                zi_logits=px_dropout,
                scale=px_scale,
            )

        # Priors
        pl = None
        # sample mask (size chemical times latent) for each datapoint
        mask = self.gumbel_action(y[:, 0].long(), x_gr)  # batch x latentdim

        # subsample actions we care about
        # extract chemical specific means

        # print(y)
        # print(y[:,0].long())
        # print(mask)

        
        
        mean_z = torch.index_select(
            self.action_prior_mean, 0, y[:, 0].long().to(self.action_prior_mean.device)
        )  # batch x latent dim

        
        # prune out entries according to mask
        # mean_z_pruned = mean_z*mask
        #order labels
        # w = torch.sigmoid(self.w)
        # w = w.squeeze()
        # batch_size = mean_z_pruned.shape[0]
        # y_multinom_labels = torch.multinomial(w, batch_size, replacement = True)
        # y_order = self.match_and_sort_labels(y[:,0], y_multinom_labels)
        # mean_z_pruned = mean_z_pruned[y_order,:]
        if self.use_chem_prior:
            pz = Normal(mean_z*mask, torch.ones_like(z))
        else:
            pz = Normal(torch.zeros_like(z), torch.ones_like(z))
        # we will enforce proba of mask to be sparse, so that means that most of the time mask should be zero, and turn off the action specific prior

        return dict(
            px=px,
            pl=pl,
            pz=pz,
            putative_labels = y[:, 0].long()
        )

    def freeze_params(self):
        # freeze
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.z_encoder.parameters():
            param.requires_grad = False
        self.px_r.requires_grad = False
        self.action_prior_logit_weight.requires_grad = False

        for _, mod in self.decoder.named_modules():
            if isinstance(mod, torch.nn.BatchNorm1d):
                mod.momentum = 0
        for _, mod in self.z_encoder.named_modules():
            if isinstance(mod, torch.nn.BatchNorm1d):
                mod.momentum = 0

    def reinit_actsparse_and_freeze(self, loc):
        # here we must reinit for the embeddings that may have been shrank by sparsity
        with torch.no_grad():
            self.action_prior_mean[loc] = 0
            self.gumbel_action.log_alpha[loc] = 5
        self.gumbel_action.threshold()
    
    def match_and_sort_labels(self,y, y_b):
        unique_to_y = list(set(y.tolist())- set(y_b.tolist()))
        y_distinct = torch.unique(y_b)

        y_b_indexes = torch.full((y_b.shape[0],),-1)
        nonused_idx_y = []
        nonused_idx_y_b = []
        for item in unique_to_y:
            idx_y = (y==item).nonzero().squeeze()
            if idx_y.numel() == 1:
                nonused_idx_y.append(idx_y.item())
            else:
                nonused_idx_y.extend(idx_y.tolist())

        for item in y_distinct:
            idx_y_b = (y_b == item).nonzero().squeeze()
            idx_y = (y == item).nonzero().squeeze()
            #if number of found elements is equal:
            if idx_y_b.numel() == idx_y.numel():
                if idx_y_b.numel() == 1:
                    y_b_indexes[idx_y_b.item()] = idx_y.item()
                else:
                    y_b_indexes[idx_y_b.tolist()] = torch.tensor(idx_y.tolist())
            # if number of found elements is not equal
            elif idx_y_b.numel() < idx_y.numel(): 
                if idx_y_b.numel() == 1:
                    y_b_indexes[idx_y_b.item()] = idx_y.tolist()[0]
                    nonused_idx_y.extend(idx_y.tolist()[1:])
                else:
                    y_b_indexes[idx_y_b.tolist()] = torch.tensor(idx_y.tolist()[:idx_y_b.numel()])
                    nonused_idx_y.extend(idx_y.tolist()[idx_y_b.numel():])
            elif idx_y_b.numel() > idx_y.numel():
                if idx_y.numel() == 0:
                    if idx_y_b.numel() == 1:
                        nonused_idx_y_b.append(idx_y_b.item())
                    elif idx_y_b.numel() > 1:
                        nonused_idx_y_b.extend(idx_y_b.tolist())
                elif idx_y.numel() == 1:
                    y_b_indexes[idx_y_b.tolist()[0]] = idx_y.item()
                    if len(idx_y_b.tolist()[1:]) == 1:
                        nonused_idx_y_b.append(idx_y_b.tolist()[1])
                    else:
                        nonused_idx_y_b.extend(idx_y_b.tolist()[1:])
                else:
                    y_b_indexes[idx_y_b.tolist()[0:idx_y.numel()]] = torch.tensor(idx_y.tolist()[:idx_y.numel()])
                    nonused_idx_y_b.extend(idx_y_b.tolist()[idx_y.numel():])
        if nonused_idx_y_b != []:
            y_b_indexes[nonused_idx_y_b[:]] = torch.tensor(nonused_idx_y[:])
        return y_b_indexes

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        replay_tensors=None,
        replay_inference_outputs=None,
        replay_generative_outputs=None,
        kl_weight: float = 1.0,
        replay_importance=0.0,
        ewc_importance=0.0,
        n_obs: int = 1.0,
    ):
        x = tensors[REGISTRY_KEYS.X_KEY]
        
        
        m = generative_outputs['putative_labels']

        # cluster assignment based on incoming labels (rather than all classes)
        w_ = torch.index_select(
            torch.sigmoid(self.w), 0, m.to(self.action_prior_mean.device)
        )
      
        clust_idx = torch.multinomial(w_, x.shape[0], replacement = True)

        mean_qz = inference_outputs['qz'].mean
        scale_qz = inference_outputs['qz'].scale
        mean_qz = mean_qz[clust_idx]
        scale_qz = scale_qz[clust_idx]
        inference_outputs['qz'] = torch.distributions.normal.Normal(mean_qz, scale_qz)

        kl_divergence_z = kl(inference_outputs["qz"], generative_outputs["pz"]).sum(
            dim=1
        )

        
        kl_divergence_l = 0.0
        reconst_loss = -generative_outputs["px"].log_prob(x).sum(-1)

        kl_local_for_warmup = kl_divergence_z
        kl_local_no_warmup = kl_divergence_l

        if self.warmup:
            weighted_kl_local = (
                self.beta * kl_weight * kl_local_for_warmup + kl_local_no_warmup 
            )
        else:
            weighted_kl_local = kl_local_for_warmup + kl_local_no_warmup 

        kl_local = dict(
            kl_divergence_l=kl_divergence_l, kl_divergence_z=kl_divergence_z
        )

        q_discrete = self.gumbel_action.get_proba()
        p_discrete = torch.sigmoid(self.action_prior_logit_weight)

        kl_discrete = torch.sum(q_discrete * torch.log(q_discrete / p_discrete))
        prior_w = torch.ones_like(self.action_prior_logit_weight)
        logp_qw = (
            torch.distributions.Beta(prior_w, prior_w * self.sparse_mask_penalty)
            .log_prob(q_discrete)
            .sum()
        )
        logp_w = (
            torch.distributions.Beta(prior_w, prior_w * self.sparse_mask_penalty)
            .log_prob(p_discrete)
            .sum()
        )

        # mixture weight prior
        prior_mw = torch.ones_like(self.w)
        w_discrete =  torch.sigmoid(self.w) # torch.nn.functional.softmax(self.w, dim=0)
        logp_mw = -(
            torch.distributions.Beta(prior_mw, prior_mw*self.sparse_mask_penalty)
            .log_prob(w_discrete)
            .sum()
        )
        
        replay_loss = torch.tensor(0.0)
        replay_loss = replay_loss.to(self.device)
        
        replay_reconst_loss = torch.tensor(0.0)
        replay_reconst_loss = replay_reconst_loss.to(self.device)
        
        if replay_tensors is not None:
            x_replay = replay_tensors[REGISTRY_KEYS.X_KEY]
            replay_kl_divergence_z = kl(replay_inference_outputs["qz"], replay_generative_outputs["pz"]).sum(
                dim=1
            )
            replay_reconst_loss = -replay_generative_outputs["px"].log_prob(x).sum(-1)
            replay_loss = n_obs * torch.mean(replay_reconst_loss + replay_kl_divergence_z) # Requires full ELBO instead?

        
        
        penalty = torch.tensor(0.0)
        penalty = penalty.to(self.device)
        

        if ewc_importance > 0.0 and replay_tensors is not None:
            # EWC regularisation
            keep_params = [n for n,p in self.old_params]

            cur_params = [ (n, p) for n,p in self.named_parameters() if n in keep_params]

            for (_, cur_param), (n, saved_param), (_, imp) in zip(
                    cur_params,
                    self.old_params,
                    self.importances,
                ):
                if cur_param.size() == saved_param.size():
#                     print("penalty computed")
                    penalty += (imp * (cur_param - saved_param).pow(2)).sum()  
                    
                else:
                    penalty += 0.0
                
 
        
        if self.use_global_kl:
            # Implementation detail described in the paper: line below describes the mathematical derivations in the paper
            # kl_global = torch.tensor(0.0) + kl_discrete - logp_w
            # Line below is the practical implementation, setting p_discrete to q_discrete
            kl_global = torch.tensor(0.0) - logp_qw
            
            loss = (
                n_obs * torch.mean(reconst_loss + weighted_kl_local)
                + kl_weight * kl_global + replay_importance*replay_loss + ewc_importance*penalty + logp_mw
            )
        else:
            loss = n_obs * torch.mean(reconst_loss + weighted_kl_local) + replay_importance*replay_loss + ewc_importance*penalty
            kl_global = torch.tensor(0.0)

        return LossRecorder(loss, reconst_loss, kl_local, kl_global, replay_reconst_loss=torch.mean(replay_reconst_loss),
                            ewc_loss=penalty, mixture_weight_prior = logp_mw) 

    @torch.no_grad()
    @auto_move_data
    def marginal_ll(self, tensors, n_mc_samples):
        sample_batch = tensors[REGISTRY_KEYS.X_KEY]
        to_sum = torch.zeros(sample_batch.size()[0], n_mc_samples)

        for i in range(n_mc_samples):
            # Distribution parameters and sampled variables
            inference_outputs, generative_outputs, losses = self.forward(tensors)
            z = inference_outputs["z"]

            p_za = generative_outputs["pz"].log_prob(z).sum(dim=1)
            q_z_x = inference_outputs["qz"].log_prob(z).sum(dim=1)
            p_x_za = -losses.reconstruction_loss
            log_prob_sum = p_x_za + p_za - q_z_x
            to_sum[:, i] = log_prob_sum

        batch_log_lkl = logsumexp(to_sum, dim=-1) - np.log(n_mc_samples)
        return batch_log_lkl
    
    # override forward
    @auto_move_data
    def forward(
        self,
        tensors,
        generative_replay_tensors=None,
        replay_tensors = None,
        get_inference_input_kwargs: Optional[dict] = None,
        get_generative_input_kwargs: Optional[dict] = None,
        inference_kwargs: Optional[dict] = None,
        generative_kwargs: Optional[dict] = None,
        loss_kwargs: Optional[dict] = None,
        compute_loss=True,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, LossRecorder],
    ]:
        """
        Forward pass through the network.

        Parameters
        ----------
        tensors
            tensors to pass through
        get_inference_input_kwargs
            Keyword args for ``_get_inference_input()``
        get_generative_input_kwargs
            Keyword args for ``_get_generative_input()``
        inference_kwargs
            Keyword args for ``inference()``
        generative_kwargs
            Keyword args for ``generative()``
        loss_kwargs
            Keyword args for ``loss()``
        compute_loss
            Whether to compute loss on forward pass. This adds
            another return value.
        """
        return _generic_forward(
            self,
            tensors,
            generative_replay_tensors,
            replay_tensors,
            inference_kwargs,
            generative_kwargs,
            loss_kwargs,
            get_inference_input_kwargs,
            get_generative_input_kwargs,
            compute_loss,
        )

def _get_dict_if_none(param):
    param = {} if not isinstance(param, dict) else param

    return param

def _generic_forward(
    module,
    tensors,
    generative_replay_tensors,
    replay_tensors,
    inference_kwargs,
    generative_kwargs,
    loss_kwargs,
    get_inference_input_kwargs,
    get_generative_input_kwargs,
    compute_loss,
):
    """Core of the forward call shared by PyTorch- and Jax-based modules."""
    inference_kwargs = _get_dict_if_none(inference_kwargs)
    generative_kwargs = _get_dict_if_none(generative_kwargs)
    loss_kwargs = _get_dict_if_none(loss_kwargs)
    get_inference_input_kwargs = _get_dict_if_none(get_inference_input_kwargs)
    get_generative_input_kwargs = _get_dict_if_none(get_generative_input_kwargs)

    inference_inputs = module._get_inference_input(
        tensors, **get_inference_input_kwargs
    )

    inference_outputs = module.inference(**inference_inputs, **inference_kwargs)
    
    if module.learnable_actions:
        generative_inputs = module._get_generative_input(
        tensors, inference_outputs, generative_replay_tensors, **get_generative_input_kwargs
        )
    else:
        generative_inputs = module._get_generative_input(
        tensors, inference_outputs, **get_generative_input_kwargs
    )
    
    generative_outputs = module.generative(**generative_inputs, **generative_kwargs)
    
    
    if replay_tensors is not None:
        replay_inference_inputs = module._get_inference_input(
            replay_tensors, **get_inference_input_kwargs
        )
        replay_inference_outputs = module.inference(**replay_inference_inputs, **inference_kwargs)
        replay_generative_inputs = module._get_generative_input(
            replay_tensors, replay_inference_outputs, generative_replay_tensors, **get_generative_input_kwargs
        )
        replay_generative_outputs = module.generative(**replay_generative_inputs, **generative_kwargs)
    
    if compute_loss and replay_tensors is not None:
        losses = module.loss(
            tensors, inference_outputs, generative_outputs, replay_tensors,
            replay_inference_outputs, replay_generative_outputs, **loss_kwargs
        )
        return inference_outputs, generative_outputs, losses
    elif compute_loss and replay_tensors is None:
        losses = module.loss(
            tensors,inference_outputs, generative_outputs, 
#             replay_tensors=None, replay_inference_outputs=None, replay_generative_outputs=None, 
            **loss_kwargs
        )
        return inference_outputs, generative_outputs, losses
    else:
        return inference_outputs, generative_outputs

import logging
from typing import List, Optional, Sequence, Any, Dict, Union, Tuple

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi._compat import Literal
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
)
from scvi.model.base import (
    ArchesMixin,
    BaseModelClass,
    RNASeqMixin,
#     UnsupervisedTrainingMixin,
    VAEMixin,
)
from scvi.utils import setup_anndata_dsp

from ._module import SpikeSlabVAEModule


# add-ons ------
from scvi.dataloaders._ann_dataloader import AnnDataLoader
from ._trainingplans import CLTrainingPlan
from scvi.dataloaders import DataSplitter
from scvi.train import TrainingPlan, TrainRunner
import os
from scvi.data._download import _download
from scvi.model._scvi import SCVI
from copy import deepcopy
from scvi.train import TrainingPlan


logger = logging.getLogger(__name__)

# override UnsupervisedTrainingMixin to use modified TrainingPlan

def _check_warmup(
    plan_kwargs: Dict[str, Any],
    max_epochs: int,
    n_cells: int,
    batch_size: int,
    train_size: float = 1.0,
) -> None:
    """
    Raises a warning if the max_kl_weight is not reached by the end of training.

    Parameters
    ----------
    plan_kwargs
        Keyword args for :class:`~scvi.train.TrainingPlan`.
    max_epochs
        Number of passes through the dataset.
    n_cells
        Number of cells in the whole datasets.
    batch_size
        Minibatch size to use during training.
    train_size
        Fraction of cells used for training.
    """
    _WARNING_MESSAGE = (
        "max_{mode}={max} is less than n_{mode}_kl_warmup={warm_up}. "
        "The max_kl_weight will not be reached during training."
    )

    n_steps_kl_warmup = plan_kwargs.get("n_steps_kl_warmup", None)
    n_epochs_kl_warmup = plan_kwargs.get("n_epochs_kl_warmup", None)

    # The only time n_steps_kl_warmup is used is when n_epochs_kl_warmup is explicitly
    # set to None. This also catches the case when both n_epochs_kl_warmup and
    # n_steps_kl_warmup are set to None and max_kl_weight will always be reached.
    if (
        "n_epochs_kl_warmup" in plan_kwargs
        and plan_kwargs["n_epochs_kl_warmup"] is None
    ):
        n_cell_train = ceil(train_size * n_cells)
        steps_per_epoch = n_cell_train // batch_size + (n_cell_train % batch_size >= 3)
        max_steps = max_epochs * steps_per_epoch
        if n_steps_kl_warmup and max_steps < n_steps_kl_warmup:
            warnings.warn(
                _WARNING_MESSAGE.format(
                    mode="steps", max=max_steps, warm_up=n_steps_kl_warmup
                )
            )
    elif n_epochs_kl_warmup:
        if max_epochs < n_epochs_kl_warmup:
            warnings.warn(
                _WARNING_MESSAGE.format(
                    mode="epochs", max=max_epochs, warm_up=n_epochs_kl_warmup
                )
            )
    else:
        if max_epochs < 400:
            warnings.warn(
                _WARNING_MESSAGE.format(mode="epochs", max=max_epochs, warm_up=400)
            )
    
def _get_loaded_data(reference_model, device=None):
    if isinstance(reference_model, str):
        attr_dict, var_names, load_state_dict, _ = _load_saved_files(
            reference_model, load_adata=False, map_location=device
        )
    else:
        attr_dict = reference_model._get_user_attributes()
        attr_dict = {a[0]: a[1] for a in attr_dict if a[0][-1] == "_"}
        var_names = reference_model.adata.var_names
        load_state_dict = deepcopy(reference_model.module.state_dict())

    return attr_dict, var_names, load_state_dict    

def _load_saved_files(
    dir_path: str,
    load_adata: bool,
    prefix: Optional[str] = None,
    map_location: Optional[Literal["cpu", "cuda"]] = None,
    backup_url: Optional[str] = None,
) -> Tuple[dict, np.ndarray, dict, AnnData]:
    """Helper to load saved files."""
    file_name_prefix = prefix or ""

    model_file_name = f"{file_name_prefix}model.pt"
    model_path = os.path.join(dir_path, model_file_name)
    try:
        _download(backup_url, dir_path, model_file_name)
        model = torch.load(model_path, map_location=map_location)
    except FileNotFoundError as exc:
        raise ValueError(
            f"Failed to load model file at {model_path}. "
            "If attempting to load a saved model from <v0.15.0, please use the util function "
            "`convert_legacy_save` to convert to an updated format."
        ) from exc

    model_state_dict = model["model_state_dict"]
    var_names = model["var_names"]
    attr_dict = model["attr_dict"]

    if load_adata:
        is_mudata = attr_dict["registry_"].get(_SETUP_METHOD_NAME) == "setup_mudata"
        file_suffix = "adata.h5ad" if is_mudata is False else "mdata.h5mu"
        adata_path = os.path.join(dir_path, f"{file_name_prefix}{file_suffix}")
        if os.path.exists(adata_path):
            if is_mudata:
                adata = mudata.read(adata_path)
            else:
                adata = anndata.read(adata_path)
        else:
            raise ValueError(
                "Save path contains no saved anndata and no adata was passed."
            )
    else:
        adata = None

    return attr_dict, var_names, model_state_dict, adata
class UnsupervisedTrainingMixin:
    """General purpose unsupervised train method."""

    def train(
        self,
        max_epochs: Optional[int] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        early_stopping: bool = False,
        plan_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        """
        Train the model.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        early_stopping
            Perform early stopping. Additional arguments can be passed in `**kwargs`.
            See :class:`~scvi.train.Trainer` for further options.
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **trainer_kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        n_cells = self.adata.n_obs
        if max_epochs is None:
            max_epochs = int(np.min([round((20000 / n_cells) * 400), 400]))

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()

        _check_warmup(plan_kwargs, max_epochs, n_cells, batch_size)

        data_splitter = DataSplitter(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )
        
        if self.useCLTrainingPlan:
            training_plan = CLTrainingPlan(self.module, 
                                         self.generative_replay_dl,
                                         self.generative_replay_adata_manager,
                                         self.replay_adata_manager, self.device,
                                         **plan_kwargs)
        elif self.module.learnable_actions and not self.useCLTrainingPlan:
            training_plan = CLTrainingPlan(self.module, 
                                           self.generative_replay_dl,
                                           self.generative_replay_adata_manager,
                                           device = self.device,
                                           **plan_kwargs)
        else:
            training_plan = TrainingPlan(self.module,
                                           **plan_kwargs)
            

        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            **trainer_kwargs,
        )
        return runner()

class SpikeSlabVAE(
    RNASeqMixin, VAEMixin, ArchesMixin, UnsupervisedTrainingMixin, BaseModelClass
):
    """
    single-cell Variational Inference [Lopez18]_.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~scvi.model.SCVI.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    dropout_rate
        Dropout rate for neural networks.
    dispersion
        One of the following:

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    gene_likelihood
        One of:

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    latent_distribution
        One of:

        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal distribution (Normal(0, I) transformed by softmax)
    **model_kwargs
        Keyword args for :class:`~scvi.module.VAE`

    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> scvi.model.SCVI.setup_anndata(adata, batch_key="batch")
    >>> vae = scvi.model.SCVI(adata)
    >>> vae.train()
    >>> adata.obsm["X_scVI"] = vae.get_latent_representation()
    >>> adata.obsm["X_normalized_scVI"] = vae.get_normalized_expression()

    Notes
    -----
    See further usage examples in the following tutorials:

    1. :doc:`/tutorials/notebooks/api_overview`
    2. :doc:`/tutorials/notebooks/harmonization`
    3. :doc:`/tutorials/notebooks/scarches_scvi_tools`
    4. :doc:`/tutorials/notebooks/scvi_in_R`
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        gumbel_network_type: str = 'noNetwork',
        dropout_rate: float = 0.1,
        latent_distribution: Literal["normal", "ln"] = "normal",
        **module_kwargs,
    ):
        super().__init__(adata)

        n_cats_per_cov = (
            self.adata_manager.get_state_registry(
                REGISTRY_KEYS.CAT_COVS_KEY
            ).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        n_batch = self.summary_stats.n_batch

        self.module = SpikeSlabVAEModule(
            n_input=self.summary_stats.n_vars,
            n_batch=n_batch,
            n_labels=self.n_labels if self.n_labels is not None else self.summary_stats.n_labels, # 
            n_input_actions = self.n_input_actions if self.n_labels is not None else None, #  note our actions is different from actions in svae
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_cats_per_cov=n_cats_per_cov,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            gumbel_network_type=gumbel_network_type,
            dropout_rate=dropout_rate,
            latent_distribution=latent_distribution,
            **module_kwargs,
        )
        self._model_summary_string = (
            "SpikeSlabVAE model with learnable interventions with the following params: \nn_hidden: {}, n_latent: {}, n_layers: {}, dropout_rate: "
            "{}, latent_distribution: {}, n_actions:{}, n_GenerativeReplay:{}, n_input_actions:{}"
        ).format(
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
            latent_distribution,
            self.n_labels,
            self.n_generativeReplay if self.n_labels is not None else None,
            self.n_input_actions if self.n_labels is not None else None
            
        )
        self.init_params_ = self._get_init_params(locals())
        self.module.useCLTrainingPlan = self.useCLTrainingPlan

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        size_factor_key: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        %(summary)s.

        Parameters
        ----------
        %(param_layer)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False
            ),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
        ]
        
        
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        
        
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
        cls.useCLTrainingPlan = False
        cls.n_labels=None
        
        
    @classmethod
    def setup_replay_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        size_factor_key: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        %(summary)s.

        Parameters
        ----------
        %(param_layer)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False
            ),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
        ]
        
        
       
        
        replay_adata_manager = AnnDataManager(
                fields=anndata_fields, setup_method_args=setup_method_args
            )

        
        replay_adata_manager.register_fields(adata, **kwargs)
        cls.replay_adata_manager = replay_adata_manager
        cls.useCLTrainingPlan = True
        
    
        
        
        
        
    @classmethod
    def prepare_generative_replay_data(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        size_factor_key: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        %(summary)s.

        Parameters
        ----------
        %(param_layer)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
#             CategoricalObsField(REGISTRY_KEYS.ACTION_KEY, labels_key),
            NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False
            ),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
        ]
        generative_replay_adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        generative_replay_adata_manager.register_fields(adata, **kwargs)
        cls.generative_replay_adata_manager = generative_replay_adata_manager
        grdl = AnnDataLoader(generative_replay_adata_manager, batch_size=1) # batch_size=adata.shape[0]
#         cls.replay_adata_tensors = next(tensors for tensors in scdl) # move to device?
        cls.generative_replay_dl = grdl # move to device? # change to dataloader
        cls.n_labels = adata.obs[labels_key].cat.categories.size
        cls.n_input_actions = adata.shape[1]
        cls.n_generativeReplay = adata.shape[0]
    

    
    def from_scvi_model(
        self,
        scvi_model: SCVI,
#         unlabeled_category: str,
#         labels_key: Optional[str] = None,
#         adata: Optional[AnnData] = None,
#         **scanvi_kwargs,
    ):
        """
        Initialize sVAE model with weights from pretrained :class:`~scvi.model.SCVI` model.
        Parameters
        ----------
        scvi_model
            Pretrained scvi model
        labels_key
            key in `adata.obs` for label information. Label categories can not be different if
            labels_key was used to setup the SCVI model. If None, uses the `labels_key` used to
            setup the SCVI model. If that was None, and error is raised.
        unlabeled_category
            Value used for unlabeled cells in `labels_key` used to setup AnnData with scvi.
        adata
            AnnData object that has been registered via :meth:`~scvi.model.SCANVI.setup_anndata`.
        scanvi_kwargs
            kwargs for scANVI model
        """
        
#         self.useCLTrainingPlan = True
        
        def requires_penalty(key):
            one = 'z_encoder' in key.split(".")[0] 
#             one = 'z_encoder.encoder.fc_layers.Layer 0.0' in key
#             two = 'decoder.px_decoder.fc_layers.Layer 0.0' in key
#             one = 'encoder' in key.split(".") 
            two = 'decoder' in key.split(".") 
            three = 'gumbel_action' in key
            four = 'action_prior_mean' in key
#             two = 'px_decoder' in key.split(".")
#             three = 'px_scale_decoder' in key.split(".")
#             four = 'px_r_decoder' in key.split(".")
#             five = 'px_dropout_decoder' in key.split(".")
            
#             if one or two: # 
#             if two: # 
            if one:
#             if three:

                return True
            else:
                return False
            
        
        
        def zerolike_params_dict(model):
            """
            Create a list of (name, parameter), where parameter is initalized to zero.
            The list has as many parameters as model, with the same size.
            :param model: a pytorch model
            """

            return [
                (k, torch.zeros_like(p).to(p.device))
                for k, p in model.named_parameters() if requires_penalty(k)
            ]
        
        def compute_importances(
            model, optimizer, dataloader, device
        ):
 
            
            importances = zerolike_params_dict(model)

            
            model.eval()
            
            for i, batch in enumerate(dataloader):
                # get only input, target and task_id from the batch
                tensors = batch
                
                ## move tensors to device 
                for key, val in tensors.items():
                    tensors[key] = val.to(device)

                optimizer.zero_grad()
                inference_inputs = model._get_inference_input(tensors)
                inference_outputs = model.inference(**inference_inputs)

                generative_inputs = model._get_generative_input(tensors, inference_outputs)
                generative_outputs = model.generative(**generative_inputs)
                scvi_loss = model.loss(tensors, inference_outputs, generative_outputs)
                loss = scvi_loss.loss
                loss.backward()
                
                param_dict = [ (n,p) for n,p in model.named_parameters() if requires_penalty(n)]
                
            
                for (k1, p), (k2, imp) in zip(
                    param_dict, importances
                ):
                    assert k1 == k2
                    if p.grad is not None:
                        imp += p.grad.data.clone().pow(2)

            # average over mini batch length
            for _, imp in importances:
                imp /= float(len(dataloader))

            return importances
        
        
        
        params = filter(lambda p: p.requires_grad, scvi_model.module.parameters())
        optimizer = torch.optim.Adam(
            params, lr=1e-3, eps=0.01, weight_decay=1e-6
        )
        
        
   
        scvi_model.to_device(self.device)
        scvi_state_dict = deepcopy(scvi_model.module.state_dict())
#         self.to_device(scvi_model.device)
        
        # model tweaking
        new_state_dict = self.module.state_dict()
        for key, load_ten in scvi_state_dict.items():
            new_ten = new_state_dict[key]
            if new_ten.size() == load_ten.size():
                continue
            # new categoricals changed size
            else:
                dim_diff = new_ten.size()[-1] - load_ten.size()[-1]
                fixed_ten = torch.cat([load_ten, new_ten[..., -dim_diff:]], dim=-1)
                scvi_state_dict[key] = fixed_ten

        #do not replace state dict for the action mask
        scvi_state_dict['action_prior_logit_weight'] = new_state_dict['action_prior_logit_weight']
        scvi_state_dict['gumbel_action.log_alpha'] = new_state_dict['gumbel_action.log_alpha']
        scvi_state_dict['gumbel_action.fixed_mask'] = new_state_dict['gumbel_action.fixed_mask']
        scvi_state_dict['action_prior_mean'] = new_state_dict['action_prior_mean']
        scvi_state_dict['w'] = new_state_dict['w']
        keys = ['gumbel_action.log_alpha_encoder.0.fc_layers.Layer 0.0.weight', 
              'gumbel_action.log_alpha_encoder.0.fc_layers.Layer 0.0.bias',
               'gumbel_action.log_alpha_encoder.0.fc_layers.Layer 1.0.weight',
              'gumbel_action.log_alpha_encoder.0.fc_layers.Layer 1.0.bias',
              'gumbel_action.log_alpha_encoder.1.weight',
              'gumbel_action.log_alpha_encoder.1.bias',
               "w_encoder.0.weight", "w_encoder.0.bias"]
        
        
        for key in keys:
            try:
                scvi_state_dict[key] = new_state_dict[key]
            except KeyError:
                pass

        
#         scvi_state_dict['decoder.px_decoder.fc_layers.Layer 0.0.weight'] = scvi_state_dict['decoder.px_decoder.fc_layers.Layer 0.0.weight'][:,:self.module.n_latent]
#         scvi_state_dict['decoder.px_decoder.fc_layers.Layer 1.0.weight'] = scvi_state_dict['decoder.px_decoder.fc_layers.Layer 1.0.weight'][:,:128] # temp fix
        
        scvi_state_dict['decoder.px_decoder.fc_layers.Layer 0.0.weight'] = new_state_dict['decoder.px_decoder.fc_layers.Layer 0.0.weight']
        scvi_state_dict['decoder.px_decoder.fc_layers.Layer 1.0.weight'] = new_state_dict['decoder.px_decoder.fc_layers.Layer 1.0.weight']
        
        self.module.load_state_dict(scvi_state_dict)
#         self.module.to(device)
        self.module.eval()

        self.was_pretrained = True
        
        
        if self.useCLTrainingPlan:
            self.module.importances = compute_importances(scvi_model.module, 
                                                           optimizer, 
                                                           AnnDataLoader(self.replay_adata_manager, batch_size=128),
                                                           scvi_model.device)


            # adds old_params to be penalised to the current model
            self.module.old_params =  [
                    (k, p.clone().detach())
                    for k, p in scvi_model.module.named_parameters() if requires_penalty(k)
                ]   


    @torch.no_grad()
    def get_elbo(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
        agg: bool = False,
    ) -> float:
        """
        Return the ELBO for the data.

        The ELBO is a lower bound on the log likelihood of the data used for optimization
        of VAEs. Note, this is not the negative ELBO, higher is better.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        """
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        # Iterate once over the data and compute the elbo
        elbo = []
        for tensors in scdl:
            _, _, scvi_loss = self.module(tensors)

            recon_loss = scvi_loss.reconstruction_loss
            kl_local = scvi_loss.kl_local
            elbo += [(recon_loss + kl_local).cpu().numpy()]

        # now aggregate by chemical
        elbo = np.concatenate(elbo)
        elbo_res = {}
        label_key = self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)[
            "original_key"
        ]
        cat = self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)[
            "categorical_mapping"
        ]
        for c in cat:
            if indices is not None:
                ind = np.where(adata.obs[label_key][indices].values == c)[0]
            else:
                ind = np.where(adata.obs[label_key].values == c)[0]
            if len(ind) > 10:
                elbo_res[c] = np.mean(elbo[ind])

        if agg:
            return pd.Series(elbo_res).values.mean()
        else:
            return elbo_res

    @torch.no_grad()
    def get_marginal_ll(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        n_mc_samples: int = 1000,
        batch_size: Optional[int] = None,
        agg: bool = False,
    ) -> float:
        """
        Return the marginal LL for the data, calculated by label

        The computation here is a biased estimator of the marginal log likelihood of the data.
        Note, this is not the negative log likelihood, higher is better.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_mc_samples
            Number of Monte Carlo samples to use for marginal LL estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        """
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        if hasattr(self.module, "marginal_ll"):
            log_lkl = []
            for tensors in scdl:
                log_lkl += [self.module.marginal_ll(tensors, n_mc_samples=n_mc_samples)]
        else:
            raise NotImplementedError(
                "marginal_ll is not implemented for current model. "
                "Please raise an issue on github if you need it."
            )

        # now aggregate by chemical
        log_lkl = np.concatenate(log_lkl)
        log_lkl_res = {}
        label_key = self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)[
            "original_key"
        ]
        cat = self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)[
            "categorical_mapping"
        ]
        for c in cat:
            if indices is not None:
                ind = np.where(adata.obs[label_key][indices].values == c)[0]
            else:
                ind = np.where(adata.obs[label_key].values == c)[0]
            if len(ind) > 10:
                log_lkl_res[c] = np.mean(log_lkl[ind])

        if agg:
            return pd.Series(log_lkl_res).values.mean()
        else:
            return log_lkl_res

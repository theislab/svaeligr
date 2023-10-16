import torch
from scvi.nn import FCLayers
from scvi.nn import Encoder

class GumbelSigmoid(torch.nn.Module):
    def __init__(self, num_action, num_latent, n_input_actions=None, gumbel_network_type = 'noNetwork', n_hidden=128, freeze=False, drawhard=True, tau=1):
        super(GumbelSigmoid, self).__init__()
        self.shape = (num_action, num_latent)
        self.freeze = freeze
        self.drawhard = drawhard
        self.log_alpha = torch.nn.Parameter(torch.zeros(self.shape)) # putting this back because is required by reset_parameters() call
        
        self.batch_threshold = 50

        # define distinct networks:
        self.network_type = gumbel_network_type
        self.log_alpha_networks = []

        if self.network_type == "encoderNetworks":
            for i in range(self.shape[0]):
                network_i = Encoder(n_input_actions, num_latent, use_batch_norm = False)
                self.log_alpha_networks.append(network_i)
            
        elif self.network_type == "scNetworks":
            for i in range(self.shape[0]):
                network_i = torch.nn.Sequential(
                FCLayers(
                    n_in=n_input_actions,
                    n_out=n_hidden,
                    n_cat_list=None,
                    n_layers=2,
                    n_hidden=n_hidden,
    #                 dropout_rate=dropout_amortization,
                    use_layer_norm=True,
                    use_batch_norm=False,
    #                 **_extra_encoder_kwargs,
                ),
                torch.nn.Linear(n_hidden, num_latent),
            )
                self.log_alpha_networks.append(network_i)
        elif self.network_type=="singleNetwork":      
            self.log_alpha_encoder = torch.nn.Sequential(
                FCLayers(
                    n_in=n_input_actions,
                    n_out=n_hidden,
                    n_cat_list=None,
                    n_layers=2,
                    n_hidden=n_hidden,
    #                 dropout_rate=dropout_amortization,
                    use_layer_norm=True,
                    use_batch_norm=False,
    #                 **_extra_encoder_kwargs,
                ),
                torch.nn.Linear(n_hidden, num_latent * num_action),
            )

        self.tau = tau
        # useful to make sure these parameters will be pushed to the GPU
        self.uniform = torch.distributions.uniform.Uniform(0, 1)
        self.register_buffer("fixed_mask", torch.ones(self.shape))
        self.reset_parameters()
    # changed this to draw one action per minibatch sample...
    def forward(self, action, x_replay=None):
#         action.to(self.log_alpha.device)
#         x_replay.to(self.log_alpha.device)
        bs = action.shape[0]
        if self.freeze:
            y = self.fixed_mask[action, :]
            return y
        else:
            shape = tuple([bs] + [self.shape[1]])
            logistic_noise = (
                self.sample_logistic(shape)
                .type(self.log_alpha.type())
                .to(self.log_alpha.device)
            )

            y_soft = None

            if self.network_type in ('encoderNetworks', 'scNetworks'):
                for i in range(self.shape[0]):
                    indexes = (action == i)
                    if self.network_type=='encoderNetworks':
                        class_vectors,_,_ = self.log_alpha_networks[i](x_replay[indexes])
                    else:
                        class_vectors = self.log_alpha_networks[i](x_replay[indexes])
                    if class_vectors.size(0) > 0 and bs > self.batch_threshold:
                        self.log_alpha.data[i,:] = torch.mean(class_vectors, dim=0, keepdim = True)
                y_soft = torch.sigmoid((self.log_alpha[action] + logistic_noise)/self.tau)

            elif self.network_type== 'singleNetwork':
                log_alpha = self.log_alpha_encoder(x_replay)
                log_alpha = log_alpha.view(self.shape[0], self.shape[1],-1)
                if bs >= self.batch_threshold:
                    self.log_alpha.data = torch.mean(log_alpha, 2)
                y_soft = torch.sigmoid((self.log_alpha[action] + logistic_noise) / self.tau)
            elif self.network_type == 'noNetwork':
                y_soft = torch.sigmoid((self.log_alpha[action] + logistic_noise) / self.tau)

            if self.drawhard:
                y_hard = (y_soft > 0.5).type(y_soft.type())

                # This weird line does two things:
                #   1) at forward, we get a hard sample.
                #   2) at backward, we differentiate the gumbel sigmoid
                y = y_hard.detach() - y_soft.detach() + y_soft

            else:
                y = y_soft

            return y

    def get_proba(self):
        """Returns probability of getting one"""
        if self.freeze:
            return self.fixed_mask
        else:
            return torch.sigmoid(self.log_alpha)

    def reset_parameters(self):
        torch.nn.init.constant_(
            self.log_alpha, 5
        )  # 5)  # will yield a probability ~0.99. Inspired by DCDI

    def sample_logistic(self, shape):
        u = self.uniform.sample(shape)
        return torch.log(u) - torch.log(1 - u)

    def threshold(self):
        proba = self.get_proba()
        self.fixed_mask.copy_((proba > 0.5).type(proba.type()))
        self.freeze = True

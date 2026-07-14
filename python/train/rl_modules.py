# ===================================================================
# This file is used to create a flate vector observation. Image from 
# overhead camera (84 x 84 x 3) -> CNN -> flatten -> 256-dim image 
# features 


# ===================================================================
import numpy as np
import torch
import torch.nn as nn
import numpy

from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import(
    DefaultPPOTorchRLModule,
)

from ray.rllib.algorithms.sac.torch.default_sac_torch_rl_module import(
    DefaultSACTorchRLModule,
)

from ray.rllib.core.models.base import ENCODER_OUT
from ray.rllib.core.models.specs.specs_dict import SpecDict

class FusionEncoder(nn.Module):
    """CNN(image) + MLP(state) -> fused feature vector. Shared by PPO & SAC"""

    def __init__(self, model_config:dict):
        super().__init__()
        state_dim   = model_config.get("state_dim", 16)
        cam_h       = model_config.get("cam_height", 84)
        cam_w       = model_config.get("cam_width", 84)
        cnn_filters = model_config.get("cnn_filters", [
            [32, [8,8], 4], [64, [4,4], 2], [64, [3,3], 1],
        ])
        fc_hidden   = model_config.get("fc_hidden", [256, 512, 256])

        cnn_layers, in_ch = [], 3
        for out_ch, kernel, stride in cnn_filters:
            cnn_layers += [nn.Conv2d(in_ch, out_ch, kernel_size = kernel, stride=stride), nn.ReLU()]
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        with torch.no_grad():
            cnn_out_flat = int(np.prod(self.cnn(torch.zeros(1,3, cam_h, cam_w)).shape[1:]))

        cnn_embed_dim = 256
        self.cnn_fc = nn.Sequential(nn.Linear(cnn_out_flat, cnn_embed_dim), nn.ReLU())

        state_embed_dim = 128
        self.state_mlp  = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, state_embed_dim), nn.ReLU(),
        )

        in_dim = cnn_embed_dim + state_embed_dim
        fusion_layers = []
        for out_dim in fc_hidden:
            fusion_layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
            in_dim = out_dim
        self.fusion = nn.Sequential(*fusion_layers)
        self.output_dim = in_dim

    def forward(self, batch:dict) -> dict:
        img = batch["image"].float()/255.0
        img = img.permute(0, 3, 1, 2)
        img_features = self.cnn_fc(self.cnn(img).reshape(img.shape[0],-1))
        state_features = self.state_mlp(batch["state"].float())
        fused = torch.cat([img_features, state_features], dim=1)
        return {ENCODER_OUT: self.fusion(fused)}




class ManipulationPPOModule(DefaultPPOTorchRLModule):
    """
    PPO Module using FusionEncoder instea of RLlib's default encoder
    """
    def setup(self):
        super().setup()
        self.encoder = FusionEncoder(self.model_config)

class ManipulationSACModule(DefaultSACTorchRLModule):
    def setup(self):
        super().setup()
        self.pi_encoder = FusionEncoder(self.model_config)
        self.qf_encoder = FusionEncoder(self.model_config)
        self.qf_target_encoder = FusionEncoder(self.model_config)

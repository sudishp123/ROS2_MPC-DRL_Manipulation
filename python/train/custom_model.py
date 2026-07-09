# ===================================================================
# This file is used to create a flate vector observation. Image from 
# overhead camera (84 x 84 x 3) -> CNN -> flatten -> 256-dim image 
# features 


# ===================================================================
import numpy as np
import torch
import torch.nn as nn

from gymnasium import spaces
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from typing import List, Tuple

class ManipulationModel(TorchModelV2, nn.Module):
    """
    Custom RLlib model for the Manipulation-v0 Dict observation space.

    Observation inputs:
        obs["state"]: (B, 16)           - joint state + EE error + obstacle dist
        obs["image"]: (B, 84, 84, 3)    - overhead camera (RLlib sens HWC format)
    
    Outputs:
        policy logits: (B, action_dim)  - 12-dim continuous action (theta_s, theta_r, theta_g)
        value:         (B,)             - scalar state value for PPO critic
    """

    def __init__(
            self,
            obs_space: spaces.Space,
            action_space: spaces.Space,
            num_outputs: int,
            model_config: ModelConfigDict,
            name: str,
    ):
        TorchModelV2.__init__(                                            
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        cfg = model_config.get("custom_model_config", {})

        state_dim   = cfg.get("state_dim", 16)
        cam_h       = cfg.get("cam_height", 84)
        cam_w       = cfg.get("cam_width", 84)
        cnn_filters = cfg.get("cnn_filters",[
            [32, [8,8], 4],
            [64, [4,4], 2],
            [64, [3,3], 1],
        ])
        fc_hidden = cfg.get("fc_hidden", [256, 512, 256])

        # -------- CNN Branch (image) -------------------
        # Input: (B, 3, H, W) - permute from HWC to CHW in forward()
        cnn_layers = []
        in_channels = 3
        for (out_ch, kernel, stride) in cnn_filters:
            cnn_layers += [
                nn.Conv2d(in_channels, out_ch, kernel_size=kernel, stride=stride),
                nn.ReLU(),
            ]
            in_channels = out_ch

        self.cnn = nn.Sequential(*cnn_layers)

        dummy = torch.zeros(1, 3, cam_h, cam_w)
        cnn_out_flat = int(np.prod(self.cnn(dummy).shape[1:]))

        cnn_embed_dim = 256
        self.cnn_fc = nn.Sequential(
            nn.Linear(cnn_out_flat, cnn_embed_dim),
            nn.ReLU()
        )

        # -------- MLP Branch (state) -------------------
        # Simple 2-layer MLP to embed the 16-dim state vector
        state_embed_dim = 128
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, state_embed_dim),
            nn.ReLU(),
        )

        # -------- Concatenate both embeddings -------------------
        fused_dim = cnn_embed_dim + state_embed_dim

        fusion_layers = []
        in_dim = fused_dim

        for out_dim in fc_hidden:
            fusion_layers += [
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
            ]
            in_dim = out_dim
        self.fusion = nn.Sequential(*fusion_layers)

        self.policy_head = nn.Linear(in_dim, num_outputs)

        self.value_head = nn.Linear(in_dim, 1)

        self._features = None

    @override(TorchModelV2)
    def forward(
        self,
        input_dict  : dict,
        state       : List[TensorType],
        seq_lens    : TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        """
        Args:
            input_dict["obs"]["state"]  : (B, 16)
            input_dict["obs]["image]    : (B, 84, 84, 3) uint8 -> normalized to [0,1]    
        
        """

        # image branch:
        img = input_dict["obs"]["image"].float() / 255.0
        img = img.permute(0, 3, 1, 2)
        img_features = self.cnn_fc(
            self.cnn(img).reshape(img.shape[0], -1)
        )

        # state branch:
        state_vec = input_dict["obs"]["state"].float()
        state_features = self.state_mlp(state_vec)

        # fusion:
        fused = torch.cat([img_features, state_features], dim=1)
        self._features = self.fusion(fused)

        logits = self.policy_head(self._features)

        return logits, state
    
    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "forward() must be called before value_function()"
        return self.value_head(self._features).squeeze(1)




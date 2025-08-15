# PyTorch
import torch
import torch.nn as nn

# Blocks
from .repos.models.blocks import (
    ConformerBlock
)

# Modules
from .repos.models.modules import (
    AudioPreprocessing,
    SpecAugment,
    Conv1dSubsampling,
    Conv2dSubsampling,
    Conv2dPoolSubsampling,
    VGGSubsampling
)

# Positional Encodings and Masks
from .repos.models.attentions import (
    SinusoidalPositionalEncoding,
    StreamingMask
)


class SEQFormerEncoder(nn.Module):
    def __init__(self, params):
        super(SEQFormerEncoder, self).__init__()

        # Audio Preprocessing
        self.preprocessing = AudioPreprocessing(params["sample_rate"], params["n_fft"], params["win_length_ms"], params["hop_length_ms"], params["n_mels"], params["normalize"], params["mean"], params["std"])
        
        # Spec Augment
        self.augment = SpecAugment(params["spec_augment"], params["mF"], params["F"], params["mT"], params["pS"])

        # Subsampling Module
        if params["subsampling_module"] == "Conv1d":
            self.subsampling_module = Conv1dSubsampling(params["subsampling_layers"], params["n_mels"], params["subsampling_filters"], params["subsampling_kernel_size"], params["subsampling_norm"], params["subsampling_act"])
        elif params["subsampling_module"] == "Conv2d":
            self.subsampling_module = Conv2dSubsampling(params["subsampling_layers"], params["subsampling_filters"], params["subsampling_kernel_size"], params["subsampling_norm"], params["subsampling_act"])
        elif params["subsampling_module"] == "Conv2dPool":
            self.subsampling_module = Conv2dPoolSubsampling(params["subsampling_layers"], params["subsampling_filters"], params["subsampling_kernel_size"], params["subsampling_norm"], params["subsampling_act"])
        elif params["subsampling_module"] == "VGG":
            self.subsampling_module = VGGSubsampling(params["subsampling_layers"], params["subsampling_filters"], params["subsampling_kernel_size"], params["subsampling_norm"], params["subsampling_act"])
        else:
            raise Exception("Unknown subsampling module:", params["subsampling_module"])
        

        # Padding Mask
        self.padding_mask = StreamingMask(left_context=params.get("left_context", params["max_pos_encoding"]), right_context=0 if params.get("causal", False) else params.get("right_context", params["max_pos_encoding"]))

        # Linear Proj
        self.linear = nn.Linear(params["subsampling_filters"][-1] * params["n_mels"] // 2**params["subsampling_layers"], params["dim_model"][0] if isinstance(params["dim_model"], list) else  params["dim_model"])

        # Dropout
        self.dropout = nn.Dropout(p=params["Pdrop"])

        # Sinusoidal Positional Encodings
        self.pos_enc = None if params["relative_pos_enc"] else SinusoidalPositionalEncoding(params["max_pos_encoding"], params["dim_model"][0] if isinstance(params["dim_model"], list) else  params["dim_model"])


        self.stage_1n2 = self.blocks = nn.ModuleList([ConformerBlock(
            dim_model=params["dim_model"][(block_id > torch.tensor(params.get("expand_blocks", []))).sum()] if isinstance(params["dim_model"], list) else params["dim_model"],
            dim_expand=params["dim_model"][(block_id >= torch.tensor(params.get("expand_blocks", []))).sum()] if isinstance(params["dim_model"], list) else params["dim_model"],
            ff_ratio=params["ff_ratio"],
            num_heads=params["num_heads"][(block_id > torch.tensor(params.get("expand_blocks", []))).sum()] if isinstance(params["num_heads"], list) else params["num_heads"], 
            kernel_size=params["kernel_size"][(block_id >= torch.tensor(params.get("expand_blocks", []))).sum()] if isinstance(params["kernel_size"], list) else params["kernel_size"], 
            att_group_size=params["att_group_size"][(block_id > torch.tensor(params.get("strided_blocks", []))).sum()] if isinstance(params.get("att_group_size", 1), list) else params.get("att_group_size", 1),
            att_kernel_size=params["att_kernel_size"][(block_id > torch.tensor(params.get("strided_layers", []))).sum()] if isinstance(params.get("att_kernel_size", None), list) else params.get("att_kernel_size", None),
            linear_att=params.get("linear_att", False),
            Pdrop=params["Pdrop"], 
            relative_pos_enc=params["relative_pos_enc"], 
            max_pos_encoding=params["max_pos_encoding"] // params.get("stride", 2)**int((block_id > torch.tensor(params.get("strided_blocks", []))).sum()),
            conv_stride=(params["conv_stride"][(block_id > torch.tensor(params.get("strided_blocks", []))).sum()] if isinstance(params["conv_stride"], list) else params["conv_stride"]) if block_id in params.get("strided_blocks", []) else 1,
            att_stride=(params["att_stride"][(block_id > torch.tensor(params.get("strided_blocks", []))).sum()] if isinstance(params["att_stride"], list) else params["att_stride"]) if block_id in params.get("strided_blocks", []) else 1,
            causal=params.get("causal", False)
        ) for block_id in range(params["num_blocks"])])

    
    def forward(self, x, x_len=None):

        # Audio Preprocessing
        x, x_len = self.preprocessing(x, x_len)

        # Spec Augment
        if self.training:
            x = self.augment(x, x_len)

        # Subsampling Module
        x, x_len = self.subsampling_module(x, x_len)

        # Padding Mask
        mask = self.padding_mask(x, x_len)

        # Transpose (B, D, T) -> (B, T, D)
        x = x.transpose(1, 2)

        # Linear Projection
        x = self.linear(x)

        # Dropout
        x = self.dropout(x)

        # Sinusoidal Positional Encodings
        if self.pos_enc is not None:
            x = x + self.pos_enc(x.size(0), x.size(1))

        # stage 1 and 2  
        attentions = []
        for block in self.stage_1n2:
            x, attention, hidden = block(x, mask)
            attentions.append(attention)

            # Strided Block
            if block.stride > 1:

                # Stride Mask (B, 1, T // S, T // S)
                if mask is not None:
                    mask = mask[:, :, ::block.stride, ::block.stride]

                # Update Seq Lengths
                if x_len is not None:
                    x_len = torch.div(x_len - 1, block.stride, rounding_mode='floor') + 1

        return x, x_len, attentions
    
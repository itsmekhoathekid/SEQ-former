# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        # stage 2.5 
        # self.s25_ln = nn.Linear(params["dim_model"][1] if isinstance(params["dim_model"], list) else params["dim_model"], params["vocab_size"])
        self.s25_ln = nn.Linear(params["dim_model"][1] if isinstance(params["dim_model"], list) else params["dim_model"], 4866)
        
        # Stage 3
        self.stage_3 = ConformerBlock(
            dim_model=params["dim_model"][1] if isinstance(params["dim_model"], list) else params["dim_model"],
            dim_expand=params["dim_model"][2] if isinstance(params["dim_model"], list) else params["dim_model"],
            ff_ratio=params["ff_ratio"],
            num_heads=params["num_heads"][2] if isinstance(params["num_heads"], list) else params["num_heads"],
            kernel_size=params["kernel_size"][2] if isinstance(params["kernel_size"], list) else params["kernel_size"],
            att_group_size=params["att_group_size"][2] if isinstance(params.get("att_group_size", 1), list) else params["att_group_size"],
            att_kernel_size=params["att_kernel_size"][2] if isinstance(params.get("att_kernel_size", None), list) else params.get("att_kernel_size", None),
            linear_att=params.get("linear_att", False),
            Pdrop=params["Pdrop"], 
            relative_pos_enc=params["relative_pos_enc"], 
            max_pos_encoding=params["max_pos_encoding"] // params.get("stride", 2),
            conv_stride=(params["conv_stride"][2] if isinstance(params["conv_stride"], list) else params["conv_stride"]) if 2 in params.get("strided_blocks", []) else 1,
            att_stride= (params["att_stride"][2] if isinstance(params["att_stride"], list) else params["att_stride"]) if 2 in params.get("strided_blocks", []) else 1,
            causal=params.get("causal", False)
        )    
    
    def keyframe_chunk_mask(
            self,
            peaks_masks: torch.Tensor,
            xs_lens: torch.Tensor,
            left = 2,
            right = 3 
    ) -> torch.Tensor:
        B, T = peaks_masks.size()
        # compute peaks index
        def generate_peak_index(b):
            peak = []
            for t in range(T):
                if not peaks_masks[b, t]: continue
                if t == 0: peak.append(t)
                elif peaks_masks[b, t-1] != peaks_masks[b, t]: peak.append(t)
            if len(peak) == 0: peak = [1]
            return peak
        # rewrite in 10.l1
        peaks_index = [generate_peak_index(b) for b in range(B)]
 
        def compute_peak_mask(i):
            peaks = peaks_index[i]
            lens = xs_lens[i]
            re = torch.zeros(T, T) != 0
            if len(peaks) == 0 : return re == False

            chunks = [[max(p - left, 0), min(p + right, lens)]  for p in peaks]
            for ch in range(1, len(chunks)):
                if chunks[ch][0] < chunks[ch-1][1]:
                    mid = (chunks[ch][0] + chunks[ch-1][1] + 1) // 2
                    chunks[ch-1][1] = mid
                    chunks[ch][0] = mid

            for ch in chunks:
                re[ch[0]:ch[1], ch[0]:ch[1]] = True
                re[ch[0]:ch[1], peaks] = True
            return re
        ret = [compute_peak_mask(i) for i in range(B)]
        ret = torch.stack(ret, dim=0)

        return ret
    
    def forward(self, x, x_len=None, targets=None, target_len=None):

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
        # return x, attention, hidden
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

        # stage 2.5
        # phải truyền vocab size vào để tính log_softmax
        # x = (B, T, D) -> (B, T, vocab_size)
        x_25 = self.s25_ln(x)
        log_probs = F.log_softmax(x_25, dim= -1)
        preds_25 = torch.argmax(log_probs, dim=-1)

        peaks_masks = preds_25.detach()

        blank_id = 0   
        # non_blank = (preds_25 != blank_id)
        # shifted = F.pad(preds_25[:, :-1], (1,0), value=-1)
        # is_new = (preds_25 != shifted)
        # peaks_masks = non_blank & is_new   # (B, T) bool

        # calculate keyframe chunk mask
        kfsa_mat = self.keyframe_chunk_mask(peaks_masks, x_len)  # (B, T, T)
        kfsa_mat = kfsa_mat.unsqueeze(1)  # (B, 1, T, T)

        combined_mask = mask.bool() & kfsa_mat  # vẫn shape (B, 1, T, T)

        # calc inter ctc spike reduce loss
        if self.training and targets is not None:
            log_probs_ctc = log_probs.permute(1, 0, 2)
            ctc_loss_fn = nn.CTCLoss(blank=blank_id, reduction="mean", zero_infinity=True)
            loss_inter = ctc_loss_fn(
                log_probs_ctc,    # (T, B, V)
                targets,          
                input_lengths=x_len,   
                target_lengths=target_len  
            )

        # stage 3
        x, attention3, hidden3 = self.stage_3(x, combined_mask)
        attentions.append(attention3)

        return x, x_len, attentions


def get_dummy_params():
    params = {
        "arch": "Conformer",
        "num_blocks": 15,
        "dim_model": [100, 140, 160],
        "ff_ratio": 4,
        "num_heads": 4,
        "kernel_size": 15,
        "Pdrop": 0.1,
        "conv_stride": 2,
        "att_stride": 1,
        "strided_blocks": [2],
        "expand_blocks": [2],
        "att_group_size": [3, 1, 1],

        "relative_pos_enc": True,
        "max_pos_encoding": 10000,

        "subsampling_module": "Conv2d",
        "subsampling_layers": 1,
        "subsampling_filters": [100],
        "subsampling_kernel_size": 3,
        "subsampling_norm": "batch",
        "subsampling_act": "swish",

        "sample_rate": 16000,
        "win_length_ms": 25,
        "hop_length_ms": 10,
        "n_fft": 512,
        "n_mels": 80,
        "normalize": False,
        "mean": -5.6501,
        "std": 4.2280,

        "spec_augment": True,
        "mF": 2,
        "F": 27,
        "mT": 10,
        "pS": 0.05
    }
    return params

if __name__ == "__main__":
    params = get_dummy_params()
    encoder = SEQFormerEncoder(params)
    
    # Giả sử đầu vào là waveform có batch_size=2, độ dài 16000 mẫu (sample)
    batch_size = 2
    signal_length = 16000
    x = torch.randn(batch_size, signal_length)
    x_len = torch.full((batch_size,), signal_length, dtype=torch.int32)
    
    # Chạy forward
    x_out, x_out_len, attentions = encoder(x, x_len)
    
    print("Output shape:", x_out.shape)
    print("Output lengths:", x_out_len)
    print("Number of attention maps:", len(attentions))
    
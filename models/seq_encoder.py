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
        
        # Stage 3
        self.stage_3 = ConformerBlock(
            dim_model=params["dim_model"][2] if isinstance(params["dim_model"], list) else params["dim_model"],
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

    # def compute_atten_mask_from_ctc_loss(self, x, target_len, targets=None):
    #     # Tính CTC logits
    #     ctc_logits = self.ctc_projection(x)  # (B, T, vocab_size)
        
    #     if targets is not None:
    #         # Tính CTC loss
    #         log_probs = F.log_softmax(ctc_logits, dim=-1)
    #         ctc_loss = F.ctc_loss(
    #             log_probs.transpose(0, 1),  # (T, B, vocab_size)
    #             targets,
    #             input_lengths=torch.full((x.size(0),), x.size(1), dtype=torch.long),
    #             target_lengths=target_len
    #         )
            
    #         # Tính attention mask dựa trên confidence score
    #         confidence = F.softmax(ctc_logits, dim=-1).max(dim=-1)[0]  # (B, T)
    #         attention_mask = (confidence > 0.5).float()  # threshold có thể điều chỉnh
            
    #         return attention_mask, ctc_loss
    #     else:
    #         # Inference mode - chỉ tính attention mask
    #         confidence = F.softmax(ctc_logits, dim=-1).max(dim=-1)[0]
    #         attention_mask = (confidence > 0.5).float()
    #         return attention_mask, None    

    def compute_log_softmax(ctc_logits, d_in, vocab_size):
        linear_layer = nn.Linear(d_in, vocab_size)
        log_probs = F.log_softmax(linear_layer(ctc_logits), dim=-1)

        return log_probs 
            
    def  compute_inter_CTC_attn_mask(self, logits, n, m, causal, threshold=0.5, base_mask=None):
        B, T, V = logits.size()

        # Check active frames  
        conf, _ = logits.max(dim=-1)  # (B, T)
        active = conf > threshold  # (B, T)

        w = torch.zeros(active, dtype=torch.float32, device=logits.device)  # (B, T)
        # output là (B, T)
        for i in range(-n, m+1): 
            src_from = max(0, -i)
            src_to   = T - max(0, i)
            if src_to > src_from:
                w[:, src_from:src_to] |= active[:, src_from + i: src_to + i]

        out_mask = torch.zeros((B, 1, T, T), dtype=torch.bool, device=logits.device) # (B, 1, T, T)

        def enable_block(mask_b, a, b):
            # clamp
            a = max(0, a); b = min(T - 1, b)
            if a <= b:
                mask_b[:, a:b+1, a:b+1] = True

        for b in range(B):
            # duyệt qua từng batch, lấy vector boolean từ w đã xử lý, nếu full false thì bỏ qua để không phải tính toán
            vec = w[b]  # (T,)
            if not vec.any():
                continue

            # edges of True segments
            on = vec.int() # true false to int 
            diff = torch.diff(torch.cat([torch.tensor([0], device=logits.device), on, torch.tensor([0], device=logits.device)]))
            # starts where diff==+1, ends where diff==-1 then -1 for inclusive index
            starts = (diff == 1).nonzero(as_tuple=False).flatten()
            ends   = (diff == -1).nonzero(as_tuple=False).flatten() - 1
            # build rectangles
            mb = out_mask[b:b+1]  # (1,1,T,T)
            for s, e in zip(starts.tolist(), ends.tolist()):
                enable_block(mb, s, e)

        if causal:
                tri = torch.tril(torch.ones((T, T), dtype=torch.bool, device=logits.device))
                out_mask = out_mask & tri.view(1, 1, T, T)

        if base_mask is not None:
            out_mask = out_mask & base_mask

        return out_mask
    
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
        # x = (B, T, D)
        log_probs = self.compute_log_softmax(x, x.size(-1), 100000)  # Giả sử vocab_size là 100000
        stage_25_mask = self.compute_inter_CTC_attn_mask(log_probs, n=2, m=2, causal=True, threshold=0.5, base_mask=mask)

        # stage 3
        stage3_mask = 0 # để tạm
        x, attention3, hidden3 = self.stage_3(x, stage3_mask)
        attentions.append(attention3)

        return x, x_len, attentions
    
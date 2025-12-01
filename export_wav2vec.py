import torch
import fairseq
import math
import sys
import onnx 

# =========================================================
# 1. MONKEY PATCH
# =========================================================
def patched_pad_to_multiple(x, multiple, dim=-1, value=0):
    if dim < 0:
        dim += x.ndim
    tsz = x.size(dim)
    m = tsz / multiple
    
    if isinstance(m, torch.Tensor):
        remainder = math.ceil(m.item()) * multiple - tsz
    else:
        remainder = math.ceil(m) * multiple - tsz
        
    if remainder == 0:
        return x, 0
    
    pad = [0] * (x.ndim * 2)
    pad[(x.ndim - 1 - dim) * 2 + 1] = remainder
    return torch.nn.functional.pad(x, pad, value=value), remainder

import fairseq.models.wav2vec.utils
import fairseq.models.wav2vec.wav2vec2
fairseq.models.wav2vec.utils.pad_to_multiple = patched_pad_to_multiple
fairseq.models.wav2vec.wav2vec2.pad_to_multiple = patched_pad_to_multiple

# =========================================================
# 2. LOAD & EXPORT
# =========================================================
print("Loading Model...")
model_path = "./model/SPRING_INX_ccc_wav2vec2_SSL.pt"
model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
wav2vec_model = model[0]
wav2vec_model.eval()
wav2vec_model.cpu()

class Wav2VecWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, source, padding_mask):
        res = self.model(source, padding_mask=padding_mask, mask=False, features_only=True)
        return res['x']

wrapped_model = Wav2VecWrapper(wav2vec_model)

# USE 3 SECONDS DUMMY INPUT (Just to be safe during trace)
dummy_input = torch.randn(1, 48000) 
dummy_mask = torch.zeros(1, 48000, dtype=torch.bool)

print("Exporting to ONNX...")
output_file = "ccc_wav2vec.onnx"

torch.onnx.export(
    wrapped_model,
    (dummy_input, dummy_mask),
    output_file,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['input_values', 'attention_mask'],
    output_names=['hidden_features'],
    # We attempt dynamic axes, but we will force it below anyway
    dynamic_axes={
        'input_values': {1: 'time'},
        'attention_mask': {1: 'time'},
        'hidden_features': {1: 'time'}
    }
)

# =========================================================
# 3. FORCE DYNAMIC SHAPES (The Fix)
# =========================================================
print("Patching ONNX shapes...")
model = onnx.load(output_file)

# Force Input 0 (Audio) to be dynamic
model.graph.input[0].type.tensor_type.shape.dim[1].dim_param = 'time'

# Force Input 1 (Mask) to be dynamic (if it exists)
if len(model.graph.input) > 1:
    model.graph.input[1].type.tensor_type.shape.dim[1].dim_param = 'time'

# Force Output (Features) to be dynamic
# The output time is roughly input_time / 320, but let's just call it 'out_time'
model.graph.output[0].type.tensor_type.shape.dim[1].dim_param = 'out_time'

onnx.save(model, output_file)
print(f"Success! Model saved to {output_file} (Dynamic Shapes Enforced)")
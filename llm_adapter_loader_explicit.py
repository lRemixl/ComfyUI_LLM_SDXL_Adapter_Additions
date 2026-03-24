import torch
from safetensors.torch import load_file
import logging
import gc
import os
import re
from .utils import get_llm_adapters, get_llm_adapter_path
from .llm_to_sdxl_adapter_explicit import LLMToSDXLAdapter
logger = logging.getLogger("LLM-SDXL-Adapter-Additions")


def convert_explicit_adapter_to_mha(state_dict):
    """
    Converts a full Adapter checkpoint trained with explicit attention (separate q, k, v Linear layers)
    to the standard nn.MultiheadAttention format (combined in_proj_weight/bias).
    
    This handles:
    1. Concatenating q_proj, k_proj, v_proj -> in_proj
    2. Renaming o_proj -> out_proj (common naming difference)
    """
    converted_dict = {}
    mha_buffers = {}
    
    # Regex to capture:
    # Group 1: Base path (e.g. "wide_attention_blocks.0.attn")
    # Group 2: Projection type (q, k, v)
    # Group 3: Parameter type (weight, bias)
    # Example match: wide_attention_blocks.0.attn.q_proj.weight
    qkv_pattern = re.compile(r"(.*)\.(q|k|v)_proj\.(weight|bias)")
    
    # Regex for Output projection renaming (if training used 'o_proj' but inference needs 'out_proj')
    # nn.MultiheadAttention uses 'out_proj', custom blocks often use 'o_proj'
    out_proj_pattern = re.compile(r"(.*)\.o_proj\.(weight|bias)")

    keys_to_remove = []

    # Pass 1: Identify QKV tensors and copy everything else
    for key, value in state_dict.items():
        qkv_match = qkv_pattern.match(key)
        out_match = out_proj_pattern.match(key)

        if qkv_match:
            base_path, proj_type, param_type = qkv_match.groups()
            
            if base_path not in mha_buffers:
                mha_buffers[base_path] = {'weight': {}, 'bias': {}}
            
            mha_buffers[base_path][param_type][proj_type] = value
        
        elif out_match:
            # Rename o_proj -> out_proj for standard nn.MultiheadAttention compatibility
            base_path, param_type = out_match.groups()
            new_key = f"{base_path}.out_proj.{param_type}"
            converted_dict[new_key] = value
        
        else:
            # Copy all other layers (norms, embeddings, MLPs) as-is
            converted_dict[key] = value

    # Pass 2: Process the grouped QKV tensors
    count_converted = 0
    for base_path, params in mha_buffers.items():
        # Handle Weights
        if all(k in params['weight'] for k in ['q', 'k', 'v']):
            # nn.MultiheadAttention expects shape (3 * embed_dim, embed_dim)
            # We concat [Q_weight, K_weight, V_weight] along dim 0
            combined_weight = torch.cat([
                params['weight']['q'],
                params['weight']['k'],
                params['weight']['v']
            ], dim=0)
            
            converted_dict[f"{base_path}.in_proj_weight"] = combined_weight
            count_converted += 1

        # Handle Biases (if they exist)
        if all(k in params['bias'] for k in ['q', 'k', 'v']):
            # nn.MultiheadAttention expects shape (3 * embed_dim)
            combined_bias = torch.cat([
                params['bias']['q'],
                params['bias']['k'],
                params['bias']['v']
            ], dim=0)
            
            converted_dict[f"{base_path}.in_proj_bias"] = combined_bias

    logger.info(f"Converted {count_converted} Attention blocks from explicit Linear layers to MultiheadAttention.")
    return converted_dict

class LLMAdapterLoaderUnfused:
    """
    ComfyUI node that loads LLM to SDXL adapter
    """
    
    def __init__(self):
        self.adapter = None
        self.current_adapter_path = None
        self.current_adapter_type = None
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    @classmethod
    def INPUT_TYPES(cls):
        adapter_types = ["gemma", "t5gemma"]
        return {
            "required": {
                "adapter_name": (get_llm_adapters(), {
                    "default": get_llm_adapters()[0] if get_llm_adapters() else None
                }),
                "type": (adapter_types, {"default": "gemma"}),
            },
            "optional": {
                "device": (["auto", "cuda:0", "cuda:1", "cpu"], {"default": "auto"}),
                "force_reload": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("LLM_ADAPTER", "STRING")
    RETURN_NAMES = ("llm_adapter", "info")
    FUNCTION = "load_adapter"
    CATEGORY = "llm_sdxl/additions"
    
    def load_adapter(self, adapter_name, type, device="auto", force_reload=False, expilict_attention=False, use_norm=False):
        """Load and initialize the LLM to SDXL adapter"""
        if device == "auto":
            device = self.device
        
        adapter_path = get_llm_adapter_path(adapter_name)
        
        # Adapter configuration presets per type
        ADAPTER_PRESETS = {
            "gemma": {
                "llm_dim": 1152,
                "sdxl_seq_dim": 2048,
                "sdxl_pooled_dim": 1280,
                "target_seq_len": 308,
                "n_wide_blocks": 2,
                "n_narrow_blocks": 3,
                "num_heads": 16,
                "dropout": 0.1,
            },
            "t5gemma": {
                "llm_dim": 2304,
                "sdxl_seq_dim": 2048,
                "sdxl_pooled_dim": 1280,
                "target_seq_len": 308,
                "n_wide_blocks": 3,
                "n_narrow_blocks": 3,
                "num_heads": 16,
                "dropout": 0.0,
            },
        }
        
        if type not in ADAPTER_PRESETS:
            raise ValueError(f"Unknown adapter type: {type}")
        config = ADAPTER_PRESETS[type]
        
        try:
            # Check if we need to reload
            if force_reload or self.adapter is None or self.current_adapter_path != adapter_path or self.current_adapter_type != type:
                # Clear previous adapter
                if self.adapter is not None:
                    del self.adapter
                    gc.collect()
                    torch.cuda.empty_cache()
                
                logger.info(f"Loading LLM to SDXL adapter from {adapter_path}")
                
                # Initialize adapter with specified parameters
                self.adapter = LLMToSDXLAdapter(
                    llm_dim=config["llm_dim"],
                    sdxl_seq_dim=config["sdxl_seq_dim"],
                    sdxl_pooled_dim=config["sdxl_pooled_dim"],
                    target_seq_len=config["target_seq_len"],
                    n_wide_blocks=config["n_wide_blocks"],
                    n_narrow_blocks=config["n_narrow_blocks"],
                    num_heads=config["num_heads"],
                    dropout=config["dropout"],
                    use_norm=use_norm,
                )
                
                # Load checkpoint if file exists
                strict_load = True
                if use_norm:
                    strict_load = False
                if os.path.exists(adapter_path):
                    checkpoint = load_file(adapter_path)
                    if expilict_attention:
                        pass
                        # checkpoint = convert_explicit_adapter_to_mha(checkpoint)
                    self.adapter.load_state_dict(checkpoint,strict=strict_load)
                    logger.info(f"Loaded adapter weights from {adapter_path}")
                else:
                    logger.warning(f"Adapter file not found: {adapter_path}, using initialized weights")
                
                # Move to device
                self.adapter.to(device)
                self.adapter.eval()
                
                self.current_adapter_path = adapter_path
                self.current_adapter_type = type
                logger.info("LLM to SDXL adapter loaded successfully")
            
            info = (
                f"Adapter: {adapter_path}\n"
                f"Type: {type}\n"
                f"Device: {device}\n"
                f"LLM dim: {config['llm_dim']}\n"
                f"SDXL seq dim: {config['sdxl_seq_dim']}\n"
                f"Target seq len: {config['target_seq_len']}"
            )
            
            return (self.adapter, info)
            
        except Exception as e:
            logger.error(f"Failed to load adapter: {str(e)}")
            raise Exception(f"Adapter loading failed: {str(e)}")



# Node mapping for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "LLMAdapterLoaderUnfused": LLMAdapterLoaderUnfused,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMAdapterLoaderUnfused": "LLM Adapter Loader Unfused",
} 
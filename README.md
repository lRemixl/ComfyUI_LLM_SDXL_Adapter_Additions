# ComfyUI_LLM_SDXL_Adapter_Additions
Additional nodes for use alongside [ComfyUI_LLM_SDXL_Adapter](https://github.com/NeuroSenko/ComfyUI_LLM_SDXL_Adapter/) 

## t5gemma_text_encode_v2.py 
**Node:** T5GEMMATextEncoder++

**Purpose:** Modified from the devlop branch of [ComfyUI_LLM_SDXL_Adapter](https://github.com/NeuroSenko/ComfyUI_LLM_SDXL_Adapter/tree/develop). Adds **chunking** for prompts greater than 512 tokens. Also significantly simplifies workflow. 

## llm_adapter_loader_explicit.py 
**Node:** LLMAdapterLoaderUnfused

**Purpose:** Load T5 Gemma adapter that has unfused QKV. Compared to the regular [version](https://huggingface.co/Minthy/Rouwei-T5Gemma-adapter_v0.2/) that has them fused. Allows for slightly better quality when loading adapters that have been trained with unfused QKVs.  

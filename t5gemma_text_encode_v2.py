import torch
import logging

logger = logging.getLogger("LLM-SDXL-Adapter-Additions")
# Modified NeuroSenko/ComfyUI_LLM_SDXL_Adapter develop branch
class T5GEMMATextEncoderAdd:
    """
    Simplified ComfyUI node that combines text encoding and adapter application.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "llm_tokenizer": ("LLM_TOKENIZER",),
                "llm_adapter": ("LLM_ADAPTER",),
                "text": ("STRING", {"multiline": True, "default": "masterpiece, best quality"}),
                "chunking": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("CONDITIONING", "info")
    FUNCTION = "encode"
    CATEGORY = "llm_sdxl/additions"

    def get_token_data(self, tokenizer, text, max_length, device, chunking):
        """
        Tokenizes text completely to calculate stats, dynamically applies padding/chunking,
        and generates an info string with token lengths and context boundaries.
        """
        # Tokenize without padding or truncation to get the absolute length and accurate offsets
        inputs = tokenizer(
            text + "<eos>",
            return_tensors="pt",
            padding=False,
            truncation=False,
            return_offsets_mapping=True
        )
        
        input_ids = inputs.input_ids[0].to(device)
        attention_mask = inputs.attention_mask[0].to(device)
        offset_mapping = inputs.offset_mapping[0]

        seq_len = input_ids.shape[0]
        total_tokens = seq_len
        
        # info string
        info_str = f"Total Tokens: {total_tokens}\n"
        if total_tokens > max_length:
            # 512th token is at index 511
            start_char = int(offset_mapping[max_length - 1][0].item())
            end_char = int(offset_mapping[max_length - 1][1].item())
            
            word = text[start_char:end_char]
            # Extract +/- 25 chars around the 512 token for context
            ctx_start = max(0, start_char - 25)
            ctx_end = min(len(text), end_char + 25)
            context = text[ctx_start:ctx_end].replace("\n", " ")
            
            info_str += f"\nExceeds {max_length} tokens!\n"
            if not chunking:
                info_str += "CHUNKING IS FALSE: Text after token 512 will be IGNORED.\n"
            info_str += f"Token {max_length} is roughly at character {start_char} (Word: '{word}')\n"
            info_str += f"Context: \"... {context} ...\""
        else:
            info_str += f"Under limit ({max_length})."

        # Handle Pad Token
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
            if isinstance(pad_token_id, list): 
                pad_token_id = pad_token_id[0]
        if pad_token_id is None:
            pad_token_id = 0

        # Chunking or Truncation
        if not chunking:
            # Truncate if sequence length > max_length
            if seq_len > max_length:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
                seq_len = max_length

            # Pad if under max_length
            if seq_len < max_length:
                pad_len = max_length - seq_len
                pad_ids = torch.full((pad_len,), pad_token_id, dtype=input_ids.dtype, device=device)
                pad_mask = torch.zeros((pad_len,), dtype=attention_mask.dtype, device=device)
                
                input_ids = torch.cat([input_ids, pad_ids])
                attention_mask = torch.cat([attention_mask, pad_mask])
                
            return [input_ids.unsqueeze(0)], [attention_mask.unsqueeze(0)], info_str

        # If chunking: slice into blocks of `max_length` and pad the final chunk
        chunked_input_ids = []
        chunked_attention_masks = []

        for i in range(0, seq_len, max_length):
            chunk_ids = input_ids[i:i+max_length]
            chunk_mask = attention_mask[i:i+max_length]
            
            # Pad if the final chunk is smaller than max_length
            if len(chunk_ids) < max_length:
                pad_len = max_length - len(chunk_ids)
                
                pad_ids = torch.full((pad_len,), pad_token_id, dtype=chunk_ids.dtype, device=device)
                pad_mask = torch.zeros((pad_len,), dtype=chunk_mask.dtype, device=device)
                
                chunk_ids = torch.cat([chunk_ids, pad_ids])
                chunk_mask = torch.cat([chunk_mask, pad_mask])
                
            chunked_input_ids.append(chunk_ids.unsqueeze(0))
            chunked_attention_masks.append(chunk_mask.unsqueeze(0))
            
        return chunked_input_ids, chunked_attention_masks, info_str

    def encode(self, llm_model, llm_tokenizer, llm_adapter, text, chunking):
        try:
            max_length = 512
            
            # Try to grab the exact device the model weights are on, fallback to cuda
            try:
                device = next(llm_model.parameters()).device
            except Exception:
                device = "cuda"

            c_input_ids, c_attention_masks, info_str = self.get_token_data(
                llm_tokenizer, text, max_length, device, chunking
            )

            all_prompt_embeds = []
            first_pooled_output = None

            with torch.no_grad():
                # Process each 512-token chunk independently
                for input_ids, attention_mask in zip(c_input_ids, c_attention_masks):
                    
                    base_out = llm_model(input_ids=input_ids, attention_mask=attention_mask)
                    final_hidden_states = base_out.last_hidden_state.to(torch.float32)

                    # Apply Adapter
                    prompt_embeds, pooled_output = llm_adapter(final_hidden_states, attention_mask=attention_mask)
                    
                    # Accumulate arrays
                    all_prompt_embeds.append(prompt_embeds.cpu().contiguous())
                    if first_pooled_output is None:
                        first_pooled_output = pooled_output.cpu().contiguous()

            # Concatenate all prompt chunks together (SDXL combining style)
            final_prompt_embeds = torch.cat(all_prompt_embeds, dim=1)

            meta = {"pooled_output": first_pooled_output}
            conditioning = [[final_prompt_embeds, meta]]
            
            logger.info(f"Encoded text (Chunks: {len(all_prompt_embeds)}) to shape: {final_prompt_embeds.shape}")
            logger.info(f"Token Info:\n{info_str}")

            return (conditioning, info_str)

        except Exception as e:
            logger.error(f"Encoding error: {e}")
            raise e

NODE_CLASS_MAPPINGS = {
    "T5GEMMATextEncoder++": T5GEMMATextEncoderAdd
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "T5GEMMATextEncoder++": "T5Gemma Text Encode++"
}
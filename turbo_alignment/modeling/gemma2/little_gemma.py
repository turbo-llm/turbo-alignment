from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM


class Gemma2ForCausalLMUlysses(Gemma2ForCausalLM):
    @classmethod
    def _autoset_attn_implementation(cls, config, use_flash_attention_2 = False, torch_dtype = None, device_map = None, check_device_map = True):
        old = config._attn_implementation
        if old == 'flash_attention_2_ulysses':
            config._attn_implementation = 'flash_attention_2'

        res = super()._autoset_attn_implementation(config, use_flash_attention_2, torch_dtype, device_map, check_device_map)
        res._attn_implementation = old
        return res

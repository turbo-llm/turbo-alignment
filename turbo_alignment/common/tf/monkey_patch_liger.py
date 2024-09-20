from transformers import PretrainedConfig, PreTrainedModel

from turbo_alignment.common.tf.liger_kernel.cross_entropy import LigerCrossEntropyLoss
from turbo_alignment.common.tf.liger_kernel.geglu import LigerGEGLUMLP
from turbo_alignment.common.tf.liger_kernel.rope import liger_rotary_pos_emb


def apply_liger_kernel_to_gemma2(
    rope: bool = True,
    cross_entropy: bool = True,
    rms_norm: bool = True,
    geglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Gemma2
    (for Gemma1 please use `apply_liger_kernel_to_gemma`) to make GPU go burrr.

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is True.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        geglu (bool): Whether to apply Liger's GeGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """
    from transformers.models.gemma2 import modeling_gemma2

    if rope:
        modeling_gemma2.apply_rotary_pos_emb = liger_rotary_pos_emb
    if cross_entropy:
        modeling_gemma2.CrossEntropyLoss = LigerCrossEntropyLoss
    if geglu:
        modeling_gemma2.Gemma2MLP = LigerGEGLUMLP

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules
        config: PretrainedConfig = model.config

        if hasattr(model, "model"):
            # The case for Gemma2ForCausalLM, Gemma2ForTokenClassification for example
            base_model = model.model
        else:
            # Direct Gemma2Model
            base_model = model

        torch_dtype = config.torch_dtype

        for decoder_layer in base_model.layers:
            if geglu:
                decoder_layer.mlp = LigerGEGLUMLP(config).to(torch_dtype)

    print('🙈'*15)

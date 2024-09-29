from transformers import PretrainedConfig, PreTrainedModel

from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.modeling.liger_kernels.cross_entropy import LigerCrossEntropyLoss
from turbo_alignment.modeling.liger_kernels.geglu import LigerGEGLUMLP
from turbo_alignment.modeling.liger_kernels.rope import liger_rotary_pos_emb

logger = get_project_logger()


def apply_liger_kernel_to_gemma2(
    rope: bool = True,
    cross_entropy: bool = True,
    geglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    logger.info('Loading Liger-Kernels for Gemma2...')

    from transformers.models.gemma2 import modeling_gemma2

    if rope:
        modeling_gemma2.apply_rotary_pos_emb = liger_rotary_pos_emb
    if cross_entropy:
        modeling_gemma2.CrossEntropyLoss = LigerCrossEntropyLoss
    if geglu:
        modeling_gemma2.Gemma2MLP = LigerGEGLUMLP

    if model is not None:
        config: PretrainedConfig = model.config

        if hasattr(model, 'model'):
            base_model = model.model
        else:
            base_model = model

        torch_dtype = config.torch_dtype

        for decoder_layer in base_model.layers:
            if geglu:
                decoder_layer.mlp = LigerGEGLUMLP(config).to(torch_dtype)

    logger.info('Liger-Kernels have been successfully applied!')

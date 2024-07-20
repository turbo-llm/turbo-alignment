from turbo_alignment.settings.generators.outputs.chat import ChatInferenceOutput


class SamplingDatasetRecord(ChatInferenceOutput):
    rewards: dict[str, float] | None = None

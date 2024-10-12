from turbo_alignment.common.registry import Registrable
from turbo_alignment.settings.modality import Modality


class ModalityReaderRegistry(Registrable):
    ...


@ModalityReaderRegistry.register(Modality.AUDIO)
class AudioModalityReaderRegistry(Registrable):
    ...


@ModalityReaderRegistry.register(Modality.IMAGE)
class ImageModalityReaderRegistry(Registrable):
    ...

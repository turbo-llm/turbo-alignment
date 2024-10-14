# mypy: ignore-errors
import logging
import os
from pathlib import Path

import cached_path as _cached_path

logger = logging.getLogger(__name__)


def cached_path(
    url_or_filename: str | os.PathLike,
    cache_dir: str | Path = None,
    extract_archive: bool = False,
    force_extract: bool = False,
) -> str:
    return str(
        _cached_path.cached_path(
            url_or_filename,
            cache_dir=cache_dir,
            extract_archive=extract_archive,
            force_extract=force_extract,
        )
    )

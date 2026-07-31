"""Freshness bookkeeping for the datasets this engine rewrites.

One call, ``mark_refreshed()``, does the two things that have to happen together
when a dataset is rewritten:

1. stamps *when* it happened, durably, in Mongo — the backend serves it on ``GET /``;
2. drops the backend's cached responses for it, so the API stops serving the old
   data immediately instead of waiting out its TTL.

Infrastructure failures here are logged and swallowed: a Redis or Mongo hiccup must
never fail an extraction run. An unknown dataset name is a programming error and
does raise.
"""

from tipi_data.repositories.dataset_updates import DatasetUpdates

from qhld_engine.infrastructure import cache
from qhld_engine.infrastructure.config.settings import get_settings
from qhld_engine.logger import get_logger


log = get_logger(__name__)

DEPUTIES = 'deputies'
PARLIAMENTARY_GROUPS = 'parliamentary-groups'
INITIATIVES = 'initiatives'

# Not a dataset: the moment a whole extraction run finished. Recorded by the last
# step of the pipeline, so anything that stops the pipeline leaves it where it was.
EXTRACTION = 'extraction'


def _cache_keys(dataset):
    """The backend cache keys this dataset invalidates. Initiatives are served
    uncached, so they are stamped but flush nothing."""
    settings = get_settings()
    keys = {
        DEPUTIES: (settings.cache_deputies, settings.cache_deputies_compact),
        PARLIAMENTARY_GROUPS: (settings.cache_groups, settings.cache_groups_compact),
        INITIATIVES: (),
        EXTRACTION: (),
    }
    return keys[dataset]


def mark_refreshed(dataset):
    keys = _cache_keys(dataset)

    try:
        DatasetUpdates.touch(dataset)
    except Exception as e:
        log.warning(f'Cannot record when "{dataset}" was refreshed due "{e}"')

    if not keys:
        return
    try:
        dropped = cache.delete(*keys)
        log.info(f'Invalidated {dropped} cached response(s) for "{dataset}"')
    except Exception as e:
        log.warning(
            f'Cannot invalidate the cached responses for "{dataset}" due "{e}"; '
            'the API will serve stale data until they expire')

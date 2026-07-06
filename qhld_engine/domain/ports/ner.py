"""Port for named-entity recognition over speech text.

Deliberately minimal: mention extraction only needs the person spans a speech
names. The adapter returns every PER span *verbatim* (including honorifics like
"el señor Sánchez"); normalization and resolution to a canonical deputy are a
separate application/domain concern (``domain.speeches.mentions``), kept out of
the port so it stays a thin wrapper over whatever NER engine backs it.
"""

from typing import Protocol


class NerPort(Protocol):
    def person_spans(self, text: str) -> list[str]:
        """Return the text of every person (PER) entity found in ``text``, in
        order of appearance and with duplicates preserved (callers count them)."""
        ...

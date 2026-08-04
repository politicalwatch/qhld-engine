"""Materialise the frozen evaluation corpus as a canonical ``.jsonl`` plus its sha256.

This is what makes the thesis's `eval-XV-1` snapshot auditable: the figures in
chapter 4 are all measured against one corpus state, and a hash is what proves a
later re-extraction has not moved it underneath them.

**Canonical means byte-reproducible.** Documents are emitted sorted by ``_id``,
keys sorted within each document, no whitespace padding, UTF-8 left as-is. Running
this twice on an unchanged corpus must produce an identical file and therefore an
identical hash — otherwise the hash could not distinguish "the corpus changed" from
"the dump was written differently", which is the only thing it is for.

Two hashes are printed because they answer different questions: the file hash
detects any change at all, while the hash of the sorted ids separates "speeches
entered or left" from "some speeches' content changed".

The output does not belong in git (tens of MB). It lives beside the other corpus
dumps in ``qhld-backups/``; the artifact of record is the hash, not the file.

Usage:
    MONGO_HOST=localhost MONGO_PORT=27018 \\
      uv run --no-sync python scripts/dump_eval_corpus.py <salida.jsonl>

To verify a snapshot rather than create one, dump to a scratch path and compare
the printed hash against the one recorded in the vault's `Cifras canónicas` §1.2.
"""

import hashlib
import json
import os
import sys

from pymongo import MongoClient


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    out = sys.argv[1]

    client = MongoClient(
        host=os.environ.get("MONGO_HOST", "localhost"),
        port=int(os.environ.get("MONGO_PORT", 27018)),
        username=os.environ.get("MONGO_USER", "qhld"),
        password=os.environ.get("MONGO_PASSWORD", "qhld"),
        authSource="admin",
    )
    speeches = client[os.environ.get("MONGO_DB_NAME", "qhlddb")].speeches

    digest = hashlib.sha256()
    ids = []
    count = 0
    with open(out, "wb") as handle:
        for doc in speeches.find().sort("_id", 1):
            ids.append(str(doc["_id"]))
            line = json.dumps(doc, sort_keys=True, ensure_ascii=False,
                              separators=(",", ":"), default=str).encode("utf-8") + b"\n"
            handle.write(line)
            digest.update(line)
            count += 1

    id_digest = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    print(f"documentos      {count}")
    print(f"fichero         {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")
    print(f"sha256(fichero) {digest.hexdigest()}")
    print(f"sha256(ids)     {id_digest}")


if __name__ == "__main__":
    main()

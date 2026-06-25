"""Integration test for ``TagInitiatives.rename`` against throwaway MongoDB.

``rename`` is DB-pure: it finds initiatives carrying ``topic/old_tag`` and rewrites
that tag's name in place, leaving other tags untouched.
"""

import pytest

from tipi_data.models.initiative import Initiative, Tag, Tagged
from tipi_data.repositories.initiatives import Initiatives

from qhld_engine.tagger.tag_initiatives import TagInitiatives

pytestmark = pytest.mark.integration


def test_rename_updates_only_the_targeted_tag(mongo_db):
    tagged = Tagged(
        knowledgebase="kb1",
        topics=["A"],
        tags=[
            Tag(topic="A", subtopic="s", tag="oldname", times=2),
            Tag(topic="A", subtopic="s2", tag="keep", times=1),
        ],
    )
    Initiatives.save(Initiative(id="i1", reference="R1", tagged=[tagged]))

    TagInitiatives().rename("A", "oldname", "newname")

    kb = Initiatives.get("i1").tagged[0]
    assert {t.tag for t in kb.tags} == {"newname", "keep"}

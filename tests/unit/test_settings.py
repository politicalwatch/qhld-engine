"""Unit tests for the consolidated Pydantic ``Settings`` — no DB.

These pin the two latent bugs the migration fixed. The old plain-``env`` config
did ``bool(env.get('AMENDMENTS_FEATURE', False))``; since env vars are always
strings, ``bool("False")`` is ``True`` — the flag was a silent no-op. Pydantic's
``bool`` parser reads the value correctly. ``_env_file=None`` keeps these tests
from reading the repo-root ``.env``.
"""

import pytest

from qhld_engine.infrastructure.config.settings import Settings

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("raw", ["False", "false", "0", "no", "off"])
def test_amendments_feature_falsey_strings_are_false(raw):
    settings = Settings(_env_file=None, amendments_feature=raw)
    assert settings.amendments_feature is False


@pytest.mark.parametrize("raw", ["True", "true", "1", "yes", "on"])
def test_amendments_feature_truthy_strings_are_true(raw):
    settings = Settings(_env_file=None, amendments_feature=raw)
    assert settings.amendments_feature is True


def test_current_legislature_false_string_is_false():
    # The same bool() bug lived on CURRENT_LEGISLATURE; it just happened to be
    # harmless because .env set it to True. Pin it anyway.
    settings = Settings(_env_file=None, current_legislature="False")
    assert settings.current_legislature is False


def test_use_alerts_false_string_is_false():
    settings = Settings(_env_file=None, use_alerts="False")
    assert settings.use_alerts is False


def test_id_legislatura_coerced_to_int():
    settings = Settings(_env_file=None, id_legislatura="15")
    assert settings.id_legislatura == 15


def test_defaults(monkeypatch):
    # _env_file=None only ignores the .env *file*; pydantic-settings still reads
    # os.environ. The qhld-engine container injects these as real env vars
    # (ID_LEGISLATURA=15, etc.), so clear them to test the genuine defaults.
    for key in (
        "MODULE_EXTRACTOR",
        "ID_LEGISLATURA",
        "CURRENT_LEGISLATURE",
        "AMENDMENTS_FEATURE",
        "USE_ALERTS",
        "LIMIT_DATE_TO_SYNC",
        "LOGLEVEL",
        "LEGISLATURE_START_DATE",
        "LEGISLATURE_END_DATE",
    ):
        monkeypatch.delenv(key, raising=False)
    settings = Settings(_env_file=None)
    assert settings.module_extractor == "spain"
    assert settings.id_legislatura == 0
    assert settings.current_legislature is True
    assert settings.amendments_feature is False
    assert settings.use_alerts is False
    assert settings.limit_date_to_sync == "2000-01-01"
    assert settings.loglevel == "INFO"

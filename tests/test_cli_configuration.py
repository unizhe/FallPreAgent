from argparse import Namespace

import pytest

from fallpre.cli import run
from fallpre.config import SiteConfig


def test_explicit_webhook_configuration_cannot_silently_disappear(tmp_path, monkeypatch):
    site = tmp_path / "site.json"
    site.write_text(SiteConfig(image_size=(640, 480)).model_dump_json())
    monkeypatch.delenv("FALLPRE_TEST_MISSING_ENDPOINT", raising=False)
    args = Namespace(site=str(site), rules=None, camera=0, enable_actions=True, speech=True,
                     webhook_env="FALLPRE_TEST_MISSING_ENDPOINT")
    with pytest.raises(ValueError, match="webhook environment variable is empty"):
        run(args)


def test_real_actions_are_never_accepted_for_recorded_video(tmp_path):
    site = tmp_path / "site.json"
    site.write_text(SiteConfig(image_size=(640, 480)).model_dump_json())
    with pytest.raises(ValueError, match="Replay is always dry-run"):
        run(Namespace(site=str(site), rules=None, camera=None, enable_actions=True))

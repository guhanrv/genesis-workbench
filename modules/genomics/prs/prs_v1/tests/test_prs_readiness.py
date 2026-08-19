"""prs_readiness is import-light (Spark is injected)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
from prs_readiness import assert_scoring_ready


def test_assert_scoring_ready_is_callable():
    assert callable(assert_scoring_ready)

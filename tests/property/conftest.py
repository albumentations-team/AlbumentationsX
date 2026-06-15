"""Hypothesis profiles for property-based tests."""

from __future__ import annotations

from hypothesis import HealthCheck, settings

settings.register_profile(
    "local",
    max_examples=25,
    deadline=None,
    suppress_health_check=(HealthCheck.function_scoped_fixture,),
)
settings.register_profile("ci-fast", max_examples=10, deadline=None, derandomize=True)
settings.register_profile("ci-nightly", max_examples=75, deadline=None, derandomize=True)
settings.register_profile("release", max_examples=100, deadline=None, derandomize=True)

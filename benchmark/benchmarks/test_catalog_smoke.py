"""Catalog-wide transform smoke benchmarks."""

from __future__ import annotations

from benchmarks.catalog import asv_case_ids, benchmark_specs, make_compose, make_data


class TimeCatalogTransformSmoke:
    """Benchmark one valid Compose path for every runnable public transform."""

    params = (asv_case_ids(),)
    param_names = ("case_id",)

    def setup(self, case_id: str) -> None:
        spec = benchmark_specs()[case_id]
        self.transform = make_compose(spec)
        self.data = make_data(spec)

    def time_transform_compose(self, case_id: str) -> None:
        self.transform(**self.data)

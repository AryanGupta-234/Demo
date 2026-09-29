"""Bounded parallel orchestration for independent Change Requests.

Each worker owns one CR for the complete pipeline invocation. Evidence discovery
and inventory remain CR-scoped through run_pre_cab and its attachment root.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .pipeline import FinalPipelineResult, run_pre_cab


@dataclass(frozen=True)
class CRProcessingResult:
    """Result for one CR, preserving its input identity."""

    cr_number: str
    result: FinalPipelineResult | None = None
    error: str | None = None


def process_crs_parallel(
    crs: Iterable[dict[str, Any]],
    *,
    attachment_root: str | None = None,
    max_workers: int = 4,
    processor: Callable[..., FinalPipelineResult] = run_pre_cab,
    **kwargs: Any,
) -> list[CRProcessingResult]:
    """Process independent CRs concurrently with a bounded worker pool.

    The processor is invoked once per CR. attachment_root is shared only as
    the evidence root; run_pre_cab resolves <root>/<CR number> so a worker
    cannot intentionally consume another CR's workspace through normal discovery.
    """

    records = list(crs)
    if not records:
        return []

    workers = max(1, min(int(max_workers), len(records)))
    ordered: list[CRProcessingResult | None] = [None] * len(records)

    def run_one(index: int, cr: dict[str, Any]) -> CRProcessingResult:
        number = str(
            cr.get("Number")
            or cr.get("Effective number")
            or cr.get("change_number")
            or f"UNKNOWN_CR_{index}"
        ).strip()
        try:
            result = processor(
                cr,
                attachment_root=attachment_root,
                **kwargs,
            )
            return CRProcessingResult(cr_number=number, result=result)
        except Exception as exc:
            return CRProcessingResult(
                cr_number=number,
                error=f"{type(exc).__name__}: {exc}",
            )

    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="pre-cab-cr",
    ) as executor:
        futures: dict[Future[CRProcessingResult], int] = {
            executor.submit(run_one, index, cr): index
            for index, cr in enumerate(records)
        }
        for future in as_completed(futures):
            index = futures[future]
            ordered[index] = future.result()

    return [item for item in ordered if item is not None]

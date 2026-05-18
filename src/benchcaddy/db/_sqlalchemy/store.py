from __future__ import annotations

from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .models import BenchmarkRun, BenchmarkSuite, BenchmarkSuiteBaseline


def _get_suite(session: Session, suite_name: str) -> BenchmarkSuite | None:
    return session.scalar(select(BenchmarkSuite).where(BenchmarkSuite.name == suite_name))


def _get_or_create_suite(session: Session, suite_name: str, target_name: str) -> BenchmarkSuite:
    suite = _get_suite(session, suite_name)
    if suite is None:
        suite = BenchmarkSuite(name=suite_name, target_name=target_name)
        session.add(suite)
        try:
            session.flush()
        except IntegrityError:
            session.rollback()
            suite = _get_suite(session, suite_name)
            if suite is None:
                raise
    return suite


def _get_suite_baseline_record(session: Session, suite_id: int) -> BenchmarkSuiteBaseline | None:
    return session.scalar(select(BenchmarkSuiteBaseline).where(BenchmarkSuiteBaseline.suite_id == suite_id))


def _resolve_suite_baseline_run(session: Session, suite: BenchmarkSuite) -> BenchmarkRun | None:
    baseline_record = _get_suite_baseline_record(session, suite.id)
    if baseline_record is None:
        return None
    return session.get(BenchmarkRun, baseline_record.run_id)


def _list_all_suites(session: Session) -> list[BenchmarkSuite]:
    return session.execute(select(BenchmarkSuite).order_by(BenchmarkSuite.name)).scalars().all()


def _list_all_runs_latest_first(session: Session, limit: int | None = None) -> list[BenchmarkRun]:
    statement: Select[tuple[BenchmarkRun]] = select(BenchmarkRun).order_by(
        BenchmarkRun.sweep_execution_id.desc(),
        BenchmarkRun.run_index.desc(),
        BenchmarkRun.id.desc(),
    )
    if limit is not None:
        statement = statement.limit(limit)
    return session.execute(statement).scalars().all()


def _count_all_runs(session: Session) -> int:
    return int(session.scalar(select(func.count(BenchmarkRun.id))) or 0)


def _list_suite_runs_latest_first(session: Session, suite_id: int) -> list[BenchmarkRun]:
    return (
        session.execute(
            select(BenchmarkRun).where(BenchmarkRun.suite_id == suite_id).order_by(BenchmarkRun.sweep_execution_id.desc(), BenchmarkRun.run_index.desc(), BenchmarkRun.id.desc())
        )
        .scalars()
        .all()
    )


def _list_suite_runs_created_desc(session: Session, suite_id: int) -> list[BenchmarkRun]:
    return session.execute(select(BenchmarkRun).where(BenchmarkRun.suite_id == suite_id).order_by(BenchmarkRun.created_at.desc())).scalars().all()


def _list_suite_runs_oldest_first(session: Session, suite_id: int) -> list[BenchmarkRun]:
    return (
        session.execute(
            select(BenchmarkRun).where(BenchmarkRun.suite_id == suite_id).order_by(BenchmarkRun.sweep_execution_id.asc(), BenchmarkRun.run_index.asc(), BenchmarkRun.id.asc())
        )
        .scalars()
        .all()
    )


def _collect_observation_labels(observation_groups: list[list[dict[str, Any]]] | Any) -> list[str]:
    labels: set[str] = set()
    for observations in observation_groups:
        for sample in observations:
            for record in sample.get("records", []):
                labels.add(str(record.get("label")))
    return sorted(labels)


def _resolve_run(session: Session, run_id: int | tuple[int, int]) -> BenchmarkRun | None:
    if isinstance(run_id, int):
        return session.get(BenchmarkRun, run_id)

    sweep_id, run_index = run_id
    run = session.scalar(
        select(BenchmarkRun).where(
            BenchmarkRun.sweep_execution_id == sweep_id,
            BenchmarkRun.run_index == run_index,
        )
    )
    if run is None and run_index == 1:
        legacy_run = session.get(BenchmarkRun, sweep_id)
        if legacy_run is not None and legacy_run.sweep_execution_id is None and legacy_run.run_index in (None, 1):
            return legacy_run
    return run

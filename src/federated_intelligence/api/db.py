"""Database layer (SQLAlchemy 2.x). SQLite by default, PostgreSQL via ``DATABASE_URL``.

Only experiment metadata, summaries and per-round scalar metrics are stored - never tensors or
raw datasets.
"""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Iterator

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    pass


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC).replace(tzinfo=None)


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(200), primary_key=True)
    exp_group: Mapped[str] = mapped_column(String(40), index=True, default="")
    variant: Mapped[str] = mapped_column(String(200), default="")
    seed: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), index=True, default="queued")  # queued|running|finished|failed
    source: Mapped[str] = mapped_column(String(20), default="api")  # api | imported
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    summary: Mapped[dict] = mapped_column(JSON, default=dict)
    per_client: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str] = mapped_column(String(2000), default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    rounds: Mapped[list[RoundMetric]] = relationship(back_populates="experiment", cascade="all, delete-orphan")


class RoundMetric(Base):
    __tablename__ = "round_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id", ondelete="CASCADE"), index=True)
    round: Mapped[int] = mapped_column(Integer)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    experiment: Mapped[Experiment] = relationship(back_populates="rounds")


def database_url() -> str:
    return os.environ.get("DATABASE_URL", "sqlite:///./federated_intelligence.db")


def make_engine(url: str | None = None):
    url = url or database_url()
    if url.startswith("sqlite"):
        kw: dict = {"connect_args": {"check_same_thread": False}}
        if ":memory:" in url:
            kw["poolclass"] = StaticPool
        return create_engine(url, **kw)
    return create_engine(url, pool_pre_ping=True)


def make_session_factory(engine) -> sessionmaker[Session]:
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)


def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    s = factory()
    try:
        yield s
    finally:
        s.close()

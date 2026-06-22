from pathlib import Path
from datetime import datetime
from typing import Optional

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Float, DateTime, Text, Integer

from config import settings

Path("data").mkdir(exist_ok=True)

engine = create_async_engine(settings.database_url, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String(36))
    score: Mapped[float] = mapped_column(Float)
    comment: Mapped[str] = mapped_column(Text, default="")
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class GenerationLog(Base):
    __tablename__ = "generation_log"

    trace_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(36), index=True)
    question: Mapped[str] = mapped_column(Text)
    viz_type: Mapped[str] = mapped_column(String(64))
    title: Mapped[str] = mapped_column(Text)
    model_id: Mapped[str] = mapped_column(String(128))
    latency_ms: Mapped[float] = mapped_column(Float)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    judge_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    judge_feedback: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class SessionMemory(Base):
    """Per-session conversational memory, persisted so it survives backend reloads.

    Holds a JSON dump of SessionState (turns + readable history + cart + summary +
    workbook refs) keyed by session_id. available_datasources is excluded from the
    blob (re-fetched fresh each session).
    """
    __tablename__ = "session_memory"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    state_json: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def save_session_memory(session_id: str, state_json: str) -> None:
    """Upsert the JSON-serialized SessionState for a session."""
    async with AsyncSessionLocal() as session:
        row = await session.get(SessionMemory, session_id)
        if row is not None:
            row.state_json = state_json
            row.updated_at = datetime.utcnow()
        else:
            session.add(SessionMemory(session_id=session_id, state_json=state_json))
        await session.commit()


async def load_session_memory(session_id: str) -> Optional[str]:
    """Return the persisted state JSON for a session, or None if absent."""
    async with AsyncSessionLocal() as session:
        row = await session.get(SessionMemory, session_id)
        return row.state_json if row is not None else None


async def delete_session_memory(session_id: str) -> None:
    """Remove a session's persisted memory (used by /session/reset)."""
    async with AsyncSessionLocal() as session:
        row = await session.get(SessionMemory, session_id)
        if row is not None:
            await session.delete(row)
            await session.commit()


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

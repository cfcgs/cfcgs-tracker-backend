from unittest.mock import AsyncMock, Mock

import pytest

from src.cfcgs_tracker.service_layer.unit_of_work import SqlAlchemyUnitOfWork


@pytest.mark.asyncio
async def test_sqlalchemy_unit_of_work_delegates_session_methods():
    session = Mock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.refresh = AsyncMock()
    session.flush = AsyncMock()
    session.execute = AsyncMock(return_value="result")
    session.begin_nested = Mock(return_value="nested")

    uow = SqlAlchemyUnitOfWork(session)

    await uow.commit()
    await uow.rollback()
    await uow.refresh("instance")
    await uow.flush()
    result = await uow.execute("statement", {"id": 1})
    nested = uow.begin_nested()

    session.commit.assert_awaited_once()
    session.rollback.assert_awaited_once()
    session.refresh.assert_awaited_once_with("instance")
    session.flush.assert_awaited_once()
    session.execute.assert_awaited_once_with("statement", {"id": 1})
    session.begin_nested.assert_called_once_with()
    assert result == "result"
    assert nested == "nested"


@pytest.mark.asyncio
async def test_sqlalchemy_unit_of_work_rolls_back_on_context_exit_with_error():
    session = Mock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.refresh = AsyncMock()
    session.flush = AsyncMock()
    session.execute = AsyncMock()
    session.begin_nested = Mock()

    uow = SqlAlchemyUnitOfWork(session)

    await uow.__aexit__(RuntimeError, RuntimeError("boom"), None)

    session.rollback.assert_awaited_once()

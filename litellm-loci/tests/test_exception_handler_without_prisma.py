"""PrismaDBExceptionHandler must not raise when the prisma package is absent.

A proxy without a database never installs prisma, yet the auth failure path asks
whether the exception is a database error. That used to raise ModuleNotFoundError
for every failed authentication and hide the real error."""
import sys

import pytest

from litellm.proxy.db.exception_handler import PrismaDBExceptionHandler


@pytest.fixture
def no_prisma(monkeypatch):
    for name in [m for m in sys.modules if m == "prisma" or m.startswith("prisma.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "prisma", None)  # makes `import prisma` raise ImportError
    monkeypatch.setitem(sys.modules, "prisma.engine", None)
    monkeypatch.setitem(sys.modules, "prisma.engine.errors", None)


def test_infrastructure_error_is_false_for_plain_exception(no_prisma):
    assert PrismaDBExceptionHandler.is_database_infrastructure_error(ValueError("x")) is False


def test_connection_error_is_false_for_plain_exception(no_prisma):
    assert PrismaDBExceptionHandler.is_database_connection_error(ValueError("x")) is False

import asyncio
import inspect


def pytest_pyfunc_call(pyfuncitem):
    """
    Minimal async test support for this repo.

    The workspace may not always have pytest-asyncio installed, so we run
    coroutine tests directly with asyncio.run when needed.
    """
    test_function = pyfuncitem.obj
    if not inspect.iscoroutinefunction(test_function):
        return None

    kwargs = {
        name: pyfuncitem.funcargs[name]
        for name in pyfuncitem._fixtureinfo.argnames
    }
    asyncio.run(test_function(**kwargs))
    return True

import sys
from contextlib import contextmanager
from io import StringIO


@contextmanager
def capture(command, *args, **kwargs):
    """
    Captures the output of the given command.
    """
    out, sys.stdout = sys.stdout, StringIO()
    command(*args, **kwargs)
    sys.stdout.seek(0)
    yield sys.stdout.read()
    sys.stdout = out

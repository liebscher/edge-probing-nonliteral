import pytest as pt

from epnl import tasks

class TestTask:

    def test_load_data(self):
        """
        Ensure that data is being loaded properly.
        """
        assert 1
import pytest
import regresion


class TestRegresion:
    def test_int(self):
        assert regresion.duplicar(4)==8

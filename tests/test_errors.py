from conftest import expect_errors
from src.pyactions.ctx import *

@expect_errors(
    """
test_errors.py:11 [test_wrong_type_for_list] Cannot assign str to branches
"""
)
def test_wrong_type_for_list():
    on.pull_request(branches=["main"])
    on.pull_request(branches="dev")
import pytest

# --- Simulate the original buggy function (as it would appear in agentic_code/tools.py) ---
def calculate_tax_buggy(amount: float, rate: float) -> float:
    """
    Calculates tax based on amount and rate.
    This version simulates the bug where 'amount' being None causes a TypeError.
    """
    # Simulate the operation that would fail if amount is None
    # e.g., trying to multiply None by a float
    return amount * rate

# --- Simulate the patched/fixed function ---
def calculate_tax_fixed(amount: float | None, rate: float) -> float:
    """
    Calculates tax based on amount and rate.
    This version fixes the bug by handling 'amount' being None gracefully.
    """
    if amount is None:
        # Treat None amount as 0 for tax calculation purposes
        # Alternatively, one might raise a ValueError if None is considered an invalid input
        # but for tax calculation, 0 is a common and safe default for missing amounts.
        amount = 0.0
    return amount * rate

# --- Pytest Reproduction Test ---

def test_calculate_tax_reproduces_type_error_with_none_amount():
    """
    Test to reproduce Issue #1: TypeError in calculate_tax when amount is None.
    This test should FAIL if the original buggy function is used.
    """
    with pytest.raises(TypeError) as excinfo:
        calculate_tax_buggy(None, 0.10)
    assert "unsupported operand type(s) for *: 'NoneType' and 'float'" in str(excinfo.value)

def test_calculate_tax_passes_with_none_amount_after_fix():
    """
    Test to ensure Issue #1 is fixed: calculate_tax handles None amount gracefully.
    This test should PASS if the fixed function is used.
    It verifies that None is treated as 0, resulting in 0 tax.
    """
    # Test with None amount, expecting 0 tax
    assert calculate_tax_fixed(None, 0.10) == 0.0
    assert calculate_tax_fixed(None, 0.25) == 0.0

    # Ensure existing behavior for non-None amounts is preserved
    assert calculate_tax_fixed(100.0, 0.10) == 10.0
    assert calculate_tax_fixed(50.0, 0.20) == 10.0
    assert calculate_tax_fixed(0.0, 0.15) == 0.0
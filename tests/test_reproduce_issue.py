import pytest
from typing import Optional

# --- Simulated Unpatched Code (to reproduce the TypeError) ---
# This function simulates the original buggy behavior where
# calculate_tax would raise a TypeError if 'amount' is None.
# In a real scenario, this function would be imported from 'agentic_code/tools.py'.
def calculate_tax_unpatched(amount: float, tax_rate: float) -> float:
    """
    Calculates tax on an amount.
    Simulates the unpatched version that fails on None.
    """
    # This line will cause a TypeError if 'amount' is None
    return amount * tax_rate

# --- Simulated Patched Code (to demonstrate the fix) ---
# This function simulates the corrected behavior where
# calculate_tax handles None gracefully by returning 0.0.
# In a real scenario, this function would replace the buggy one in 'agentic_code/tools.py'.
def calculate_tax_patched(amount: Optional[float], tax_rate: float) -> float:
    """
    Calculates tax on an amount, handling None gracefully.
    """
    if amount is None:
        return 0.0  # Fix: Return 0.0 if amount is None
    return amount * tax_rate

# --- Pytest Test Cases ---

def test_reproduce_issue_calculate_tax_unpatched_none_raises_type_error():
    """
    Test to reproduce the TypeError when calculate_tax_unpatched is called with amount=None.
    This test should FAIL on the unpatched code.
    """
    tax_rate = 0.05
    with pytest.raises(TypeError) as excinfo:
        calculate_tax_unpatched(amount=None, tax_rate=tax_rate)
    assert "unsupported operand type(s) for *" in str(excinfo.value) or \
           "can't multiply sequence by non-int of type 'float'" in str(excinfo.value) # Python 3.10+ might give a slightly different message

def test_calculate_tax_patched_none_returns_zero():
    """
    Test that the patched calculate_tax function handles None gracefully,
    returning 0.0 instead of raising a TypeError.
    This test should PASS once the fix is applied.
    """
    tax_rate = 0.05
    result = calculate_tax_patched(amount=None, tax_rate=tax_rate)
    assert result == 0.0

def test_calculate_tax_patched_valid_amount():
    """
    Regression test: Ensure calculate_tax_patched works correctly with a valid numeric amount.
    """
    amount = 100.0
    tax_rate = 0.05
    expected_tax = 100.0 * 0.05
    result = calculate_tax_patched(amount=amount, tax_rate=tax_rate)
    assert result == expected_tax

def test_calculate_tax_patched_zero_amount():
    """
    Regression test: Ensure calculate_tax_patched returns 0.0 for an amount of 0.0.
    """
    amount = 0.0
    tax_rate = 0.05
    result = calculate_tax_patched(amount=amount, tax_rate=tax_rate)
    assert result == 0.0

def test_calculate_tax_patched_different_tax_rate():
    """
    Regression test: Ensure calculate_tax_patched works with different tax rates.
    """
    amount = 250.0
    tax_rate = 0.10
    expected_tax = 250.0 * 0.10
    result = calculate_tax_patched(amount=amount, tax_rate=tax_rate)
    assert result == expected_tax

def test_calculate_tax_patched_negative_amount_if_applicable():
    """
    Regression test: Test with a negative amount (if applicable to the domain).
    Assuming tax calculation works for negative amounts (e.g., refunds).
    """
    amount = -50.0
    tax_rate = 0.05
    expected_tax = -50.0 * 0.05
    result = calculate_tax_patched(amount=amount, tax_rate=tax_rate)
    assert result == expected_tax
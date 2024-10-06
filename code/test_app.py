import pytest

def test_always_passes():
    """A simple test that always passes."""
    assert True  # This will always pass

def test_always_fails():
    """A simple test that always fails."""
    assert False  # This will always fail (optional)

if __name__ == "__main__":
    pytest.main()
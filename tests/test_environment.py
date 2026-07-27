def test_environment():
    import quantlab
    import vectorbt

    assert quantlab is not None
    assert vectorbt is not None

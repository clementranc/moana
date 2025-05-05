from moana.lens import ResonantCaustic

def test_causitc_creation():
    caustic = ResonantCaustic(sep= 1.0, q= 1e-3)
    assert caustic.xcoords.shape[0] > 10
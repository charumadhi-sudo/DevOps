import os
import pytest
import pandas as pd
import yaml
import shutil

@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    # Setup: ensure params.yaml exists
    assert os.path.exists("params.yaml")
    yield
    # Teardown: clean up generated test artifacts if wanted
    # (We can keep them or leave them)

def test_prepare():
    import src.prepare as prepare
    # Run prepare
    prepare.main()
    
    assert os.path.exists("data/train.csv")
    assert os.path.exists("data/test.csv")
    assert os.path.exists("model/scaler.pkl")
    
    # Read test and train csv
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    assert not train_df.empty
    assert not test_df.empty
    assert "target" in train_df.columns
    assert "target" in test_df.columns

def test_train():
    import src.train as train
    # Run train
    train.main()
    
    assert os.path.exists("model/model.pkl")

def test_evaluate():
    import src.evaluate as evaluate
    # Run evaluate (should pass and exit 0)
    try:
        evaluate.main()
    except SystemExit as e:
        assert e.code == 0
        
    assert os.path.exists("metrics.json")

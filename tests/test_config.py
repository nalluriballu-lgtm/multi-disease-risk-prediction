from src.config import DATASETS

def test_four_disease_configs_exist():
    assert set(DATASETS) == {"diabetes", "heart", "kidney", "breast_cancer"}


class ModelConfig:
    model_name: str = "SGDClassification"
    random_state: int = 2
    early_stopping: bool =True
    seperate_test_set = True
    model_names: list[str] = ["SGDClassification", 'LinearRegression']
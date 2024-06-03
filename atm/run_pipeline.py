from model.pipelines.train_pipeline import train_pipeline

"""
Usage:
    From project dirctory, run
    python atm/run_pipeline.py

"""

if __name__ == "__main__":
    train_pipeline(["./data/dataset_train.csv", "./data/dataset-test.csv"])
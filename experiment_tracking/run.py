import os
import mlflow
import argparse
import torch
import torcheval
from torcheval.metrics.functional import r2_score
from predictor_model.config import config
from predictor_model.models import LSTM_model
from predictor_model.processing.data_handling import load_dataset
from predictor_model import data_pipeline, training_pipeline


#### Set hyperparameters here!
# hyperparameters = {
#     "embedding_dim": config.EMBEDDING_DIM,
#     "hidden_dim": config.HIDDEN_DIM,
#     "num_layers": config.NUM_LAYERS,
#     "dropout_rate": config.DROPOUT_RATE,
#     "learning_rate": config.LEARNING_RATE,
#     "weight_decay": config.WEIGHT_DECAY,
#     "epochs": config.EPOCHS
# }

hyperparameters = {
    "embedding_dim": config.EMBEDDING_DIM,
    "hidden_dim": config.HIDDEN_DIM,
    "num_layers": config.NUM_LAYERS,
    "dropout_rate": config.DROPOUT_RATE,
    "learning_rate": config.LEARNING_RATE,
    "weight_decay": config.WEIGHT_DECAY,
    "epochs": 10,
}


def main(model_type="lstm", hyperparameters=hyperparameters):
    train_data = load_dataset("train_tiny.csv")
    test_data = load_dataset("test_tiny.csv")
    train_dataloader = data_pipeline.pipeline(train_data)
    test_dataloader = data_pipeline.pipeline(test_data)
    if model_type == "lstm":
        model = LSTM_model.create_model(
            embedding_dim=hyperparameters["embedding_dim"],
            hidden_dim=hyperparameters["hidden_dim"],
            n_layers=hyperparameters["num_layers"],
            dropout_rate=hyperparameters["dropout_rate"],
        )

    loss_fn = torch.nn.L1Loss()  # could be changed
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparameters["learning_rate"],
        weight_decay=hyperparameters["weight_decay"],
    )
    mlflow.set_experiment("test experiment")
    with mlflow.start_run() as run:
        mlflow.log_params(hyperparameters)
        for epoch in range(hyperparameters["epochs"]):
            print(f"Epoch: {epoch}\n------------")
            training_pipeline.train_step(model, train_dataloader, loss_fn, optimizer)
        eval = training_pipeline.eval_step(model, test_dataloader, loss_fn)
        mlflow.log_metric("r2", eval)


if __name__ == "__main__":
    main()

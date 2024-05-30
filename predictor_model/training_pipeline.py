from predictor_model.config import config
from predictor_model.models import LSTM_model
from predictor_model.processing import dataloaders, data_handling
from predictor_model import data_pipeline
import torcheval
import torch
import os
from torcheval.metrics.functional import r2_score

train_data = data_handling.load_dataset(config.TRAIN_DATA_FILE)
test_data = data_handling.load_dataset(config.TEST_DATA_FILE)


def train_step(model, dataloader, loss_fn, optimizer):
    model.train()
    model.to(config.DEVICE)
    train_loss = 0
    train_r2 = 0
    for text, label in dataloader:
        optimizer.zero_grad()
        label, text = label.to(config.DEVICE), text.to(config.DEVICE)
        y_pred = model(text)
        loss = loss_fn(y_pred.squeeze(), label.squeeze())
        train_loss += loss
        r2 = r2_score(y_pred.squeeze(), label.squeeze())
        train_r2 += r2
        loss.backward()

        optimizer.step()

    train_loss /= len(dataloader)
    train_r2 /= len(dataloader)
    print(f"Train Loss: {train_loss}, r-squared score: {train_r2}")


def eval_step(model, dataloader, loss_fn):
    model.eval()

    eval_loss = 0
    eval_r2 = 0
    with torch.inference_mode():
        for text, label in dataloader:
            label, text = label.to(config.DEVICE), text.to(config.DEVICE)
            y_pred = model(text)
            loss = loss_fn(y_pred.squeeze(), label.squeeze())
            eval_loss += loss
            r2 = r2_score(y_pred.squeeze(), label.squeeze())
            eval_r2 += r2

        eval_loss /= len(dataloader)
        eval_r2 /= len(dataloader)
        print(f"Test Loss: {eval_loss}, r-squared score: {eval_r2}")


def train():
    train_dataloader = data_pipeline.pipeline(train_data)
    test_dataloader = data_pipeline.pipeline(test_data, train=False)

    model = LSTM_model.create_model()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    loss_fn = torch.nn.L1Loss()
    for epoch in range(config.EPOCHS):
        print(f"Epoch: {epoch}\n------------------")
        train_step(model, train_dataloader, loss_fn, optimizer)
        eval_step(model, test_dataloader, loss_fn)

    data_handling.save_model(
        model, os.path.join(config.SAVE_MODEL_PATH, config.SAVE_MODEL_NAME)
    )


if __name__ == "__main__":
    train()

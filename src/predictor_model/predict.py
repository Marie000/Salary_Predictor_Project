import torch
import numpy as np
from predictor_model.config import config
from predictor_model.processing import text_pipeline, data_handling


def predict(text):
    model = data_handling.load_model(name="model_3.pt")

    with torch.no_grad():
        text = torch.tensor(
            text_pipeline.text_pipeline(text), dtype=torch.int64, device=config.DEVICE
        )
        text = torch.unsqueeze(text, 0)
        result = model(text).squeeze()
        result = round(np.exp(result.item()), 2)
        return result


if __name__ == "__main__":
    text = input("enter job description here: ")
    prediction = predict(text)
    print(f"predicted salary: ${str(prediction)}")

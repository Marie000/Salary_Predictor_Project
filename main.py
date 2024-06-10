from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from predictor_model.predict import predict


class JobDescription(BaseModel):
    description: str


app = FastAPI()


@app.get("/")
def main_page():
    return "Welcome to the Salary Predictor API"


@app.post("/predict")
def predict_route(job: JobDescription):
    job_obj = job.model_dump()
    result = predict(job_obj["description"])
    return {"result": result}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)

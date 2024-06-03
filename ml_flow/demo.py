import os
import mlflow
import argparse
import time


def eval(x, y):
    return x**2 + y**2


def main(input1, input2):
    mlflow.set_experiment("Experiment 1")
    with mlflow.start_run():
        # mlflow.set_tags(["tag1", "tag2"])
        mlflow.log_param("param1", input1)
        mlflow.log_param("param2", input2)
        metric_eval = eval(input1, input2)
        mlflow.log_metric("eval_metric", metric_eval)
        os.makedirs("dummy", exist_ok=True)
        with open("dummy/example.txt", "wt") as f:
            f.write(time.asctime())
        mlflow.log_artifact("dummy")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--param1", "-p1", type=int, default=5)
    args.add_argument("--param2", "-p2", type=int, default=10)
    parsed_args = args.parse_args()
    main(parsed_args.param1, parsed_args.param2)

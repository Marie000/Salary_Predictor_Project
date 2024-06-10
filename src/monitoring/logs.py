import whylogs as why
from whylogs.api.writer.whylabs import WhyLabsWriter
import os
import datetime
import numpy as np
from monitoring.create_demo_data import dfs
from predictor_model.processing.preprocessing import preprocess_salary
from predictor_model.predict import predict

os.environ["WHYLABS_DEFAULT_ORG_ID"] = "org-N6maXt"
os.environ["WHYLABS_API_KEY"] = (
    "Xq0OFc9fht.aKAqLlBXmDDrPyDZHux8W4KtajXuAG81SikApMW8sE8mL70RyyMsM:org-N6maXt"
)
os.environ["WHYLABS_DEFAULT_DATASET_ID"] = "model-3"

writer = WhyLabsWriter()

# for i, df in enumerate(dfs):

#     dt = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(days=i)
#     profile = why.log(df).profile()
#     profile.set_dataset_timestamp(dt)
#     writer.write(file=profile.view())

# Using the existing model, track predictions over time

for i, df in enumerate(dfs):
    dt = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(days=i)
    processed_df = preprocess_salary(df)
    processed_df = processed_df[["description", "salary"]]
    processed_df["salary"] = np.exp(processed_df["salary"])

    processed_df["prediction"] = processed_df["description"].map(predict)

    results = why.log_regression_metrics(
        processed_df, target_column="salary", prediction_column="prediction"
    )
    profile = results.profile()
    profile.set_dataset_timestamp(dt)

    results.writer("whylabs").write()

#!/usr/bin/env python
# coding: utf-8

# Import dependencies

import csv
import os
import re
import uuid
from datetime import datetime
from io import StringIO
from pathlib import Path

import fasttext
import neptune.new as neptune
import neptune.new.integrations.optuna as optuna_utils
import optuna
import pandas as pd
import plotly.graph_objects as go
from neptune.new.types import File
from nltk.corpus import stopwords
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

pd.options.plotting.backend = "plotly"
path = Path()

N_JOBS = 1
N_TRIALS = 10
UPLOAD_SIZE_THRESHOLD = 100  # in MB


# (neptune) Initialize a neptune project
# [Read the docs](https://docs.neptune.ai/you-should-know/core-concepts#project)

WORKSPACE_NAME = "showcase"
PROJECT_NAME = "project-text-classification"

project = neptune.init_project(name=f"{WORKSPACE_NAME}/{PROJECT_NAME}")


# (neptune) Log project level metadata

# ## Version and track datasets
# [Read the docs](https://docs.neptune.ai/how-to-guides/data-versioning)

DATASET_PATH_S3 = "s3://neptune-examples/data/text-classification"

project["data/files"].track_files(str(DATASET_PATH_S3))

df_raw = pd.read_csv(f"{DATASET_PATH_S3}/legal_text_classification.csv")
df_raw.dropna(subset=["case_text"], inplace=True)
df_raw.drop_duplicates(subset="case_text", inplace=True)


# (neptune) Log dataset sample
# [Read the docs](https://docs.neptune.ai/you-should-know/what-can-you-log-and-display#files)

csv_buffer = StringIO()
df_raw.sample(100).to_csv(csv_buffer, index=False)
project["data/sample"].upload(File.from_stream(csv_buffer, extension="csv"))


# (neptune) Log metadata plots
# [Read the docs](https://docs.neptune.ai/you-should-know/what-can-you-log-and-display#images)

fig = df_raw.case_outcome.value_counts().plot(kind="bar")
fig.update_xaxes(title="Case outcome")
fig.update_yaxes(title="No. of cases")

project["data/distribution"].upload(fig)


# # Data processing


def clean_text(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Cleans a dataframe `df` string column `col` by applying the following transformations:
    * Convert string to lower-case
    * Remove punctuation
    * Remove numbers
    * Remove single-letter words
    * Remove stopwords
    * Remove multiple and leading/trailing whitespaces

    Args:
        df: Dataframe containing sgtring columns `col` to be cleaned
        col: String column to be cleaned

    Returns:
        A copy of the dataframe `df` with the column `col` cleaned
    """

    tqdm.pandas()
    stop = set(stopwords.words("english"))
    pat = r"\b(?:{})\b".format("|".join(stop))

    _df = df.copy()
    _df[col] = (
        df[col]
        .progress_apply(str.lower)  # Converting to lowercase
        .progress_apply(lambda x: re.sub(r"[^\w\s]", " ", x))  # Removing punctuation
        .progress_apply(
            lambda x: " ".join(x for x in x.split() if not any(c.isdigit() for c in x))
        )  # Removing numbers
        .progress_apply(lambda x: re.sub(r"\b\w\b", "", x))  # Removing single-letter words
        .str.replace(pat, "", regex=True)  # Removing stopwords
        .progress_apply(lambda x: re.sub(r" +", " ", x))  # Removing multiple-whitespaces
        .str.strip()  # Removing leading and whitepaces
    )

    return _df


df_fasttext_raw = df_raw[["case_outcome", "case_text"]]
df_fasttext_raw["label"] = "__label__" + df_fasttext_raw.case_outcome.str.replace(" ", "_")
df_fasttext_raw = df_fasttext_raw[["label", "case_text"]]

DATASET_PATH_LOCAL = path.cwd().parent.parent.joinpath("data")

if not os.path.exists(DATASET_PATH_LOCAL):
    os.makedirs(DATASET_PATH_LOCAL)

DATASET_PATH_LOCAL_FASTTEXT = DATASET_PATH_LOCAL.joinpath("fasttext")

if not os.path.exists(DATASET_PATH_LOCAL_FASTTEXT):
    os.makedirs(DATASET_PATH_LOCAL_FASTTEXT)

TO_CSV_KWARGS = {
    "sep": " ",
    "header": False,
    "index": False,
    "quoting": csv.QUOTE_NONE,
    "quotechar": "",
    "escapechar": " ",
}

df_fasttext_raw.to_csv(DATASET_PATH_LOCAL_FASTTEXT.joinpath("raw.txt"), **TO_CSV_KWARGS)

df_processed = clean_text(df_fasttext_raw, "case_text")
df_processed.drop_duplicates(subset="case_text", inplace=True)

df_processed.to_csv(DATASET_PATH_LOCAL_FASTTEXT.joinpath("processed.txt"), **TO_CSV_KWARGS)

X = df_processed["case_text"]
y = df_processed["label"]

X_train, X_, y_train, y_ = train_test_split(X, y, stratify=y, train_size=0.7)
X_valid, X_test, y_valid, y_test = train_test_split(X_, y_, stratify=y_, train_size=0.5)

print(f"Training size: {X_train.shape}")
print(f"Validation size: {X_valid.shape}")
print(f"Test size: {X_test.shape}")

df_train = pd.DataFrame(data=[y_train, X_train]).T
df_valid = pd.DataFrame(data=[y_valid, X_valid]).T
df_test = pd.DataFrame(data=[y_test, X_test]).T

df_train.to_csv(DATASET_PATH_LOCAL_FASTTEXT.joinpath("train.txt"), **TO_CSV_KWARGS)
df_valid.to_csv(DATASET_PATH_LOCAL_FASTTEXT.joinpath("valid.txt"), **TO_CSV_KWARGS)
df_test.to_csv(DATASET_PATH_LOCAL_FASTTEXT.joinpath("test.txt"), **TO_CSV_KWARGS)


# (neptune) Initialize optuna study-level run

sweep_id = uuid.uuid1()
print(f"Optuna sweep-id: {sweep_id}")

run = neptune.init_run(
    project=f"{WORKSPACE_NAME}/{PROJECT_NAME}",
    name="Fasttext text classification",
    description="Optuna tuned fasttext text classification",
    tags=["fasttext", "optuna", "study-level", "script"],
)


# (neptune) Track run-specific files
# [Read the docs](https://docs.neptune.ai/how-to-guides/data-versioning/compare-datasets#step-2-add-tracking-of-the-dataset-version)

run["data/files"].track_files(os.path.relpath(DATASET_PATH_LOCAL_FASTTEXT))

csv_buffer = StringIO()

df_fasttext_raw.sample(100).to_csv(csv_buffer, index=False)
run["data/sample"].upload(File.from_stream(csv_buffer, extension="csv"))


# (neptune) Log metadata to run
# [Read the docs](https://docs.neptune.ai/you-should-know/logging-metadata)

metadata = {
    "train_size": len(df_train),
    "test_size": len(df_test),
}

run["data/metadata"] = metadata


# Log sweep and trial parameters
# [Read the docs](https://docs.neptune.ai/integrations-and-supported-tools/hyperparameter-optimization/optuna)


def objective_with_logging(trial):

    params = {
        "lr": trial.suggest_float("lr", 0.1, 1, step=0.1),
        "dim": trial.suggest_int("dim", 10, 1000, log=True),
        "ws": trial.suggest_int("ws", 1, 10),
        "epoch": trial.suggest_int("epoch", 10, 100),
        "minCount": trial.suggest_int("minCount", 1, 10),
        "wordNgrams": trial.suggest_int("wordNgrams", 1, 3),
        "loss": trial.suggest_categorical("loss", ["hs", "softmax", "ova"]),
        "bucket": trial.suggest_int("bucket", 1000000, 3000000, log=True),
        "lrUpdateRate": trial.suggest_int("lrUpdateRate", 1, 10),
        "t": trial.suggest_float("t", 0.00001, 0.1, log=True),
    }

    # (neptune) create a trial-level Run
    run_trial_level = neptune.init_run(
        project=f"{WORKSPACE_NAME}/{PROJECT_NAME}",
        name="Fasttext text classification",
        description="Optuna tuned fasttext text classification",
        tags=["fasttext", "optuna", "trial-level", "script"],
    )

    # (neptune) log sweep id to trial-level Run
    run_trial_level["sweep_id"] = sweep_id

    # (neptune) log parameters of a trial-level Run
    clf = fasttext.train_supervised(
        input=str(DATASET_PATH_LOCAL_FASTTEXT.joinpath("train.txt")),
        verbose=0,
        **params,
    )

    properties = {k: v for k, v in vars(clf).items() if k not in ["_words", "f"]}
    run_trial_level["model/properties"] = properties

    # (neptune) run training and calculate the score for this parameter configuration
    preds = [clf.predict(text)[0][0] for text in X_valid.values]

    run_trial_level["validation/metrics/classification_report"] = classification_report(
        y_valid,
        preds,
        output_dict=True,
        zero_division=0,
    )

    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_valid,
        preds,
        average="weighted",
        zero_division=0,
    )

    run_trial_level["validation/metrics/precision"] = precision
    run_trial_level["validation/metrics/recall"] = recall
    run_trial_level["validation/metrics/f1_score"] = f1_score

    # (neptune) stop trial-level Run
    run_trial_level.stop()

    return f1_score


neptune_callback = optuna_utils.NeptuneCallback(run)

study = optuna.create_study(direction="maximize")
study.optimize(
    objective_with_logging,
    n_trials=N_TRIALS,
    callbacks=[neptune_callback],
    n_jobs=N_JOBS,
)

run["study/sweep_id"] = sweep_id


# (neptune) Register a model
# [Read the docs](https://docs.neptune.ai/how-to-guides/model-registry)

model = neptune.init_model(
    model="TXTCLF-FTXT",  # Reinitializing existing model
    # name="fasttext", # Required only for new models
    # key="FTXT", # Required only for new models
    project=f"{WORKSPACE_NAME}/{PROJECT_NAME}",
)


# (neptune) Create a new model version
# [Read the docs](https://docs.neptune.ai/how-to-guides/model-registry/creating-model-versions)

model_version = neptune.init_model_version(
    project=f"{WORKSPACE_NAME}/{PROJECT_NAME}",
    model=model.get_structure()["sys"]["id"].fetch(),
)


# (neptune) Associate model version to run and vice-versa

run_dict = {
    "id": run.get_structure()["sys"]["id"].fetch(),
    "name": run.get_structure()["sys"]["name"].fetch(),
    "url": run.get_run_url(),
}

model_version["run"] = run_dict

model_version_dict = {
    "id": model_version.get_structure()["sys"]["id"].fetch(),
    "url": model_version.get_url(),
}

run["model"] = model_version_dict

clf = fasttext.train_supervised(
    input=str(DATASET_PATH_LOCAL_FASTTEXT.joinpath("train.txt")),
    verbose=5,
    **study.best_params,
)


# (neptune) Upload serialized model to model registry
# [Read the docs](https://docs.neptune.ai/how-to-guides/model-registry/creating-model-versions)

MODEL_PATH = path.cwd().parent.parent.joinpath("models")

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

MODEL_NAME = str(MODEL_PATH.joinpath(f"fasttext_{datetime.now().strftime('%Y%m%d%H%M%S')}.bin"))
clf.save_model(MODEL_NAME)

if os.path.getsize(MODEL_NAME) < 1024 * 1024 * UPLOAD_SIZE_THRESHOLD:  # 100 MB
    print("Uploading serialized model")
    model_version["serialized_model"].upload(str(MODEL_NAME))
else:
    print(
        f"Model is larger than UPLOAD_SIZE_THRESHOLD ({UPLOAD_SIZE_THRESHOLD} MB). Tracking pointer to model file"
    )
    model_version["serialized_model"].track_files(os.path.relpath(MODEL_NAME))


# (neptune) Log model properties to model_version

properties = {k: v for k, v in vars(clf).items() if k not in ["_words", "f"]}

model_version["properties"] = properties


# (neptune) Log parameters, metrics and debugging information to run and model version

preds = [clf.predict(text)[0][0] for text in X_test.values]

precision, recall, f1_score, _ = precision_recall_fscore_support(
    y_test,
    preds,
    average="weighted",
    zero_division=0,
)

print(f"Precision: {precision}\nRecall: {recall}\nF1-score: {f1_score}")

run["test/metrics/precision"] = model_version["metrics/precision"] = precision
run["test/metrics/recall"] = model_version["metrics/recall"] = recall
run["test/metrics/f1_score"] = model_version["metrics/f1_score"] = f1_score

print(classification_report(y_test, preds, zero_division=0))


# (neptune) Log each metric in its separate nested namespace
run["test/metrics/classification_report"] = classification_report(
    y_test, preds, output_dict=True, zero_division=0
)

# (neptune) Log classification report as an HTML dataframe
df_clf_rpt = pd.DataFrame(classification_report(y_test, preds, output_dict=True, zero_division=0)).T
run["test/metrics/classification_report/report"].upload(File.as_html(df_clf_rpt))

fig = ConfusionMatrixDisplay.from_predictions(
    y_test, preds, xticks_rotation="vertical", colorbar=False
)
run["test/debug/plots/confusion_matrix"].upload(fig.figure_)

df_test["prediction"] = preds

labels = [s.replace("__label__", "") for s in df_test.label.value_counts().index]
fig = go.Figure(
    data=[
        go.Bar(name="Actual", x=labels, y=df_test.label.value_counts()),
        go.Bar(name="Prediction", x=labels, y=df_test.prediction.value_counts()),
    ]
)
fig.update_layout(title="Actual vs Prediction", barmode="group")

run["test/debug/plots/prediction_distribution"].upload(fig)

# (neptune) Log misclassified results

df_debug = df_test[df_test.label != df_test.prediction]

csv_buffer = StringIO()

df_debug.to_csv(csv_buffer, index=False)
run["test/debug/misclassifications"].upload(File.from_stream(csv_buffer, extension="csv"))


# (neptune) Explore the [project](https://app.neptune.ai/showcase/project-text-classification) in the Neptune app

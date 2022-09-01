#!/usr/bin/env python
# coding: utf-8

# # Initialize a neptune project
# [Read the docs](https://docs.neptune.ai/you-should-know/core-concepts#project)

# In[3]:
import argparse
import csv
from io import StringIO

import fasttext
import neptune.new as neptune
import pandas as pd
import plotly.graph_objects as go
from neptune.new.types import File
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score

# In[4]:
parser = argparse.ArgumentParser()
parser.add_argument("workspace", help="workspace name")
parser.add_argument("project", help="project name")
args = parser.parse_args()

WORKSPACE_NAME = args.workspace
PROJECT_NAME = args.project


# In[6]:


DATASET_PATH = "classification/data"


# In[8]:


# In[9]:


df_raw = pd.read_csv(f"{DATASET_PATH}/raw/legal_text_classification.csv")
df_raw.dropna(subset=["case_text"], inplace=True)
df_raw


# In[10]:


df_raw.isna().sum()


# In[11]:


sum(df_raw.case_text.duplicated())


# In[12]:


df_raw.drop_duplicates(subset="case_text", inplace=True)
sum(df_raw.case_text.duplicated())


# In[13]:


# # Initialize a new neptune run for baseline model
# [Read the docs](https://docs.neptune.ai/you-should-know/core-concepts#run)

# In[18]:


run = neptune.init_run(
    project=f"{WORKSPACE_NAME}/{PROJECT_NAME}",
    name="text classification using fasttext",
    description="training on raw data",
    tags=["fasttext", "raw"],
)


# In[19]:


df_fasttext_raw = df_raw[["case_outcome", "case_text"]]
df_fasttext_raw["label"] = "__label__" + df_fasttext_raw.case_outcome
df_fasttext_raw = df_fasttext_raw[["label", "case_text"]]
df_fasttext_raw


# In[20]:


df_fasttext_raw.to_csv(
    f"{DATASET_PATH}/fasttext/raw.txt",
    sep=" ",
    header=False,
    index=False,
    quoting=csv.QUOTE_NONE,
    quotechar="",
    escapechar=" ",
)


# ## Track run-specific files
# [Read the docs](https://docs.neptune.ai/how-to-guides/data-versioning/compare-datasets#step-2-add-tracking-of-the-dataset-version)

# In[22]:


csv_buffer = StringIO()

df_fasttext_raw.sample(100).to_csv(csv_buffer, index=False)
run["data/sample"].upload(File.from_stream(csv_buffer, extension="csv"))


# In[23]:


def train_test_valid_split(X: pd.DataFrame, y: list) -> pd.DataFrame:
    """Splits `X` into train, test and validation sets stratified on `y`"""

    from sklearn.model_selection import train_test_split

    X_train, _X_test, y_train, _y_test = train_test_split(X, y, stratify=y, test_size=0.4)
    X_test, X_valid, y_test, y_valid = train_test_split(
        _X_test, _y_test, stratify=_y_test, test_size=0.5
    )

    print(X_train.shape)
    print(len(y_train))
    print(X_test.shape)
    print(len(y_test))
    print(X_valid.shape)
    print(len(y_valid))

    return X_train, y_train, X_test, y_test, X_valid, y_valid


# In[24]:


X = df_fasttext_raw["case_text"]
y = df_fasttext_raw["label"]


# In[25]:


X_train, y_train, X_test, y_test, X_valid, y_valid = train_test_valid_split(X, y)


# In[26]:


df_train = pd.DataFrame(data=[y_train, X_train]).T
df_train


# In[27]:


df_test = pd.DataFrame(data=[y_test, X_test]).T
df_valid = pd.DataFrame(data=[y_valid, X_valid]).T


# In[28]:


df_train.to_csv(
    f"{DATASET_PATH}/raw/train.txt",
    sep=" ",
    header=False,
    index=False,
    quoting=csv.QUOTE_NONE,
    quotechar="",
    escapechar=" ",
)
df_test.to_csv(
    f"{DATASET_PATH}/raw/test.txt",
    sep=" ",
    header=False,
    index=False,
    quoting=csv.QUOTE_NONE,
    quotechar="",
    escapechar=" ",
)
df_valid.to_csv(
    f"{DATASET_PATH}/raw/valid.txt",
    sep=" ",
    header=False,
    index=False,
    quoting=csv.QUOTE_NONE,
    quotechar="",
    escapechar=" ",
)


# In[29]:


run["data/files"].track_files(f"{DATASET_PATH}/raw")


# ## Log metadata to run
# [Read the docs](https://docs.neptune.ai/you-should-know/logging-metadata)

# In[30]:


metadata = {
    "train_size": len(df_train),
    "test_size": len(df_test),
    "valid_size": len(df_valid),
}
metadata


# In[31]:


run["data/metadata"] = metadata


# In[42]:


# In[43]:


clf = fasttext.train_supervised(input=f"{DATASET_PATH}/raw/train.txt")


# In[44]:


clf.save_model("classification/models/fasttext_baseline_script.bin")


# ## Log parameters, metrics and debugging information to run

# In[48]:


# In[49]:


_, precision, recall = clf.test(f"{DATASET_PATH}/raw/test.txt")
print(precision, recall)


# In[50]:


run["test/metrics/precision"] = precision
run["test/metrics/recall"] = recall


# In[51]:


preds = [clf.predict(text)[0][0] for text in X_test.values]
set(preds)


# In[52]:


print(classification_report(y_test, preds, zero_division=0))
run["test/metrics/classification_report"] = classification_report(
    y_test, preds, output_dict=True, zero_division=0
)


# In[56]:


df_clf_rpt = pd.DataFrame(classification_report(y_test, preds, output_dict=True, zero_division=0)).T
run["test/metrics/classification_report/report"].upload(File.as_html(df_clf_rpt))


# In[57]:


f1_score(y_test, preds, average="weighted")
run["test/metrics/f1_score"] = f1_score(y_test, preds, average="weighted")


# In[58]:


fig = ConfusionMatrixDisplay.from_predictions(
    y_test, preds, xticks_rotation="vertical", colorbar=False
)
run["test/debug/plots/confusion_matrix"].upload(fig.figure_)


# In[59]:


df_test["prediction"] = preds
df_test


# In[60]:


# In[61]:


labels = [s.replace("__label__", "") for s in df_test.label.unique()]
fig = go.Figure(
    data=[
        go.Bar(name="Actual", x=labels, y=df_test.label.value_counts()),
        go.Bar(name="Prediction", x=labels, y=df_test.prediction.value_counts()),
    ]
)
fig.update_layout(title="Actual vs Prediction", barmode="group")
fig.show()


# In[62]:


run["test/debug/plots/prediction_distribution"].upload(fig)


# In[63]:


df_debug = df_test[df_test.label != df_test.prediction]

csv_buffer = StringIO()

df_debug.to_csv(csv_buffer, index=False)
run["test/debug/misclassifications"].upload(File.from_stream(csv_buffer, extension="csv"))


# ## Stop current run

# In[64]:


run.stop()

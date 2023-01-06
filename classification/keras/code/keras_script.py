#!/usr/bin/env python
# coding: utf-8

# # Text classification using Keras with Neptune tracking
# Notebook inspired from https://keras.io/examples/nlp/text_classification_from_scratch/

# ## Setup

# In[1]:


import os

import tensorflow as tf
import utils

# (Neptune) Import Neptune and initialize a project

# In[2]:


os.environ["NEPTUNE_PROJECT"] = "showcase/project-text-classification"


# In[3]:


import neptune.new as neptune

project = neptune.init_project()


# ## Data preparation
# We are using the IMDB sentiment analysis data available at https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz. For the purposes of this demo, we've uploaded this data to S3 at https://neptune-examples.s3.us-east-2.amazonaws.com/data/text-classification/aclImdb_v1.tar.gz and will be downloading it from there.

# ### (Neptune) Track datasets using Neptune
# Since this dataset will be used among all the runs in the project, we track it at the project level

# In[4]:


project["data/keras/files"].track_files(
    "s3://neptune-examples/data/text-classification/aclImdb_v1.tar.gz"
)
project.sync()


# ### (Neptune) Download files from S3 using Neptune

# In[5]:


print("Downloading data...")
project["data/keras/files"].download("..")


# ### Prepare data

# In[6]:


utils.extract_files(source="../aclImdb_v1.tar.gz", destination="..")
utils.prep_data(imdb_folder="../aclImdb", dest_path="../data")


# (Neptune) Upload dataset sample to Neptune project

# In[7]:


import random

base_namespace = "data/keras/sample/"

project[base_namespace]["train/pos"].upload(
    f"../data/train/pos/{random.choice(os.listdir('../data/train/pos'))}"
)
project[base_namespace]["train/neg"].upload(
    f"../data/train/neg/{random.choice(os.listdir('../data/train/neg'))}"
)
project[base_namespace]["test/pos"].upload(
    f"../data/test/pos/{random.choice(os.listdir('../data/test/pos'))}"
)
project[base_namespace]["test/neg"].upload(
    f"../data/test/neg/{random.choice(os.listdir('../data/test/neg'))}"
)


# ### Generate training, validation, and test datasets

# In[8]:


data_params = {
    "batch_size": 32,
    "validation_split": 0.2,
    "max_features": 2000,
    "embedding_dim": 128,
    "sequence_length": 500,
    "seed": 1,
}


# (Neptune) Log data metadata to Neptune

# In[9]:


run = neptune.init_run(name="Keras text classification", tags=["keras"])


# In[10]:


run["data/params"] = data_params


# In[11]:


import re
import string

from tensorflow.keras.layers import TextVectorization

# In[12]:


raw_train_ds, raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    "../data/train",
    batch_size=data_params["batch_size"],
    validation_split=data_params["validation_split"],
    subset="both",
    seed=data_params["seed"],
)

raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    "../data/test", batch_size=data_params["batch_size"]
)

print(f"Number of batches in raw_train_ds: {raw_train_ds.cardinality()}")
print(f"Number of batches in raw_val_ds: {raw_val_ds.cardinality()}")
print(f"Number of batches in raw_test_ds: {raw_test_ds.cardinality()}")


# ### Clean data

# In[13]:


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(stripped_html, f"[{re.escape(string.punctuation)}]", "")


vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=data_params["max_features"],
    output_mode="int",
    output_sequence_length=data_params["sequence_length"],
)

text_ds = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)


# ### Vectorize data

# In[14]:


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


# Vectorize the data.
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# Do async prefetching / buffering of the data for best performance on GPU.
train_ds = train_ds.cache().prefetch(buffer_size=10)
val_ds = val_ds.cache().prefetch(buffer_size=10)
test_ds = test_ds.cache().prefetch(buffer_size=10)


# ## Modelling

# (Neptune) Create a new model and model version

# In[15]:


from neptune.new.exceptions import NeptuneModelKeyAlreadyExistsError

project_key = project.get_structure()["sys"]["id"].fetch()

try:
    model = neptune.init_model(name="keras", key="KER")
    model.stop()
except NeptuneModelKeyAlreadyExistsError:
    # If it already exists, we don't have to do anything.
    pass

model_version = neptune.init_model_version(model=f"{project_key}-KER", name="keras")


# ### Build a model

# In[16]:


model_params = {
    "dropout": 0.5,
    "strides": 3,
    "activation": "relu",
    "kernel_size": 7,
    "loss": "binary_crossentropy",
    "optimizer": "adam",
    "metrics": ["accuracy"],
}


# In[17]:


model_version["params"] = model_params


# In[18]:


keras_model = utils.build_model(model_params, data_params)


# ### Train the model

# (Neptune) Initialize the Neptune callback

# In[19]:


from neptune.new.integrations.tensorflow_keras import NeptuneCallback

neptune_callback = NeptuneCallback(run=run, log_model_diagram=True, log_on_batch=True)


# In[20]:


training_params = {
    "epochs": 3,
}


# In[21]:


# Fit the model using the train and test datasets.
keras_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=training_params["epochs"],
    callbacks=neptune_callback,
)
# Training parameters are logged automatically to Neptune


# ### Evaluate the model

# In[22]:


_, curr_model_acc = keras_model.evaluate(test_ds, callbacks=neptune_callback)


# ## (Neptune) Associate run with model and vice-versa

# In[23]:


run_meta = {
    "id": run.get_structure()["sys"]["id"].fetch(),
    "name": run.get_structure()["sys"]["name"].fetch(),
    "url": run.get_url(),
}

print(run_meta)


# In[24]:


model_version["run"] = run_meta


# In[25]:


model_version_meta = {
    "id": model_version.get_structure()["sys"]["id"].fetch(),
    "name": model_version.get_structure()["sys"]["name"].fetch(),
    "url": model_version.get_url(),
}

print(model_version_meta)


# In[26]:


run["training/model/meta"] = model_version_meta


# ## (Neptune) Upload serialized model and model weights to Neptune

# In[27]:


model_version["serialized_model"] = keras_model.to_json()


# In[28]:


keras_model.save_weights("model_weights.h5")
model_version["model_weights"].upload("model_weights.h5")


# (Neptune) Wait for all operations to sync with Neptune servers

# In[29]:


model_version.sync()


# ## (Neptune) Promote best model to production

# ### (Neptune) Fetch current production model

# In[30]:


with neptune.init_model(with_id=f"{project_key}-KER") as model:
    model_versions_df = model.fetch_model_versions_table().to_pandas()


# In[31]:


production_models = model_versions_df[model_versions_df["sys/stage"] == "production"]["sys/id"]
assert (
    len(production_models) == 1
), f"Multiple model versions found in production: {production_models.values}"


# In[32]:


prod_model_id = production_models.values[0]
print(f"Current model in production: {prod_model_id}")


# In[33]:


npt_prod_model = neptune.init_model_version(with_id=prod_model_id)
npt_prod_model_params = npt_prod_model["params"].fetch()
prod_model = tf.keras.models.model_from_json(
    npt_prod_model["serialized_model"].fetch(), custom_objects=None
)

npt_prod_model["model_weights"].download()
prod_model.load_weights("model_weights.h5")


# ### (Neptune) Evaluate current model on lastest test data

# In[34]:


# using the model's original loss and optimizer, but the current metric
prod_model.compile(
    loss=npt_prod_model_params["loss"],
    optimizer=npt_prod_model_params["optimizer"],
    metrics=model_params["metrics"],
)

_, prod_model_acc = prod_model.evaluate(test_ds)


# ### (Neptune) If challenger model outperforms production model, promote it to production

# In[36]:


print(f"Production model accuracy: {prod_model_acc}\nChallenger model accuracy: {curr_model_acc}")

if curr_model_acc > prod_model_acc:
    print("Promoting challenger to production")
    npt_prod_model.change_stage("archived")
    model_version.change_stage("production")
else:
    print("Archiving challenger model")
    model_version.change_stage("archived")

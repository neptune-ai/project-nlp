#!/usr/bin/env python
# coding: utf-8

# # Text classification using Keras with Neptune tracking
# Notebook inspired from https://keras.io/examples/nlp/text_classification_from_scratch/

# ## Setup

# In[41]:


import os

import numpy as np
import tensorflow as tf

# (Neptune) Import Neptune and initialize a project

# In[42]:


os.environ["NEPTUNE_PROJECT"] = "showcase/project-text-classification"


# In[43]:


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


# ### Extract downloaded files

# In[6]:


import tarfile

print("Extracting data...")
my_tar = tarfile.open("../aclImdb_v1.tar.gz")
my_tar.extractall("..")
my_tar.close()


# ### Remove `unsup` subfolder and rename `aclImdb` folder

# In[7]:


import shutil

shutil.rmtree("../aclImdb/train/unsup")
os.remove("../aclImdb_v1.tar.gz")

if os.path.exists("../data"):
    shutil.rmtree("../data")
os.rename("../aclImdb", "../data")


# (Neptune) Upload dataset sample to Neptune project

# In[8]:


import random

project["data/keras/sample/train/pos"].upload(
    f"../data/train/pos/{random.choice(os.listdir('../data/train/pos'))}"
)
project["data/keras/sample/train/neg"].upload(
    f"../data/train/neg/{random.choice(os.listdir('../data/train/neg'))}"
)
project["data/keras/sample/test/pos"].upload(
    f"../data/test/pos/{random.choice(os.listdir('../data/test/pos'))}"
)
project["data/keras/sample/test/neg"].upload(
    f"../data/test/neg/{random.choice(os.listdir('../data/test/neg'))}"
)


# ### Generate training, validation, and test datasets

# In[44]:


data_params = {
    "batch_size": 32,
    "validation_split": 0.2,
    "max_features": 20000,
    "embedding_dim": 128,
    "sequence_length": 500,
    "seed": 42,
}


# (Neptune) Log data metadata to Neptune

# In[45]:


run = neptune.init_run(name="Keras text classification", tags=["keras"])


# In[46]:


run["data/params"] = data_params


# In[47]:


import re
import string

from tensorflow.keras.layers import TextVectorization

# In[48]:


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


# ### Previewing data

# In[49]:


for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(5):
        print(text_batch.numpy()[i])
        print(label_batch.numpy()[i])


# ### Clean data

# In[50]:


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

# In[51]:


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

# In[52]:


from neptune.new.exceptions import NeptuneModelKeyAlreadyExistsError

project_key = project.get_structure()["sys"]["id"].fetch()

try:
    model = neptune.init_model(name="keras", key="KER")
    model.stop()
    model_version = neptune.init_model_version(model=f"{project_key}-KER", name="keras")
except NeptuneModelKeyAlreadyExistsError:
    model_version = neptune.init_model_version(model=f"{project_key}-KER", name="keras")


# ### Build a model

# In[53]:


model_params = {
    "dropout": 0.5,
    "strides": 3,
    "activation": "relu",
    "kernel_size": 7,
    "loss": "binary_crossentropy",
    "optimizer": "adam",
    "metrics": ["accuracy"],
}


# In[54]:


model_version["params"] = model_params


# In[55]:


from tensorflow.keras import layers

# A integer input for vocab indices.
inputs = tf.keras.Input(shape=(None,), dtype="int64")

# Next, we add a layer to map those vocab indices into a space of dimensionality
# 'embedding_dim'.
x = layers.Embedding(data_params["max_features"], data_params["embedding_dim"])(inputs)
x = layers.Dropout(model_params["dropout"])(x)

# Conv1D + global max pooling
x = layers.Conv1D(
    data_params["embedding_dim"],
    model_params["kernel_size"],
    padding="valid",
    activation=model_params["activation"],
    strides=model_params["strides"],
)(x)
x = layers.Conv1D(
    data_params["embedding_dim"],
    model_params["kernel_size"],
    padding="valid",
    activation=model_params["activation"],
    strides=model_params["strides"],
)(x)
x = layers.GlobalMaxPooling1D()(x)

# We add a vanilla hidden layer:
x = layers.Dense(data_params["embedding_dim"], activation=model_params["activation"])(x)
x = layers.Dropout(model_params["dropout"])(x)

# We project onto a single unit output layer, and squash it with a sigmoid:
predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

keras_model = tf.keras.Model(inputs, predictions)

# Compile the model with binary crossentropy loss and an adam optimizer.
keras_model.compile(
    loss=model_params["loss"], optimizer=model_params["optimizer"], metrics=model_params["metrics"]
)


# ### Train the model

# (Neptune) Initialize the Neptune callback

# In[56]:


from neptune.new.integrations.tensorflow_keras import NeptuneCallback

neptune_callback = NeptuneCallback(run=run, log_model_diagram=True, log_on_batch=True)


# In[57]:


training_params = {
    "epochs": 3,
}


# In[58]:


# Fit the model using the train and test datasets.
keras_model.fit(
    train_ds, validation_data=val_ds, epochs=training_params["epochs"], callbacks=neptune_callback
)
# Training parameters are logged automatically to Neptune


# ### Evaluate the model

# In[59]:


_, acc = keras_model.evaluate(test_ds, callbacks=neptune_callback)


# ## (Neptune) Associate run with model and vice-versa

# In[60]:


run_meta = {
    "id": run.get_structure()["sys"]["id"].fetch(),
    "name": run.get_structure()["sys"]["name"].fetch(),
    "url": run.get_url(),
}
run_meta


# In[61]:


model_version["run"] = run_meta


# In[62]:


model_version_meta = {
    "id": model_version.get_structure()["sys"]["id"].fetch(),
    "name": model_version.get_structure()["sys"]["name"].fetch(),
    "url": model_version.get_url(),
}
model_version_meta


# In[63]:


run["training/model/meta"] = model_version_meta


# ## (Neptune) Upload serialized model and model weights to Neptune

# In[64]:


model_version["serialized_model"] = keras_model.to_json()


# In[65]:


keras_model.save_weights("model_weights.h5")
model_version["model_weights"].upload("model_weights.h5")


# (Neptune) Wait for all operations to sync with Neptune servers

# In[66]:


model_version.sync()


# ## (Neptune) Promote best model to production

# ### (Neptune) Fetch current production model

# In[67]:


model = neptune.init_model(with_id=f"{project_key}-KER")
model_versions_df = model.fetch_model_versions_table().to_pandas()
model.stop()
model_versions_df


# In[68]:


production_models = model_versions_df[model_versions_df["sys/stage"] == "production"]["sys/id"]
assert (
    len(production_models) == 1
), f"Multiple model versions found in production: {production_models.values}"


# In[69]:


prod_model_id = production_models.values[0]
print(f"Current model in production: {prod_model_id}")


# In[70]:


prod_model = neptune.init_model_version(with_id=prod_model_id)
prod_model_params = prod_model["params"].fetch()
loaded_prod_model = tf.keras.models.model_from_json(
    prod_model["serialized_model"].fetch(), custom_objects=None
)

prod_model["model_weights"].download()
loaded_prod_model.load_weights("model_weights.h5")


# ### (Neptune) Evaluate current model on lastest test data

# In[71]:


# using the model's original loss and optimizer, but the current metric
loaded_prod_model.compile(
    loss=prod_model_params["loss"],
    optimizer=prod_model_params["optimizer"],
    metrics=model_params["metrics"],
)

_, prod_model_acc = loaded_prod_model.evaluate(test_ds)


# ### (Neptune) If challenger model outperforms production model, promote it to production

# In[72]:


print(f"Production model accuracy: {prod_model_acc}\nChallenger model accuracy: {acc}")

if acc > prod_model_acc:
    print("Promoting challenger to production")
    prod_model.change_stage("archived")
    model_version.change_stage("production")
else:
    print("Archiving challenger model")
    model_version.change_stage("archived")

prod_model.stop()


# ## (Neptune) Stop tracking

# In[73]:


model_version.stop()
run.stop()
project.stop()

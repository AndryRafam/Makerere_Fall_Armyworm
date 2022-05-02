import tensorflow as tf
import numpy as np
import random
import pandas as pd

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

train_df = pd.read_csv("Train.csv")
dict = {0:"healthy", 1:"fall_armyworm"}
train_df.Label = train_df.Label.map(dict)
print(train_df.head())


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(
    rotation_range = 10,
    zoom_range = 0.1,
    validation_split = 0.2,
)

train_ds = train_gen.flow_from_dataframe(
    directory = "Images",
    dataframe = train_df,
    x_col = "Image_id",
    y_col = "Label",
    target_size = (256,256),
    batch_size = 32,
    class_mode = "categorical",
    shuffle = True,
    subset = "training",
)

val_ds = train_gen.flow_from_dataframe(
    directory = "Images",
    dataframe = train_df,
    x_col = "Image_id",
    y_col = "Label",
    target_size = (256,256),
    batch_size = 32,
    class_mode = "categorical",
    shuffle = True,
    subset = "validation",
)

from tensorflow.keras.applications.vgg19 import preprocess_input, VGG19

base_model = VGG19(input_shape=(256,256,3),include_top=False,weights="imagenet")
base_model.trainable = True

for layer in base_model.layers:
    if layer.name == "block3_pool":
        break
    layer.trainable = False

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Flatten
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import CategoricalCrossentropy

def model(y):
    x = preprocess_input(y)
    x = base_model(x,training=False)
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(128,activation="relu")(x)
    x = Dropout(0.2,seed=42)(x)
    x = Dense(64,activation="relu")(x)
    x = Dropout(0.2,seed=42)(x)
    output = Dense(2,activation="sigmoid")(x)
    model = Model(y,output,name="transfer_vgg19")
    return model

model = model(tf.keras.Input(shape=(256,256,3)))
model.summary()
model.compile(RMSprop(learning_rate=1e-5),CategoricalCrossentropy(),metrics=["accuracy"])


from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

if __name__=="__main__":
    checkpoint = ModelCheckpoint("fall_armyworm.h5",save_weights_only=False,save_best_only=True)
    model.fit(train_ds,epochs=5,validation_data=val_ds,callbacks=[checkpoint])
    best = load_model("fall_armyworm.h5")
    loss,acc = best.evaluate(val_ds)
    print("\nAccuracy: {:.2f} %".format(100*acc))
    print("Loss: {:.2f} %".format(100*loss))



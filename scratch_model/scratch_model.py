import tensorflow as tf
import pandas as pd
import random
import numpy as np

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


from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Rescaling, Conv2D, MaxPool2D, Flatten, Dense, Dropout

def model(y):
    x = Rescaling(1./255)(y)
    x = Conv2D(64,3,padding="same",activation="relu",strides=(2,2))(x)
    x = Conv2D(64,3,padding="same",activation="relu",strides=(2,2))(x)
    x = MaxPool2D(pool_size=(2,2),strides=(2,2))(x)
    
    x = Conv2D(128,3,padding="same",activation="relu",strides=(2,2))(x)
    x = Conv2D(128,3,padding="same",activation="relu",strides=(2,2))(x)
    x = MaxPool2D(pool_size=(2,2),strides=(2,2))(x)

    x = Conv2D(256,3,padding="same",activation="relu",strides=(2,2))(x)
    x = Conv2D(256,3,padding="same",activation="relu",strides=(2,2))(x)
    x = Conv2D(256,3,padding="same",activation="relu",strides=(2,2))(x)

    x = Flatten()(x)
    x = Dense(512,activation="relu")(x)
    x = Dropout(0.2,seed=42)(x)
    x = Dense(512,activation="relu")(x)
    x = Dropout(0.2,seed=42)(x)
    
    output = Dense(2)(x)
    model = Model(y,output)
    return model


from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import CategoricalCrossentropy

model = model(Input(shape=(256,256,3)))
model.summary()
model.compile(RMSprop(learning_rate=1e-3),CategoricalCrossentropy(from_logits=True),metrics=["accuracy"])


from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

if __name__=="__main__":
    checkpoint = ModelCheckpoint("fall_armyworm.h5",save_weights_only=False,save_best_only=True,monitor="val_accuracy")
    model.fit(train_ds,epochs=25,validation_data=val_ds,callbacks=[checkpoint])
    best = load_model("fall_armyworm.h5")
    loss,acc = best.evaluate(val_ds)
    print("\nAccuracy: {:.2f} %".format(100*acc))
    print("Loss: {:.2f} %".format(100*loss))

from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model


def expand_model(model):
  preds = model.output
  preds = GlobalAveragePooling2D()(preds)
  preds = Dense(1024,activation='relu')(preds) 
  preds = Dense(1024,activation='relu')(preds) 
  preds = Dense(512,activation='relu')(preds) 
  preds = Dense(number_of_catagories,activation='softmax')(preds)
  return preds

def set_training_layers(model, untrained_layers=20):
  for idl,layer in enumerate(model.layers):
    if idl<untrained_layers:
      layer.trainable = False
    else: 
      layer.trainable = True
  return model 


# define
train_location = './train/'

number_of_catagories = 4

# build base
base_model = MobileNet(weights='imagenet', include_top=False) 
expanded_model = expand_model(base_model)

# build final
tmp_model = Model(inputs=base_model.input, outputs=expanded_model)

final_model = set_training_layers(tmp_model)

# compile
final_model.compile(optimizer='Adam', 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])
              

# set up training data
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(train_location, 
                                                    target_size=(224,224),
                                                    color_mode='rgb',
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    shuffle=True)



step_size_train=train_generator.n//train_generator.batch_size

# train model
final_model.fit_generator(generator=train_generator,
                          steps_per_epoch=step_size_train,
                          epochs=5)



























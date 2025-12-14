import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from keras.utils import to_categorical
from keras.saving import register_keras_serializable
from keras.models import load_model
from sklearn.model_selection import train_test_split
import tensorflow_advanced_segmentation_models as tasm
from simple_multi_unet_model import multi_unet_model, jacard_coef
from unet_xception_model import create_model, focal_tversky_loss
#from simple_multi_unet_model import multi_unet_model, jacard_coef
import numpy as np
from IPython.display import clear_output 
import matplotlib.pyplot as plt
from PIL import Image
import random
import os
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint


root_directory = "Data/resized/"
patch_size=128
TRAIN_SIZE = 800
training =0
model =1
size =256

# <----------------------Check dataset---------------------------->
def CheckDataset(image_dataset, label_dataset):
    image_number = random.randint(0, len(image_dataset) - 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.title("Image")
    plt.imshow(np.array(image_dataset[image_number]))

    plt.subplot(122)
    plt.title("Label")
    label = np.array(label_dataset[image_number])
    if label.ndim == 2:
        plt.imshow(label, cmap='gray')
    else:
        plt.imshow(label)

    plt.show()
 # <-------------------------------------------------------------->


def normalize(input_image):
    input_image = tf.cast(input_image, tf.float32)/255
    return input_image

def formDataset(input_type, patch_size, root_directory):
    pic_dataset=[]
    for path,subdirs,files in os.walk(root_directory):
        print(path)
    #Splits path at directories' separator
        dirname = path.split(os.path.sep)[-1]
        #print(dirname)
        if dirname==input_type:
            pics = sorted(os.listdir(path))
            #print(root_directory, pics)
            num=0
            for num_of_pics, pic_name in enumerate(pics):
                if pic_name.endswith(".png"):
                    num+=1
                    pic = np.array(Image.open(path+'/'+pic_name))
                    if np.max(pic) >1:
                         pic = normalize(pic)
                    pic_dataset.append(pic)
                if num>=TRAIN_SIZE:
                    break
    return pic_dataset

image_dataset=formDataset("image",patch_size,root_directory+'train'+str(size)+'/')
#print("image dataset_created")
label_dataset=formDataset("label",patch_size,root_directory+'train'+str(size)+'/')
#print("label_dataset_created")
#print(label_dataset)
print("Number of images:", len(image_dataset))
print("Number of labels:", len(label_dataset))


image_dataset = np.array(image_dataset)
labels_dataset =  np.array(label_dataset)


#print (np.unique(label_dataset))
    

# <----------------------Check dataset---------------------------->
#CheckDataset(image_dataset, label_dataset)

####################################################################

n_classes = len(np.unique(label_dataset))
#print(n_classes)
label_cat = to_categorical(labels_dataset, num_classes=n_classes)

X_train,X_test,Y_train,Y_test=train_test_split(image_dataset,label_cat,test_size=0.2,random_state=42)

base_model=tf.keras.applications.MobileNetV2(input_shape=[patch_size,patch_size,3], include_top=False)


weights = [1,2]
#print(weights)

dice_loss = tasm.losses.DiceLoss(class_weights=weights)
focal_loss = tasm.losses.CategoricalFocalLoss()

@register_keras_serializable(package="CustomLosses")
def total_loss(y_true, y_pred):
    return dice_loss(y_true, y_pred) + focal_loss(y_true, y_pred)

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]

metrics = ['accuracy', jacard_coef]
if training==1:
    if model==1:
        def get_model():
            return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)


        model = get_model()
        model.compile(optimizer='adam', loss=focal_tversky_loss, metrics=metrics)

        model.summary()
        early_stopping = EarlyStopping(
            monitor ='val_loss',
            patience=10,
            restore_best_weights =True,
            verbose =1
        )

        lr_scheduler=ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience =5,
            verbose = 1,
            min_lr=1e-6
        )
        checkpoint = ModelCheckpoint(
            'best_rooftop_model.keras',
            monitor='val_jacard_coef',
            mode='max',
            save_best_only=True,
            verbose=1
        )

        callbacks = [early_stopping,lr_scheduler, checkpoint]

        history1 = model.fit(
            X_train, Y_train,
            batch_size=2,
            verbose=1,
            epochs=100,
            validation_data=(X_test, Y_test), 
            shuffle=False,
            callbacks=callbacks)

        model.save('.keras')
    #################################################
    if model==2:
        def get_model():
            return create_model()

        model = get_model()
        model.compile(optimizer='adam', loss=total_loss, metrics=metrics)

        model.summary()
        early_stopping = EarlyStopping(
            monitor ='val_loss',
            patience=10,
            restore_best_weights =True,
            verbose =1
        )

        lr_scheduler=ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience =5,
            verbose = 1,
            min_lr=1e-6
        )
        checkpoint = ModelCheckpoint(
            'best_rooftop_model.keras',
            monitor='val_jacard_coef',
            mode='max',
            save_best_only=True,
            verbose=1
        )

        callbacks = [early_stopping,lr_scheduler, checkpoint]

        history1 = model.fit(
            X_train, Y_train,
            batch_size=2,
            verbose=1,
            epochs=100,
            validation_data=(X_test, Y_test), 
            shuffle=False,
            callbacks=callbacks)

        model.save('final_xception_unet.keras')
    history = history1
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    acc = history.history['jacard_coef']
    val_acc = history.history['val_jacard_coef']

    plt.plot(epochs, acc, 'y', label='Training IoU')
    plt.plot(epochs, val_acc, 'r', label='Validation IoU')
    plt.title('Training and validation IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend()
    plt.show()
if training ==0:
    model = load_model('best_rooftop_model.keras',
                       custom_objects={'focal_tversky_loss': focal_tversky_loss,
                                   'jacard_coef':jacard_coef})
    print("model loaded succesfully")
    
     # Make predictions
    y_pred = model.predict(X_test)
    y_pred_argmax = np.argmax(y_pred, axis=3)
    y_test_argmax = np.argmax(Y_test, axis=3)

    # Compute Mean IoU
    from keras.metrics import MeanIoU
    print("Creating test_image data")
    test_image_dataset = formDataset("image", patch_size, root_directory="Data/resized/test"+str(size)+"/")
    test_label_dataset = formDataset("label", patch_size, root_directory="Data/resized/test"+str(size)+"/")
    # test_image_dataset = formDataset("image", patch_size, root_directory="Data/resized/mine"+str(size)+"/")
    # test_image_dataset = np.array(test_image_dataset)
    print(len(test_image_dataset))
    n_classes = 2
    IOU_keras = MeanIoU(num_classes=n_classes)
    IOU_keras.update_state(y_test_argmax, y_pred_argmax)
    print("Mean IoU =", IOU_keras.result().numpy())

    # Visualize a random test sample
    import random
    try:
        while True:
            test_img_number = random.randint(0, len(test_image_dataset) - 1)
            print(f"Showing prediction for test image #{test_img_number}")

            test_img = test_image_dataset[test_img_number]
            ground_truth = test_label_dataset[test_img_number]

            test_img_input = np.expand_dims(test_img, 0)
            prediction = model.predict(test_img_input)
            predicted_img = np.argmax(prediction, axis=3)[0, :, :]

            # Display results
            plt.figure(figsize=(18, 6))
            plt.subplot(1, 3, 1)
            plt.title('Testing Image')
            plt.imshow(test_img)

            plt.subplot(1, 3, 2)
            plt.title('Ground Truth')
            plt.imshow(ground_truth, cmap='gray')

            plt.subplot(1, 3, 3)
            plt.title('Predicted Mask')
            plt.imshow(predicted_img, cmap='gray')

            plt.tight_layout()
            plt.show()
    except:
        KeyboardInterrupt('q')
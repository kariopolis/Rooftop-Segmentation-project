import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from keras.utils import to_categorical
from keras.saving import register_keras_serializable
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow_datasets as tfds
import tensorflow_advanced_segmentation_models as tasm
from simple_multi_unet_model import multi_unet_model, jacard_coef
import numpy as np
from IPython.display import clear_output 
import matplotlib.pyplot as plt
from PIL import Image
import random
import os

root_directory = "Data/resized/"
patch_size=128
TRAIN_SIZE = 25
training =0

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
        print(dirname)
        if dirname==input_type:
            pics = sorted(os.listdir(path))
            print(root_directory, pics)
            num=0
            for num_of_pics, pic_name in enumerate(pics):
                if pic_name.endswith(".png"):
                    num+=1
                    pic = np.array(Image.open(path+'/'+pic_name))
                    if np.max(pic) >1:
                         pic = normalize(pic)
                    pic_dataset.append(pic)
                # if num>=TRAIN_SIZE*64:
                #     break
    return pic_dataset

image_dataset=formDataset("image",patch_size,root_directory+'train/')
print("image dataset_created")
label_dataset=formDataset("label",patch_size,root_directory+'train/')
print("label_dataset_created")
#print(label_dataset)

image_dataset = np.array(image_dataset)
labels_dataset =  np.array(label_dataset)

#print (np.unique(label_dataset))
    

# <----------------------Check dataset---------------------------->
#CheckDataset(image_dataset, label_dataset)

####################################################################

n_classes = len(np.unique(label_dataset))
label_cat = to_categorical(labels_dataset, num_classes=n_classes)

X_train,X_test,Y_train,Y_test=train_test_split(image_dataset,label_cat,test_size=0.2,random_state=42)

base_model=tf.keras.applications.MobileNetV2(input_shape=[128,128,3], include_top=False)


weights = [1.222,1.222]
#print(weights)

@register_keras_serializable(package="CustomLosses")
def total_loss(y_true, y_pred):
    dice_loss = tasm.losses.DiceLoss(class_weights=weights)
    focal_loss = tasm.losses.CategoricalFocalLoss()
    return dice_loss(y_true, y_pred) + focal_loss(y_true, y_pred)

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]

metrics = ['accuracy', jacard_coef]
if training==1:
    def get_model():
        return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

    model = get_model()
    model.compile(optimizer='adam', loss=total_loss, metrics=metrics)
    model.summary()

    history1 = model.fit(X_train, Y_train,batch_size=8,verbose=1,epochs=11,validation_data=(X_test, Y_test), shuffle=False)

    model.save('rooftop_segmentation_model.keras')

     ########################################################

    BACKBONE = 'resnet34'
    preprocess_input = tasm.get_preprocessing(BACKBONE)
    # preprocess input
    X_train_prepr = preprocess_input(X_train)
    X_test_prepr = preprocess_input(X_test)

    # define model
    model_resnet_backbone = tasm.Unet(BACKBONE, encoder_weights='imagenet', classes=n_classes, activation='softmax')

    # compile keras model with defined optimozer, loss and metrics
    #model_resnet_backbone.compile(optimizer='adam', loss=focal_loss, metrics=metrics)
    model_resnet_backbone.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

    print(model_resnet_backbone.summary())


    history2=model_resnet_backbone.fit(X_train_prepr, 
            Y_train,
            batch_size=10, 
            epochs=30,
            verbose=1,
            validation_data=(X_test_prepr, Y_test))
    
    model.save('rooftop_segmentation_model_cat_crossentropy.kera')
    ###########################################################

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
    model = load_model('rooftop_segmentation_model.keras',
                       custom_objects={'dice_loss_plus_2focal_loss': total_loss,
                                   'jacard_coef':jacard_coef})
    print("model loaded succesfully")
    
     # Make predictions
    y_pred = model.predict(X_test)
    y_pred_argmax = np.argmax(y_pred, axis=3)
    y_test_argmax = np.argmax(Y_test, axis=3)

    # Compute Mean IoU
    from keras.metrics import MeanIoU
    print("Creating test_image data")
    test_image_dataset = formDataset("image", patch_size, root_directory="Data/resized/test/image/image/")
    test_image_dataset = np.array(test_image_dataset)
    print(test_image_dataset)
    n_classes = 2
    IOU_keras = MeanIoU(num_classes=n_classes)
    IOU_keras.update_state(y_test_argmax, y_pred_argmax)
    print("Mean IoU =", IOU_keras.result().numpy())

    # Visualize a random test sample
    import random
    test_img_number = random.randint(0, len(X_test) - 1)
    print(f"Showing prediction for test image #{test_img_number}")
    
    test_img = test_image_dataset[0]
    #ground_truth = y_test_argmax[test_img_number]

    test_img_input = np.expand_dims(test_img, 0)
    prediction = model.predict(test_img_input)
    predicted_img = np.argmax(prediction, axis=3)[0, :, :]

    # Display results
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.title('Testing Image')
    plt.imshow(test_img)

    # plt.subplot(1, 3, 2)
    # plt.title('Ground Truth')
    # plt.imshow(ground_truth, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title('Predicted Mask')
    plt.imshow(predicted_img, cmap='gray')

    plt.tight_layout()
    plt.show()
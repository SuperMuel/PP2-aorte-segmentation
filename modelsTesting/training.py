# %%
import os
import matplotlib.pyplot as plt
import keras
from keras import models
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras


# SEED : un nombre entier utilisé pour initialiser le générateur de nombres aléatoires. 
# En définissant la même graine de générateur de nombres aléatoires, on peut s'assurer que les résultats sont reproductibles.

# BATCH_SIZE_TRAIN : taille des lots utilisée lors de l'entraînement du modèle.
# BATCH_SIZE_TEST : taille des lots utilisée lors du test ou de l'évaluation du modèle.
# IMAGE_HEIGHT : hauteur de l'image en pixels à laquelle toutes les images d'entrée seront redimensionnées
# IMAGE_WIDTH : largeur de l'image en pixels à laquelle toutes les images d'entrée seront redimensionnées
# IMG_SIZE : une tuple contenant la hauteur et la largeur de l'image à laquelle toutes les images d'entrée seront redimensionnées.

# data_dir_train : chemin du répertoire qui contient les données d'entraînement.
# data_dir_train_image : chemin du sous-répertoire qui contient les images d'entraînement.
# data_dir_train_mask : chemin du sous-répertoire qui contient les masques correspondants pour les images d'entraînement.
# data_dir_test : chemin du répertoire qui contient les données de test.
# data_dir_test_image : chemin du sous-répertoire qui contient les images de test.
# data_dir_test_mask : chemin du sous-répertoire qui contient les masques correspondants pour les images de test.

# NUM_TRAIN : le nombre total d'exemples d'entraînement.
# NUM_TEST : le nombre total d'exemples de test.

# NUM_OF_EPOCHS : nombre d'époques que le modèle va utiliser pour s'entraîner, 
#                 càd nombre de fois que le modèle va parcourir l'ensemble des données d'entraînement pendant l'entraînement.


# %%
############# CONSTANTS DEFINITION #############
################################################

SEED = 1000
BATCH_SIZE_TRAIN = 16        # tester differente val°
BATCH_SIZE_TEST = 16
NUM_OF_EPOCHS = 100

IMAGE_HEIGHT = 512          # tester differente val°
IMAGE_WIDTH = 512
IMG_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)



# %%
############# PATHS AND DIRECTORIES DEFINITION #############
############################################################

data_dir_train = 'data/slices/'
# The images should be stored under: "data/slices/img/"
data_dir_train_image = os.path.join(data_dir_train, 'img/')
# The images should be stored under: "data/slices/masks/"
data_dir_train_mask = os.path.join(data_dir_train, 'masks/')

data_dir_test = 'data/slices/'
# The images should be stored under: "data/slices/img/imgTest/"
data_dir_test_image = os.path.join(data_dir_test, 'imgTest/')
# The images should be stored under: "data/slices/masksTest/"
data_dir_test_mask = os.path.join(data_dir_test, 'masksTest/')


def dir_creation(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

dir_creation(data_dir_train)
dir_creation(data_dir_train_image)
dir_creation(data_dir_train_mask)

dir_creation(data_dir_test)
dir_creation(data_dir_test_image)
dir_creation(data_dir_test_mask)


# %%
################### FUNCTION DEFINITIONS ####################
#############################################################

def create_segmentation_generator_train(img_path, msk_path, BATCH_SIZE):
    datagen = ImageDataGenerator(rescale=1/255)
    
    img_generator = datagen.flow_from_directory(img_path, target_size=IMG_SIZE, class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, seed=SEED)
    msk_generator = datagen.flow_from_directory(msk_path, target_size=IMG_SIZE, class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, seed=SEED)
    return zip(img_generator, msk_generator)


def create_segmentation_generator_test(img_path, msk_path, BATCH_SIZE):
    datagen = ImageDataGenerator(rescale=1/255)
    
    img_generator = datagen.flow_from_directory(img_path, target_size=IMG_SIZE, class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, seed=SEED)
    msk_generator = datagen.flow_from_directory(msk_path, target_size=IMG_SIZE, class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, seed=SEED)
    return zip(img_generator, msk_generator)

train_generator = create_segmentation_generator_train(data_dir_train_image, data_dir_train_mask, BATCH_SIZE_TRAIN)
test_generator = create_segmentation_generator_test(data_dir_test_image, data_dir_test_mask, BATCH_SIZE_TEST)


def display(display_list):
    plt.figure(figsize=(15,15))
    
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]), cmap='gray')
    plt.show()

def show_dataset(datagen, num=1):
    for i in range(0,num):
        image,mask = next(datagen)
        display([image[0], mask[0]])  
        #  if on dataPreparation SLICE_X = True, SLICE_Y = True, SLICE_Z = True  then   0 -> x,  1 -> y  and 2 -> z 
    

# %%
################### EXEMPLE PLOT GENERATOR ####################
###############################################################

show_dataset(train_generator, 5)

#%%
################### COUNT NUMBER OF IMAGES FOR TRAINING AND TESTING ####################
########################################################################################

def count_images(directory):
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            count += 1
    return count

total_train_images = count_images("data/slices/img/img")
print("Nombre total d'images pour le training:", total_train_images)
total_test_images = count_images("data/slices/imgTest/img")
print("Nombre total d'images pour le test:", total_test_images)

NUM_TRAIN = total_train_images
NUM_TEST = total_test_images

# %%
################### U-NET DEFINITION - CONVOLUTIONS, MAX_POOLING... ####################
########################################################################################

def unet(n_levels, initial_features=32, n_blocks=2, kernel_size=3, pooling_size=2, in_channels=1, out_channels=1):
    inputs = keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, in_channels))
    x = inputs
    
    # definition of convolution parameters, passed as a parameter when used keras.layers.Conv2D
    convpars = dict(kernel_size=kernel_size, activation='relu', padding='same')
    
    #downstream
    skips = {}
    for level in range(n_levels):
        for _ in range(n_blocks): # number of convolutions in each level
            x = keras.layers.Conv2D(initial_features * 2 ** level, **convpars)(x)
        if level < n_levels - 1:
            skips[level] = x
            x = keras.layers.MaxPool2D(pooling_size)(x)
            
    # upstream
    for level in reversed(range(n_levels-1)):
        x = keras.layers.Conv2DTranspose(initial_features * 2 ** level, strides=pooling_size, **convpars)(x)
        x = keras.layers.Concatenate()([x, skips[level]])
        for _ in range(n_blocks):
            x = keras.layers.Conv2D(initial_features * 2 ** level, **convpars)(x)
            
    # output
    activation = 'sigmoid' if out_channels == 1 else 'softmax'
    x = keras.layers.Conv2D(out_channels, kernel_size=1, activation=activation, padding='same')(x)
    
    return keras.Model(inputs=[inputs], outputs=[x], name=f'UNET-L{n_levels}-F{initial_features}')


EPOCH_STEP_TRAIN = NUM_TRAIN // BATCH_SIZE_TRAIN
EPOCH_STEP_TEST = NUM_TEST // BATCH_SIZE_TEST

# %%
################### U-NET APPLICATION AND SAVING RESULTS ###############################
########################################################################################

model = unet(3)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()


# How to use model.fit or model.fit_generator --> https://keras.io/api/models/model_training_apis/

model.fit_generator(generator=train_generator, 
                    steps_per_epoch=EPOCH_STEP_TRAIN, 
                    validation_data=test_generator, 
                    validation_steps=EPOCH_STEP_TEST,
                    epochs=NUM_OF_EPOCHS,
                    shuffle=True)


model.save(f'UNET-{total_train_images}totalTrainImages_{total_test_images}totalTestImages_{IMAGE_HEIGHT}height_{IMAGE_WIDTH}width_{BATCH_SIZE_TRAIN}batch_{NUM_OF_EPOCHS}epochs.h5')

# %%
#### U-NET LOADING RESULTS IF U-NET ALREADY APPLIED BEFORE ####
###############################################################

# EXEMPLE  --> model = models.load_model(f'UNET-{total_train_images}totalTrainImages_{total_test_images}totalTestImages_{IMAGE_HEIGHT}height_{IMAGE_WIDTH}width_{BATCH_SIZE_TRAIN}batch_{NUM_OF_EPOCHS}epochs.h5')

#model=models.load_model('UNET-13666totalTrainImages_2016totalTestImages_64height_64width_10batch_30epochs.h5')
#model=models.load_model('UNET-13666totalTrainImages_2016totalTestImages_128height_128width_32batch_50epochs.h5')
#model=models.load_model('UNET-13666totalTrainImages_2016totalTestImages_512height_512width_16batch_30epochs.h5')
model=models.load_model('UNET-13666totalTrainImages_2016totalTestImages_512height_512width_16batch_100epochs.h5')

test_generator = create_segmentation_generator_test(data_dir_test_image, data_dir_test_mask, 1)

# %%
################### RESULTS ###############################
###########################################################

def show_prediction(datagen, num=1):
    for i in range(0,num):
        image,mask = next(datagen)
        pred_mask = model.predict(image)[0] > 0.5
        display([image[0], mask[0], pred_mask])



show_prediction(test_generator, 10)

# %%

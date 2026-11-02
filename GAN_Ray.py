import numpy as np
import os
import skimage.transform as trans
from keras.models import *
from keras.losses import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras.utils import *
from keras.objectives import binary_crossentropy
from keras import backend as K
from keras.applications.vgg16 import VGG16
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import cv2
import skimage.io as io
import natsort
#from utils import load_images
###############################Parameters######################################
# Data & model configuration
image_size=(256,256)
num_channels = 3
batch = 1
epoch_num=10
D_epochs = 2
GD_epochs = 2
train_images=2130
val_images=265
Training = True     #True= If want to train on new dataset
No_prior_weights = False #True= if any of discriminative/generative model weights are not available otherwise false
Train_sharp_image_path = '/home/rak038/Deblurring/Train/Sharp'
Train_blur_image_path = '/home/rak038/Deblurring/Train/Blur'
Val_sharp_image_path = '/home/rak038/Deblurring/Val/Blur'
Val_blur_image_path = '/home/rak038/Deblurring/Val/Sharp'
Test_image_path = '/home/rak038/Deblurring/Test/Test_blur/0/'
Test_save_path = "/home/rak038/Deblurring/Test/Test_out/"

################################# Models #################################
def generative(pretrained_weights_gen,input_shape):

    Inputs = Input(shape=input_shape, name='Generator_Input')
    conv1 = Conv2D(64, 7, activation = 'relu', padding = 'same', dilation_rate=(2,2), kernel_initializer = 'he_normal')(Inputs)
    conv1 = Conv2D(64, 7, activation = 'relu', padding = 'same', dilation_rate=(4,4), kernel_initializer = 'he_normal')(conv1)
    conv1 = Conv2D(64, 7, activation = 'relu', padding = 'same', dilation_rate=(8,8), kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', strides=(2,2),kernel_initializer = 'he_normal')(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', dilation_rate=(4,4),kernel_initializer = 'he_normal')(conv2)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', dilation_rate=(8,8),kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', strides=(2,2), kernel_initializer = 'he_normal')(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', dilation_rate=(4,4), kernel_initializer = 'he_normal')(conv3)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', dilation_rate=(8,8), kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
#############################Residual Blocks##############################
    x=conv3
    features=(256,512,1024)
    def res_block(x,fea):
        res_conv11 = Conv2D(fea, 3, activation = 'relu', padding = 'same', dilation_rate=(2,2), kernel_initializer = 'he_normal')(x)
        res_conv1 = Conv2D(fea, 3, activation = 'relu', padding = 'same', dilation_rate=(4,4), kernel_initializer = 'he_normal')(res_conv11)
        res_conv1 = Conv2D(fea, 3, activation = 'relu', padding = 'same', dilation_rate=(8,8), kernel_initializer = 'he_normal')(res_conv1)
        x = add([res_conv11,res_conv1])
        res = BatchNormalization()(x)
        return res
    for i in range(3):
        x = res_block(x,features[i])
############################################################################
    up1 = (UpSampling2D(size = (2,2)))(x)
    conv4 = Conv2DTranspose(256, 3, activation = 'relu', padding = 'same', dilation_rate=(2,2), kernel_initializer = 'he_normal')(up1)
    conv4 = Conv2DTranspose(256, 3, activation = 'relu', padding = 'same', dilation_rate=(4,4), kernel_initializer = 'he_normal')(conv4)
    conv4 = Conv2DTranspose(256, 3, activation = 'relu', padding = 'same', dilation_rate=(8,8), kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    
    up2 = (UpSampling2D(size = (2,2)))(conv4)
    conv5 = Conv2DTranspose(128, 3, activation = 'relu', padding = 'same', dilation_rate=(2,2), kernel_initializer = 'he_normal')(up2)
    conv5 = Conv2DTranspose(128, 3, activation = 'relu', padding = 'same', dilation_rate=(4,4), kernel_initializer = 'he_normal')(conv5)
    conv5 = Conv2DTranspose(128, 3, activation = 'relu', padding = 'same', dilation_rate=(8,8), kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    
    conv6 = Conv2DTranspose(64, 3, activation = 'relu', padding = 'same', dilation_rate=(2,2), kernel_initializer = 'he_normal')(conv5)
    conv6 = Conv2DTranspose(64, 3, activation = 'relu', padding = 'same', dilation_rate=(4,4), kernel_initializer = 'he_normal')(conv6)
    conv6 = Conv2DTranspose(64, 3, activation = 'relu', padding = 'same', dilation_rate=(8,8), kernel_initializer = 'he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    
    conv7 = Conv2D(3, 7, activation = 'tanh', padding = 'same', dilation_rate=(1,1), kernel_initializer = 'he_normal')(conv6)
    Outputs = add([conv7,Inputs])
    
    model = Model(inputs=Inputs, outputs=Outputs, name='Generator')
    model.summary()
    
    
    if(pretrained_weights_gen):
        model.load_weights(pretrained_weights_gen)
    return model

def discriminator(pretrained_weights_dis,input_shape):
    Inputs = Input(shape=input_shape, name='Discriminator_Input')
    
    #conv11 = Conv2D(32, 4, padding = 'same', strides=(2,2),kernel_initializer='glorot_uniform')(Inputs)
    #conv1 = LeakyReLU(0.1)(conv11)
    #conv1 = BatchNormalization()(conv1)
    #conv1 = Conv2D(32, 4, padding = 'same', strides=(1,1),kernel_initializer='glorot_uniform')(conv1)
    #conv1 = LeakyReLU(0.1)(conv1)
    #conv1 = BatchNormalization()(conv1)
    #conv1 = Conv2D(32, 4, padding = 'same', strides=(1,1),kernel_initializer='glorot_uniform')(conv1)
    #conv1 = LeakyReLU(0.1)(conv1)
    #conv1 = BatchNormalization()(conv1)
    #conv1 = add([conv11,conv1])

    conv21 = Conv2D(64, 4, padding = 'same', strides=(2,2),kernel_initializer = 'he_normal')(Inputs)
    conv2 = LeakyReLU(0.2)(conv21)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, 4, padding = 'same', dilation_rate=(4,4),kernel_initializer = 'he_normal')(conv2)
    conv2 = LeakyReLU(0.2)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, 4, padding = 'same', dilation_rate=(8,8),kernel_initializer = 'he_normal')(conv2)
    conv2 = LeakyReLU(0.2)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = add([conv21,conv2])
    #conv2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
    conv31 = Conv2D(128, 4, padding = 'same', strides=(2,2),kernel_initializer = 'he_normal')(conv2)
    conv3 = LeakyReLU(0.2)(conv31)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, 4, padding = 'same', dilation_rate=(4,4),kernel_initializer = 'he_normal')(conv3)
    conv3 = LeakyReLU(0.2)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, 4, padding = 'same', dilation_rate=(8,8),kernel_initializer = 'he_normal')(conv3)
    conv3 = LeakyReLU(0.2)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = add([conv31,conv3])

    conv41 = Conv2D(256, 4, padding = 'same', strides=(2,2),kernel_initializer = 'he_normal')(conv3)
    conv4 = LeakyReLU(0.2)(conv41)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, 4, padding = 'same', dilation_rate=(4,4),kernel_initializer = 'he_normal')(conv4)
    conv4 = LeakyReLU(0.2)(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, 4, padding = 'same', dilation_rate=(8,8),kernel_initializer = 'he_normal')(conv4)
    conv4 = LeakyReLU(0.2)(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = add([conv41,conv4])
    
    conv51 = Conv2D(512, 4, padding = 'same', strides=(2,2),kernel_initializer = 'he_normal')(conv4)
    conv5 = LeakyReLU(0.2)(conv51)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, 4, padding = 'same', dilation_rate=(4,4),kernel_initializer = 'he_normal')(conv5)
    conv5 = LeakyReLU(0.2)(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, 4, padding = 'same', dilation_rate=(8,8),kernel_initializer = 'he_normal')(conv5)
    conv5 = LeakyReLU(0.2)(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = add([conv51,conv5])
    
    #conv61 = Conv2D(512, 4, padding = 'same', strides=(1,1),kernel_initializer='glorot_uniform')(conv5)
    #conv6 = LeakyReLU(0.2)(conv61)
    #conv6 = BatchNormalization()(conv6)
    #conv6 = Conv2D(1024, 4, padding = 'same', dilation_rate=(4,4),kernel_initializer='glorot_uniform')(conv6)
    #conv6 = LeakyReLU(0.2)(conv6)
    #conv6 = BatchNormalization()(conv6)
    #conv6 = Conv2D(1024, 4, padding = 'same', dilation_rate=(8,8),kernel_initializer='glorot_uniform')(conv6)
    #conv6 = LeakyReLU(0.2)(conv6)
    #conv6 = BatchNormalization()(conv6)
    #conv6 = add([conv61,conv6])
       
    x = Flatten()(conv5)
    #x = Dense(4096, activation='sigmoid')(x)
    #x = BatchNormalization()(x)
    #x = Dense(1024, activation='sigmoid')(x)
    #x = BatchNormalization()(x)
    x = Dense(512, activation='tanh')(x)
    x = BatchNormalization()(x)
    Outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=Inputs, outputs=Outputs, name='Discriminator')
    model.summary()
    
    if(pretrained_weights_dis):
        model.load_weights(pretrained_weights_dis)
    return model

def generator_containing_discriminator(generator, discriminator):
    inputs = Input(shape = (image_size[0],image_size[1],num_channels))
    generated_image = generator(inputs)
    outputs = discriminator(generated_image)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def generator_containing_discriminator_multiple_outputs(generator, discriminator):
    inputs = Input(shape = (image_size[0],image_size[1],num_channels))
    generated_image = generator(inputs)
    outputs = discriminator(generated_image)
    model = Model(inputs=inputs, outputs=[generated_image, outputs])
    return model

def gan(generator, discriminator):
	# make weights in the discriminator not trainable
    discriminator.trainable = False
	# connect them
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
	# compile model
    return model

#################################Losses#########################################

def perceptual_loss(y_true, y_pred):
    #vgg = VGG16(include_top=False, weights='imagenet', input_shape=(image_size[0],image_size[1],num_channels))
    #loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    #loss_model.trainable = False
    #return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))
    #reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred)) * image_size[0] * image_size[1]
    mse_loss = K.mean(mse(y_true, y_pred), axis=(1,2)) * image_size[0] * image_size[1]
    return (mse_loss)

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)

def l1_loss(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))

def custom_loss(y_true, y_pred):
    # Below code convert image gradient in x&y direction
    dx_true= cv2.Sobel(y_true,0, dx=1,dy=0)
    dy_true= cv2.Sobel(y_true,0, dx=0,dy=1)
    
    dx_pred= cv2.Sobel(y_pred,0, dx=1,dy=0)
    dy_pred= cv2.Sobel(y_pred,0, dx=0,dy=1)

    return K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)


optimizer = Adam(lr=1E-4)
###################################Data Processing#############################

train_datagen = ImageDataGenerator(rescale = None,
                                   featurewise_center=False,
                                   featurewise_std_normalization=False,
                                   rotation_range=40,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.1,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=None)
################################################################################


True_Blur_Train_set = train_datagen.flow_from_directory('/home/rak038/Deblurring/Train/Blur',
                                                        target_size = image_size,
                                                        batch_size = 1,
                                                        class_mode = None,
                                                        color_mode= 'rgb',
                                                        seed = None)
True_Blur_Val_set = val_datagen.flow_from_directory('/home/rak038/Deblurring/Val/Blur',
                                                    target_size = image_size,
                                                    batch_size = 1,
                                                    class_mode = None,
                                                    color_mode= 'rgb',
                                                    seed = None)
##################################################################################


training=train_datagen.flow_from_directory('/home/rak038/Deblurring/Train',
                                            target_size = image_size,
                                            batch_size = 1,
                                            class_mode = "binary",
                                            color_mode= 'rgb',
                                            seed = None)

validation=val_datagen.flow_from_directory('/home/rak038/Deblurring/Val',
                                            target_size = image_size,
                                            batch_size = 1,
                                            class_mode = "binary",
                                            color_mode= 'rgb',
                                            seed = None)
###############################################################################
def trainGenerator(image_path,mask_path,size):
    train_image_generator = train_datagen.flow_from_directory(image_path,
                                                         batch_size = 1,
                                                         class_mode= None,
                                                         color_mode= 'rgb',
                                                         seed = None,
                                                         target_size=image_size)
    train_mask_generator = train_datagen.flow_from_directory(mask_path,
                                                         batch_size = 1,
                                                         class_mode= None,
                                                         color_mode= 'rgb',
                                                         seed = None,
                                                         target_size=image_size)
    train_generator = zip(train_image_generator, train_mask_generator)
    return (train_generator)

####################### VAlidation Dataset

def valGenerator(image_path,mask_path,size):
    val_image_generator = val_datagen.flow_from_directory(image_path,
                                                      batch_size = 1,
                                                      class_mode= None,
                                                      color_mode= 'rgb',
                                                      seed = None,
                                                      target_size=image_size)
    val_mask_generator = val_datagen.flow_from_directory(mask_path,
                                                     batch_size = 1,
                                                     class_mode= None,
                                                     color_mode= 'rgb',
                                                     seed = None,
                                                     target_size=image_size)
    val_generator = zip(val_image_generator, val_mask_generator)
    return (val_generator)

train_data=trainGenerator(size=image_size,
                          image_path = Train_blur_image_path,
                          mask_path = Train_sharp_image_path)

val_data=valGenerator(size=image_size,
                      image_path = Val_blur_image_path,
                      mask_path = Val_sharp_image_path )


########################## Test Output Images #################################
def testGenerator(test_path,target_size):
    ls=natsort.natsorted(os.listdir(test_path))
    for image_file_name in ls:
        if image_file_name.endswith(".jpg"):
            im = cv2.imread(test_path+image_file_name)
            img=cv2.resize(img,target_size)
            img = np.reshape(img,(1,)+img.shape)
            yield img

def saveResult(save_path,npyfile):
    for i,item in enumerate(npyfile):
        cv2.imwrite(os.path.join(save_path , '%d.jpg'%i),item)
###################################Train & Validate############################
'''Train and Validate first discriminator then Generative model'''
if Training:
    for iterations in range(epoch_num):
        
        if No_prior_weights:
            checkpoint = ModelCheckpoint('discriminator1.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')#.hdf5
            csv_logger = CSVLogger('./dis1.out', append=True, separator=';')#
            earlystopping = EarlyStopping(monitor = 'loss', 
                                          min_delta = 0.001, 
                                          patience = 1,
                                          mode = 'min',
                                          restore_best_weights=True)
            callbacks_list = [checkpoint, csv_logger, earlystopping]
            
            d = discriminator(None,(image_size[0],image_size[1],num_channels))
            d.trainable = True
            d.compile(optimizer = optimizer, loss=wasserstein_loss)
            
            d.fit_generator(training,
                            epochs=1,
                            steps_per_epoch = train_images,
                            validation_data = validation,
                            validation_steps = val_images,
                            callbacks = callbacks_list)

            g = generative(None,(image_size[0],image_size[1],num_channels))
            g.compile(optimizer = optimizer, loss='binary_crossentropy', metrics = ['accuracy'])
            checkpoint = ModelCheckpoint('generative1.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')#Tumor_ResU_rvd.hdf5
            csv_logger = CSVLogger('./gen1.out', append=True, separator=';')#./Tumor_result_rvd.out
            earlystopping = EarlyStopping(monitor = 'loss',
                                          min_delta = 0.001, 
                                          patience = 1,
                                          mode = 'min',
                                          restore_best_weights=True)
            callbacks_list = [checkpoint, csv_logger, earlystopping]
            
            g.fit_generator(train_data,epochs=1,
                            steps_per_epoch = train_images,
                            validation_data = val_data,
                            validation_steps = val_images,
                            callbacks = callbacks_list)
            
            No_prior_weights=False

        d = discriminator('discriminator1.h5',(image_size[0],image_size[1],num_channels))
        g = generative('generative1.h5',(image_size[0],image_size[1],num_channels))
        d.compile(optimizer = optimizer, loss=wasserstein_loss)
        d.trainable= True
        
        fake_train = g.predict_generator(True_Blur_Train_set,steps = np.ceil(train_images),verbose=1)
        fake_val = g.predict_generator(True_Blur_Val_set,steps = np.ceil(val_images),verbose=1)
        
        #print(fake_train.shape)

        out_train_fake_batch, out_val_fake_batch = np.zeros((len(True_Blur_Train_set), 1)), np.zeros((len(True_Blur_Val_set), 1))


        fake_val_gen=(fake_val,out_val_fake_batch)

        checkpoint = ModelCheckpoint('discriminator1.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')#.hdf5
        csv_logger = CSVLogger('./dis1.out', append=True, separator=';')#
        earlystopping = EarlyStopping(monitor = 'loss', 
                                      min_delta = 0.001, 
                                      patience = 2,
                                      mode = 'min',
                                      restore_best_weights=True)
        callbacks_list = [checkpoint, csv_logger, earlystopping]
        
        d.fit(x=fake_train,y=out_train_fake_batch,
                batch_size=batch,
                epochs=2,
                callbacks=callbacks_list,
                validation_data=fake_val_gen)

        d.save_weights('discriminator1.h5')
        
        d = discriminator('discriminator1.h5',(image_size[0],image_size[1],num_channels))
        d.compile(optimizer = optimizer, loss=wasserstein_loss)

        checkpoint = ModelCheckpoint('discriminator1.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')#.hdf5
        csv_logger = CSVLogger('./dis1.out', append=True, separator=';')#
        earlystopping = EarlyStopping(monitor = 'loss', 
                                      min_delta = 0.001, 
                                      patience = 2,
                                      mode = 'min',
                                      restore_best_weights=True)
        callbacks_list = [checkpoint, csv_logger, earlystopping]
        
        
        d.fit_generator(training,
                        epochs=D_epochs,
                        steps_per_epoch = train_images,
                        validation_data = validation,
                        validation_steps = val_images,
                        callbacks = callbacks_list)
        
        d.save_weights('discriminator.h5')
                    
        #################################Train Generative Model #######################
        d = discriminator('discriminator1.h5',(image_size[0],image_size[1],num_channels))
        g = generative('generative1.h5',(image_size[0],image_size[1],num_channels))
        d.trainable=False
        g.trainable=True
        d_on_g = gan(g, d)
        d_on_g.compile(optimizer=optimizer, loss= wasserstein_loss)
        csv_logger = CSVLogger('./dis_gen1.out', append=True, separator=';')
        checkpoint = ModelCheckpoint('dis_gen1.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')
        earlystopping = EarlyStopping(monitor = 'loss', verbose = 1,
                                      min_delta = 0.001, patience = 2,
                                      mode = 'min',
                                      restore_best_weights=True)
        
        callbacks_list = [csv_logger, checkpoint, earlystopping]
        
        d_on_g.fit_generator(training,
                             epochs=GD_epochs,
                             steps_per_epoch = train_images,
                             validation_data=validation,
                             validation_steps= val_images,
                             callbacks = callbacks_list)
    
        g.save_weights('generative1.h5')
        d.trainable= True

g = generative('generative1.h5',(image_size[0],image_size[1],num_channels))
g.trainable=False
'''
g.compile(optimizer = optimizer, loss='binary_crossentropy', metrics = ['accuracy'])
csv_logger = CSVLogger('./gen1.out', append=True, separator=';')#./Tumor_result_rvd.out
callbacks_list = [csv_logger]
            
g.fit_generator(train_data,epochs=D_epochs,
                steps_per_epoch = train_images,
                validation_data = val_data,
                validation_steps = val_images,
                callbacks = callbacks_list)
'''
#################################Train Generative Model #######################
test_generator = testGenerator(test_path = Test_image_path,target_size=image_size)

results = g.predict_generator(test_generator,265,verbose=1)

saveResult(Test_save_path,results)
##############################################################################

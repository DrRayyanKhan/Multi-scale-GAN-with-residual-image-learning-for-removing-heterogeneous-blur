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
import tensorflow as tf
import tensorlayer as tl
from keras.preprocessing.image import ImageDataGenerator
import cv2
import skimage.io as io
import natsort
from PIL import Image
from tensorflow.python.ops import io_ops
from tensorflow.keras.applications.vgg19 import VGG19,preprocess_input
import math
#import tensorflow_addons as tfa
from keras.regularizers import l2
#from utils import load_images
###############################Parameters######################################
# Data & model configuration
image_size=(256,256)
num_channels = 3
batch_size = 16
epoch_num=60
train_images=1635
val_images=468
batches = int(train_images/batch_size)
validation_steps=int(val_images/batch_size)
Training = True     #True= If want to train on new dataset
critic_updates =1
No_prior_weights = False #True= if any of discriminative/generative model weights are not available otherwise false
Train_sharp_image_path = '/home/rak038/Deblurring/Traingropro/1'           #Sharp
Train_blur_image_path = '/home/rak038/Deblurring/Traingropro/0'            #Blur
Val_sharp_image_path = '/home/rak038/Deblurring/Valgropro/1'
Val_blur_image_path = '/home/rak038/Deblurring/Valgropro/0'
Test_image_path = '/home/rak038/Deblurring/Testgropro/Test_blur/0/'
Test_save_path = "/home/rak038/Deblurring/Testgropro/Test_out/"

################################# Models #################################
def generative(pretrained_weights_gen,input_shape):

    Inputs = Input(shape=input_shape, name='Generator_Input')
    
    lambda_grad =  Lambda(lambda x:tf.image.sobel_edges(x))(Inputs)
    I_edge =  Lambda(lambda x:tf.sqrt(tf.math.square(x[:,:,:,:,0])+tf.math.square(x[:,:,:,:,1])))(lambda_grad)
    #I_con = Concatenate(axis=3)([Inputs,I_edge])
    I_sub= Subtract()([Inputs,I_edge])
    '''
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', dilation_rate=(1,1))(I_sub)
    #conv1 = Dropout(0.1)(conv1)
    #conv1 = BatchNormalization()(conv1)
    conv1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #conv1 = Add()([conv11,conv1])

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', dilation_rate=(1,1))(conv1)
    #conv2 = Dropout(0.1)(conv2)
    #conv2 = BatchNormalization()(conv2)
    conv2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #conv2 = Add()([conv2,conv21])
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', dilation_rate=(1,1))(conv2)
    #conv3 = Dropout(0.1)(conv3)
    #conv3 = BatchNormalization()(conv3)
    conv3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    #conv3 = Add()([conv3,conv31])
#############################Residual Blocks##########################
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', dilation_rate=(1,1))(conv3)
    #conv4 = Dropout(0.1)(conv4)
    conv4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    #conv4 = BatchNormalization()(conv4)
    #conv4 = Add()([conv4,conv41])
    
    x = Conv2D(16, 1, padding = 'same', strides=(1,1))(conv4)
    #x = Dropout(0.1)(x)
    #x = BatchNormalization()(conv)
    def res_block(x):
        conv_shape = K.int_shape(x)
        res_conv11 = Conv2D(16, 3, activation = 'relu',padding = 'same', dilation_rate=(2,2))(x)
        res_conv1 = Conv2D(16, 3, padding = 'same', dilation_rate=(8,8))(res_conv11)
        GP = AveragePooling2D(pool_size=(2, 2))(res_conv1)
        GP = Flatten()(GP)
        x1 = Dense(4096, activation='relu')(GP)
        x1 = Dense(4096, activation='relu')(x1)
        x1 = Reshape(x.shape.as_list()[1:])(x1)
        x2 = Add()([x1,res_conv11])
        x = Add()([x,x2])
        return x
    for i in range(6):
        x = res_block(x)
#######################################################################
    conv0 = Conv2D(512, 3, activation = 'relu',padding = 'same', dilation_rate=(2,2))(UpSampling2D(size = (2,2))(x))
    #conv0 = Dropout(0.1)(conv0)
    #conv0 = BatchNormalization()(conv0)
    #conv0 = Add()([conv0,conv01])

    conv5 = Conv2D(256, 3, activation = 'relu',padding = 'same', dilation_rate=(2,2))(UpSampling2D(size = (2,2))(conv0))
    #conv5 = Dropout(0.1)(conv5)
    #conv5 = BatchNormalization()(conv5)
    #conv5 = Add()([conv5,conv51])

    conv6 = Conv2D(128, 3, activation = 'relu',padding = 'same', dilation_rate=(2,2))(UpSampling2D(size = (2,2))(conv5))
    #conv6 = Dropout(0.1)(conv6)
    #conv6 = BatchNormalization()(conv6)
    #conv6 = Add()([conv6,conv61])
    
    conv7 = Conv2D(64, 3, activation = 'relu',padding = 'same', dilation_rate=(2,2))(UpSampling2D(size = (2,2))(conv6))
    #conv7 = Dropout(0.1)(conv7)
    conv8 = Conv2D(3, 1, activation = 'tanh',padding = 'same', dilation_rate=(1,1))(conv7)

    Outputs = Add()([conv8, Inputs])
    
    #conv9 = Conv2D(3, 1,padding = 'same', dilation_rate=(1,1))(conv9)
    '''
    
    conv1 = Conv2D(64, 7, activation = 'relu', padding = 'same', dilation_rate=(1,1))(I_edge)
    conv1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', dilation_rate=(2,2))(conv1)
    conv2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', dilation_rate=(4,4))(conv2)
    x = MaxPooling2D(pool_size=(2, 2))(conv3)
        
    def res_dense_block(x):
        conv11 = Conv2D(256, 1, padding = 'same', dilation_rate=(2,2))(x)
        conv1 = Conv2D(256, 3, activation = 'relu', padding = 'same', dilation_rate=(4,4))(conv11)
        conv2 = Conv2D(256, 1, padding = 'same', dilation_rate=(8,8))(conv1)
        x2 = Add()([conv2,conv11])
        x = Concatenate(axis=3)([x2,x])
        return x
    for i in range(6):
        x = res_dense_block(x)
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', dilation_rate=(1,1))(UpSampling2D(size = (2,2))(x))
    
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', dilation_rate=(2,2))(UpSampling2D(size = (2,2))(conv3))
    
    conv5 = Conv2D(64, 3, activation = 'relu', padding = 'same', dilation_rate=(4,4))(UpSampling2D(size = (2,2))(conv4))
    
    conv6 =  Conv2D(3, 1, activation = 'sigmoid', padding = 'same', dilation_rate=(1,1))(conv5)
    Outputs = Add()([conv6,Inputs])

    model = Model(inputs=Inputs, outputs=Outputs, name='Generator')
    
    model.summary()
    
    if(pretrained_weights_gen):
        model.load_weights(pretrained_weights_gen)
    return model

def discriminator(pretrained_weights_dis,input_shape):
    Inputs = Input(shape=input_shape, name='Discriminator_Input')
    
    conv21 = Conv2D(64, 4, padding = 'same', strides=(2,2))(Inputs)
    conv2 = LeakyReLU(0.2)(conv21)        
    conv31 = Conv2D(128, 4, padding = 'same', strides=(2,2))(conv2)
    conv3 = BatchNormalization()(conv31)
    #conv3 = tfa.layers.InstanceNormalization()(conv31)
    conv3 = LeakyReLU(0.2)(conv3)
    conv41 = Conv2D(256, 4, padding = 'same', strides=(2,2))(conv3)
    conv4 = BatchNormalization()(conv41)
    #conv4 = tfa.layers.InstanceNormalization()(conv41)
    conv4 = LeakyReLU(0.2)(conv4)
    conv4 = BatchNormalization()(conv4)
    #conv4 = tfa.layers.InstanceNormalization()(conv4)
    conv51 = Conv2D(512, 4, padding = 'same', dilation_rate=(4,4))(conv4)
    conv5 = BatchNormalization()(conv51)
    #conv5 = tfa.layers.InstanceNormalization()(conv51)
    conv5 = LeakyReLU(0.2)(conv5)
    conv6 = Conv2D(1, 4, padding = 'same', dilation_rate=(8,8))(conv5) 
    x = Flatten()(conv6)
    x = Dense(4096)(x)
    Outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=Inputs, outputs=Outputs, name='Discriminator')
    model.summary()
     
    if(pretrained_weights_dis):
        model.load_weights(pretrained_weights_dis)
    return model

def generator_containing_discriminator(generator, discriminator):
    inputs = Input(shape = (image_size[0],image_size[1],num_channels))
    discriminator.trainable=False
    generator.trainable=True
    generated_image = generator(inputs)
    outputs = discriminator(generated_image)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def generator_containing_discriminator_multiple_outputs(generator, discriminator):
    inputs = Input(shape = (image_size[0],image_size[1],num_channels))
    discriminator.trainable=False
    generator.trainable=True
    generated_image = generator(inputs)
    outputs = discriminator(generated_image)
    model = Model(inputs=inputs, outputs=[generated_image, outputs])
    return model

def gan(generator, discriminator):
	# make weights in the discriminator not trainable
    discriminator.trainable = False
    generator.trainable=True
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
	# compile model
    return model

#################################Losses#########################################
#Load VGG model
#vgg = VGG19(include_top=False, weights='imagenet', input_shape = [image_size[0],image_size[1],num_channels])
#vgg.trainable = False
#lossModel = Model([vgg.input], vgg.get_layer('pool4').output, name = 'vggL')

#@tf.function()
def perceptual_loss(y_true, y_pred):
    alpha = 0.1
    beta = 0.01
    #loss_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(image_size[0],image_size[1],num_channels))
    #loss_model.trainable = False
    #dense_loss= K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))
    #y_true1 = tf.reshape(y_true,[32,32,3])
    #y_pred1 = tf.reshape(y_pred,[32,32,3])
    #loss_vgg = tf.reduce_mean(tf.square(lossModel(preprocess_input(y_true1*255)) - lossModel(preprocess_input(y_pred1*255))))
    #reconstruction_loss = binary_crossentropy(y_true, y_pred)
    trueLap = tf.image.sobel_edges(y_true)
    predLap = tf.image.sobel_edges(y_pred)
    grad_loss = K.mean(K.abs(trueLap - predLap))
    l2_loss = MSE(y_true, y_pred)
    l1_loss = MAE(y_true, y_pred)
    Total_loss= K.mean((1-alpha-beta)*l1_loss + alpha*l2_loss + beta*grad_loss)
    return Total_loss

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)

optimizer = Adam(lr=1E-5)#,beta_1=0.9, beta_2=0.999, epsilon=1e-08)
###################################Data Processing#############################

train_datagen = ImageDataGenerator(rescale = None)

val_datagen = ImageDataGenerator(rescale=None)
################################################################################

True_Sharp_Train_set = train_datagen.flow_from_directory(Train_sharp_image_path,
                                                        target_size = image_size,
                                                        class_mode = None,
                                                        batch_size=batch_size,
                                                        color_mode= 'rgb',
                                                        seed = None)
True_Sharp_Val_set = val_datagen.flow_from_directory(Val_sharp_image_path,
                                                    target_size = image_size,
                                                    class_mode = None,
                                                    batch_size=batch_size,
                                                    color_mode= 'rgb',
                                                    seed = None)
True_Blur_Train_set = train_datagen.flow_from_directory(Train_blur_image_path,
                                                        target_size = image_size,
                                                        class_mode = None,
                                                        batch_size=batch_size,
                                                        color_mode= 'rgb',
                                                        seed = None)
True_Blur_Val_set = val_datagen.flow_from_directory(Val_blur_image_path,
                                                    target_size = image_size,
                                                    class_mode = None,
                                                    batch_size=batch_size,
                                                    color_mode= 'rgb',
                                                    seed = None)

out_train_true_batch,out_val_true_batch = np.ones((batch_size,1)),np.ones((batch_size,1))

train_data = zip(True_Blur_Train_set, True_Sharp_Train_set)
val_data = zip(True_Blur_Val_set, True_Sharp_Val_set)
num_array_true_sharp_train=True_Sharp_Train_set.next()
num_array_true_blur_train=True_Blur_Train_set.next()
num_array_true_sharp_val=True_Sharp_Val_set.next()
num_array_true_blur_val=True_Blur_Val_set.next()
##################################################################################
training=train_datagen.flow_from_directory('/home/rak038/Deblurring/Train01',
                                            target_size=image_size,
                                            color_mode="rgb",
                                            classes=['0','1'],
                                            class_mode='binary',
                                            batch_size = batch_size,
                                            seed=None)
validation=val_datagen.flow_from_directory('/home/rak038/Deblurring/Val',
                                            target_size=image_size,
                                            color_mode="rgb",
                                            classes=['0','1'],
                                            class_mode='binary',
                                            batch_size = batch_size,
                                            seed=None)
########################## Test Output Images #################################
def testGenerator(test_path,target_size):
    ls=natsort.natsorted(os.listdir(test_path))
    for image_file_name in ls:
        if image_file_name.endswith(".png"):
            im = cv2.imread(test_path+image_file_name)
            img=cv2.resize(im,target_size)
            img = np.reshape(img,(1,)+img.shape)
            yield img

def saveResult(save_path,npyfile):
    for i,item in enumerate(npyfile):
        cv2.imwrite(os.path.join(save_path ,'%d.png'%i),item)
###################################Train & Validate############################
'''Train and Validate first discriminator then Generative model'''
if Training:

    for iterations in range(epoch_num):
        
        if No_prior_weights:
            '''
            d = discriminator(None,(image_size[0],image_size[1],num_channels))
            d.trainable = True
            d.compile(optimizer = optimizer, loss = wasserstein_loss, metrics = ['accuracy'])
            checkpoint = ModelCheckpoint('discriminator1.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')#.hdf5
            #csv_logger = CSVLogger('./dis.out', append=True, separator=';')#
            earlystopping = EarlyStopping(monitor = 'loss',
                                            min_delta = 0,
                                            patience = 1,
                                            mode = 'min',
                                            restore_best_weights=True)
            callbacks_list = [checkpoint,  earlystopping]
            d.fit_generator(training,
                            steps_per_epoch = batches,
                            epochs=20,
                            validation_data = None,
                            callbacks = callbacks_list)
            '''
            loss=[perceptual_loss]
            g = generative(None,(image_size[0],image_size[1],num_channels))
            g.trainable = True
            g.compile(optimizer = optimizer, loss=loss, metrics = ['accuracy'])
            checkpoint = ModelCheckpoint('generative1.h5', monitor='loss', verbose=1, save_best_only=True, mode = 'min')#.hdf5
            #csv_logger = CSVLogger('./gen.out', append=True, separator=';')#
            earlystopping = EarlyStopping(monitor = 'loss',
                                            min_delta = 0,
                                            patience = 0,
                                            mode = 'min',
                                            restore_best_weights=True)
            callbacks_list = [checkpoint, earlystopping]

            g.fit_generator(train_data,epochs=1,
                            steps_per_epoch = batches,
                            validation_data = None,
                            callbacks = callbacks_list)
            
            No_prior_weights=False
        
        d = discriminator('discriminator1.h5',(image_size[0],image_size[1],num_channels))
        g = generative('generative1.h5',(image_size[0],image_size[1],num_channels))
        d.compile(optimizer = optimizer, loss = wasserstein_loss)
        d.trainable= True
        True_Blur_Train_set.reset()
        fake_train = g.predict(True_Blur_Train_set)
        #fake_val = g.predict_generator(True_Blur_Val_set,steps = np.ceil(val_images),verbose=1)
        #print(fake_train.shape)
        out_train_fake_batch = np.zeros((train_images, 1))
        #fake_val_gen=(fake_val,out_val_fake_batch)
        for _ in range(critic_updates):
            #checkpoint = ModelCheckpoint('discriminator1.h5', monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')#.hdf5
            #csv_logger = CSVLogger('./dis.out', append=True, separator=';')#
            earlystopping = EarlyStopping(monitor = 'loss',
                                            min_delta = 0,
                                            patience = 1,
                                            mode = 'min',
                                            restore_best_weights = True)
            callbacks_list = [earlystopping]
            
            d.fit(x=num_array_true_sharp_train,y=out_train_true_batch,
                  epochs=1,
                  steps_per_epoch = batches,
                  validation_data=None,
                  callbacks = callbacks_list)
            
            #checkpoint = ModelCheckpoint('discriminator1.h5', monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')#.hdf5
            #csv_logger = CSVLogger('./dis.out', append = True, separator = ';')#
            earlystopping = EarlyStopping(monitor = 'loss',
                                            min_delta = 0,
                                            patience = 1,
                                            mode = 'min',
                                            restore_best_weights=True)
            callbacks_list = [earlystopping]

            d.fit(x=fake_train,y=out_train_fake_batch,
                  epochs=1,
                  batch_size=batch_size,
                  validation_data=None,
                  callbacks = callbacks_list)
        #################################Train Generative Model #######################
        d.trainable= False
        d_on_g = generator_containing_discriminator_multiple_outputs(g, d)
        d_on_g.compile(optimizer=optimizer, loss = [perceptual_loss,wasserstein_loss],metrics = ['accuracy'])
        
        csv_logger = CSVLogger('./dis_gen.out', append=True, separator=';')#
        earlystopping = EarlyStopping(monitor = 'loss',
                                      min_delta = 0,
                                      patience = 0,
                                      mode = 'min',
                                      restore_best_weights=True)
        callbacks_list = [csv_logger, earlystopping]
        d_on_g.fit(x=num_array_true_blur_train,y=[num_array_true_sharp_train,out_train_true_batch],
                  epochs=1,
                  steps_per_epoch = batches,
                  validation_data=None,
                  callbacks = callbacks_list)

        print('Epoch',iterations)
        g.save_weights('generative1.h5')
        d.save_weights('discriminator1.h5')
        d.trainable= True
        
g = generative('generative1.h5',(image_size[0],image_size[1],num_channels))
g.trainable=False
'''
g.compile(optimizer = optimizer, loss=custom_loss, metrics = ['accuracy'])
checkpoint = ModelCheckpoint('generative.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')#.hdf5
csv_logger = CSVLogger('./gen.out', append=True, separator=';')#
earlystopping = EarlyStopping(monitor = 'loss', 
                            min_delta = 0.001, 
                            patience = 3,
                            mode = 'min')
callbacks_list = [checkpoint, csv_logger, earlystopping]
                                        
g.fit_generator(train_data,
                epochs=100,
                steps_per_epoch = train_images,
                validation_data = val_data,
                validation_steps = val_images,
                callbacks=callbacks_list)
#g.save_weights('generative1.h5')
'''
#################################Train Generative Model #######################
test_generator = testGenerator(test_path = Test_image_path,target_size=image_size)

results = g.predict_generator(test_generator,1111,verbose=1)#,263,verbose=1)

saveResult(Test_save_path,results)
##############################################################################

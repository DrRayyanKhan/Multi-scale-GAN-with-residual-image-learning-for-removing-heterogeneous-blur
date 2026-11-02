###############################################
from enhancement import *
from data_TVT import *
from keras.callbacks import *
from keras.utils import *
#################################################
parallel_model = enhancement(None,(image_size[0],image_size[1],num_channels))

def com_loss():
    def COM(y_true,y_pred):
        l1_loss = MAE(lossModel(y_true),lossModel(y_pred))
        #rvd=tf.abs(K.sum(tf.cast(y_true,tf.float32) - tf.cast(y_pred,tf.float32))/(K.sum(tf.cast(y_true,tf.float32)+1)))
        #dice=(1-(2*K.sum(y_true*y_pred)/(K.sum(y_true) + K.sum(y_pred))))
        #com=(rvd+dice)
        return l1_loss
    return COM
loss_fun=com_loss()
#parallel_model = multi_gpu_model(model, gpus=2)

parallel_model.compile(optimizer = Adam(lr = 1e-4), loss = loss_fun, metrics = ['accuracy'])#'binary_crossentropy''accuracy'


checkpoint = ModelCheckpoint('enhancement_seg.h5', monitor='loss',
                                verbose=1, save_best_only=True, mode='min')#Tumor_ResU_rvd.hdf5

#reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2,
 #                               min_lr=0.00001, min_delta=0.0001, mode='min', verbose=1)

csv_logger = CSVLogger('result', append=True, separator=';')#./Tumor_result_rvd.out

earlystopping = EarlyStopping(monitor = 'loss', verbose = 1,
                                min_delta = 0.0001, patience = 10, mode = 'min',restore_best_weights=True)

callbacks_list = [checkpoint, csv_logger, earlystopping]

train_data=trainGenerator(size=image_size,batch=batch_size,image_path='/globalhome/rak038/HPC/Enhancement/Train/0/0/',mask_path=Train_sharp_image_path)
val_data=valGenerator(size=image_size,batch=batch_size,image_path='/globalhome/rak038/HPC/Enhancement/Train/0/0/',mask_path=Train_sharp_image_path)
results = parallel_model.fit_generator(train_data,
        epochs=2000,
        steps_per_epoch = (train_images/batch_size), #(NO_OF_TRAINING_IMAGES//BATCH_SIZE),
        validation_data=val_data,
        validation_steps= (train_images/batch_size),  #(NO_OF_VAL_IMAGES//BATCH_SIZE), 
        callbacks=callbacks_list)


test_generator = testGenerator(test_path=Test_image_path,target_size=image_size)
results = parallel_model.predict_generator(test_generator,test_images,verbose=1)
saveResult(Test_save_path_1,results)

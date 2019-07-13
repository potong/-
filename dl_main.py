from keras.callbacks import ModelCheckpoint, EarlyStopping
from dl_model import *


combine_weights_file="weights/combine_weights.hdf5"

def esim_train(X1,X2,Y,i):
    model = esim()
    model.summary()
    esim_weights_file = "weights/esim_weights"+str(i)+".hdf5"
    def get_callbacks(filepath, patience=2):
        es = EarlyStopping('val_loss', patience=patience, mode="min")
        msave = ModelCheckpoint(filepath, monitor='val_loss',save_best_only=True)
        return [es, msave]
    callbacks = get_callbacks(filepath=esim_weights_file, patience=5)
    model.fit(x=[X1, X2], y=Y, batch_size=1024, validation_split=0.05, epochs=35, verbose=1, callbacks=callbacks)
def esim_predict(test_x1,test_x2,i):
    model = esim()
    esim_weights_file = "weights/esim_weights" + str(i) + ".hdf5"
    model.load_weights(filepath=esim_weights_file)
    result = model.predict([test_x1, test_x2])
    return result
def decomosable_train(X1,X2,Y,i):
    model = decomposable_attention()
    model.summary()
    file_path = "weights/decom_weights_test"+str(i)+".hdf5"
    def get_callbacks(filepath, patience=2):
        es = EarlyStopping('val_loss', patience=patience, mode="min")
        msave = ModelCheckpoint(filepath, monitor='val_loss',save_best_only=True)
        return [es, msave]
    callbacks = get_callbacks(filepath=file_path, patience=10)
    model.fit(x=[X1, X2], y=Y, batch_size=1024, validation_split=0.05, epochs=35, verbose=1, callbacks=callbacks)
def decom_predict(test_x1,test_x2,i):
    model = decomposable_attention()
    file_path = "weights/decom_weights_test" + str(i) + ".hdf5"
    model.load_weights(filepath=file_path)
    result = model.predict([test_x1, test_x2])
    return result

def combine_train(X1,X2,Y):
    model = combine_()
    model.summary()
    file_path = combine_weights_file
    def get_callbacks(filepath, patience=2):
        es = EarlyStopping('val_loss', patience=patience, mode="min")
        msave = ModelCheckpoint(filepath, monitor='val_loss',save_best_only=True)
        return [es, msave]
    callbacks = get_callbacks(filepath=file_path, patience=10)
    model.fit(x=[X1, X2], y=Y, batch_size=256, validation_split=0.05, epochs=35, verbose=1, callbacks=callbacks)
def combine_predict(test_x1,test_x2):
    model = combine_()
    model.load_weights(filepath=combine_weights_file)
    result = model.predict([test_x1, test_x2])
    return result


from sklearn.cross_validation import cross_val_score
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

model = Sequential()
model.add(Dense(8, input_shape=(X_train.shape[1],)))
model.add(Activation('relu'))
model.add(Dense(7))
model.add(Activation('relu'))
model.add(Dense(6))
model.add(Activation('relu'))
model.add(Dense(5))
model.add(Activation('relu'))
model.add(Dense(4))
model.add(Activation('relu'))
model.add(Dense(3))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

my_optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=my_optimizer, loss='binary_crossentropy')
early_stopping_monitor = EarlyStopping(patience=6)
model.fit(X_train, y, validation_split=0.2, epochs=50, batch_size=16392, callbacks=[early_stopping_monitor])
y_pred = model.predict(X_train)
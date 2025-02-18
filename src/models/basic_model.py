from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Rescaling, Flatten
from tensorflow.keras.optimizers import RMSprop, Adam

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        self.model = Sequential([
            Rescaling(1./255, input_shape=input_shape),  # Normalize input pixels

            layers.Conv2D(8, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D(pool_size=(2,2)),

            layers.Conv2D(16, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D(pool_size=(2,2)),

            layers.Conv2D(32, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D(pool_size=(2,2)),

            layers.Conv2D(64, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D(pool_size=(2,2)),

            Flatten(),
            layers.Dense(16, activation='relu'),  # Fully connected layer with reasonable params
            layers.Dense(categories_count, activation='softmax')  # Output layer
        ])
    
    def _compile_model(self):
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

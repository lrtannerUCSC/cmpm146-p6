from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.optimizers import Adam

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        self.model = Sequential([
            Rescaling(1./255, input_shape=input_shape),  # Normalize pixel values to [0,1]
            
            layers.Conv2D(32, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D(pool_size=(2,2)),

            layers.Conv2D(32, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D(pool_size=(2,2)),

            layers.Conv2D(48, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D(pool_size=(2,2)),

            layers.Flatten(),
            layers.Dense(8, activation='relu'),
            layers.Dense(categories_count, activation='softmax')  # Output layer
        ])
    
    def _compile_model(self):
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

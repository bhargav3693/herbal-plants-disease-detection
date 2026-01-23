import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import json

class MED117TransferModel:
    def __init__(self, num_classes=114, img_size=224):
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = self.build_model()
    
    def build_model(self):
        print("📦 Loading pre-trained MobileNetV2...")
        
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size, self.img_size, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom layers
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model created successfully!")
        return model
    
    def train(self, train_dir, test_dir, epochs=15, batch_size=32):
        print("\n📸 Preparing image data generators...")
        
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2
        )
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        callbacks = [
            ModelCheckpoint(
                'models/med117_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        print("="*70)
        print("🚀 Starting Transfer Learning Training")
        print("="*70)
        print(f"📊 Training samples: {train_generator.samples}")
        print(f"📊 Testing samples: {test_generator.samples}")
        print(f"📊 Number of classes: {len(train_generator.class_indices)}")
        print(f"📊 Batch size: {batch_size}")
        print(f"📊 Epochs: {epochs}")
        print(f"⏰ Estimated time: 30-45 minutes")
        print("="*70)
        
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=test_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save class names
        class_names = {v: k for k, v in train_generator.class_indices.items()}
        class_names_list = [class_names[i] for i in range(len(class_names))]
        
        with open('models/class_names.json', 'w') as f:
            json.dump(class_names_list, f, indent=2)
        
        print("="*70)
        print("✅ Training Complete!")
        print(f"📁 Model saved: models/med117_model.h5")
        print(f"📁 Class names saved: models/class_names.json")
        print("="*70)
        
        return history

if __name__ == "__main__":
    model = MED117TransferModel(num_classes=114)
    history = model.train(
        train_dir='data/raw/train',
        test_dir='data/raw/test',
        epochs=15,
        batch_size=32
    )
    
    print("\n✅ Model training completed successfully!")
    print("📌 Next step: streamlit run app.py")

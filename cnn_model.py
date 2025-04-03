from pyspark.sql import functions as F
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from tensorflow.keras import regularizers

class CNNModel:
    def __init__(self):
        self.model = None
        self.feature_stats = None
        
    def build_model(self, input_dim):
        """
        Build a CNN model with proper initialization and regularization
        """
        model = Sequential([
            # First Conv block
            Conv1D(filters=32, kernel_size=3, activation='relu', padding='same',
                   kernel_initializer='glorot_uniform', input_shape=(input_dim, 1),
                   kernel_regularizer=regularizers.l2(0.001)),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            
            # Second Conv block
            Conv1D(filters=64, kernel_size=3, activation='relu', padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(0.001)),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            
            # Flatten and dense layers
            Flatten(),
            Dense(64, activation='relu', kernel_initializer='glorot_uniform',
                  kernel_regularizer=regularizers.l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Output layer (binary classification)
            Dense(1, activation='sigmoid')
        ])
        
        # Use a more moderate learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',  # Standard binary loss
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model
    
    def normalize_features(self, X):
        """
        Apply feature normalization
        """
        if self.feature_stats is None:
            # Calculate mean and std for each feature
            self.feature_stats = {
                'mean': np.mean(X, axis=0),
                'std': np.std(X, axis=0)
            }
            # Handle zeros in std to avoid division by zero
            self.feature_stats['std'] = np.where(self.feature_stats['std'] == 0, 1, self.feature_stats['std'])
        
        # Apply normalization: (x - mean) / std
        X_normalized = (X - self.feature_stats['mean']) / self.feature_stats['std']
        return X_normalized
        
    def train(self, train_data, val_data):
        """
        Train the CNN model with proper preprocessing and handling of NaN values
        """
        # Convert Spark dataframes to numpy arrays
        X_train = np.array([row.features.toArray() if hasattr(row.features, 'toArray') 
                           else np.array(row.features) 
                           for row in train_data.select("features").collect()])
                           
        y_train = np.array([float(row.label) for row in train_data.select("label").collect()])
        
        X_val = np.array([row.features.toArray() if hasattr(row.features, 'toArray') 
                         else np.array(row.features) 
                         for row in val_data.select("features").collect()])
                         
        y_val = np.array([float(row.label) for row in val_data.select("label").collect()])
        
        # Handle NaN and infinite values
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply feature normalization
        X_train = self.normalize_features(X_train)
        X_val = self.normalize_features(X_val)
        
        # Log data shape and check for NaNs after preprocessing
        print(f"Shape of X_train: {X_train.shape}")
        print(f"Shape of y_train: {y_train.shape}")
        print(f"Any NaNs in X_train after preprocessing: {np.isnan(X_train).any()}")
        print(f"Any NaNs in y_train: {np.isnan(y_train).any()}")
        print(f"Range of X_train values: [{np.min(X_train)}, {np.max(X_train)}]")
        print(f"Range of y_train values: [{np.min(y_train)}, {np.max(y_train)}]")
        print(f"Class distribution in training set: {np.bincount(y_train.astype(int))}")
        
        # Reshape for CNN (adding channel dimension)
        input_dim = X_train.shape[1]
        X_train = X_train.reshape(X_train.shape[0], input_dim, 1)
        X_val = X_val.reshape(X_val.shape[0], input_dim, 1)
        
        # Build the model
        self.model = self.build_model(input_dim)
        
        # Set up callbacks for early stopping and learning rate reduction
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'best_cnn_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=0.0005,
                verbose=1
            )
        ]
        
        # Train the model with a smaller batch size
        history = self.model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=64,  # Smaller batch size for better gradient estimates
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, test_data):
        """
        Evaluate the CNN model
        """
        if self.model is None:
            print("Model has not been trained yet.")
            return 0
            
        # Convert Spark dataframe to numpy arrays
        X_test = np.array([row.features.toArray() if hasattr(row.features, 'toArray') 
                          else np.array(row.features) 
                          for row in test_data.select("features").collect()])
                          
        y_test = np.array([float(row.label) for row in test_data.select("label").collect()])
        
        # Handle NaN and infinite values
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply the same normalization as done during training
        X_test = self.normalize_features(X_test)
        
        # Reshape for CNN (adding channel dimension)
        input_dim = X_test.shape[1]
        X_test = X_test.reshape(X_test.shape[0], input_dim, 1)
        
        # Evaluate the model
        metrics = self.model.evaluate(X_test, y_test, verbose=1)
        print(f"Test loss: {metrics[0]}, Test accuracy: {metrics[1]}, Test AUC: {metrics[2]}")
        
        # Generate predictions for more detailed metrics
        y_pred = self.model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Calculate confusion matrix components
        tp = np.sum((y_test == 1) & (y_pred_binary == 1))
        fp = np.sum((y_test == 0) & (y_pred_binary == 1))
        tn = np.sum((y_test == 0) & (y_pred_binary == 0))
        fn = np.sum((y_test == 1) & (y_pred_binary == 0))
        
        # Calculate additional metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        return metrics[1]  # Return accuracy
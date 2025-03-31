from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


class CNNModel:
    def __init__(self, input_dim=None, num_classes=None, epochs=50, batch_size=64):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def build_model(self, input_dim, num_classes):
        # Tạo mô hình CNN
        model = Sequential([
            Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(input_dim, 1)),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=128, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def preprocess_for_keras(self, spark_df):
        # Chuyển đổi từ Spark DataFrame sang numpy arrays cho Keras
        pandas_df = spark_df.select("features", "label").toPandas()

        # Trích xuất đặc trưng và nhãn
        features = np.array([np.array(x.toArray()) for x in pandas_df["features"]])
        labels = pandas_df["label"].values

        # Điều chỉnh kích thước đặc trưng cho mô hình CNN
        features = features.reshape((features.shape[0], features.shape[1], 1))

        return features, labels

    def train(self, train_data, val_data=None):
        # Tiền xử lý dữ liệu
        X_train, y_train = self.preprocess_for_keras(train_data)

        if val_data is not None:
            X_val, y_val = self.preprocess_for_keras(val_data)
            validation_data = (X_val, y_val)
        else:
            validation_data = None

        # Lấy số chiều đầu vào và số lớp (nếu chưa được đặt)
        if self.input_dim is None:
            self.input_dim = X_train.shape[1]

        if self.num_classes is None:
            self.num_classes = len(np.unique(y_train))

        # Xây dựng mô hình
        self.model = self.build_model(self.input_dim, self.num_classes)

        # Định nghĩa early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        # Huấn luyện mô hình
        history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            callbacks=[early_stopping],
            verbose=1
        )

        return history.history

    def evaluate(self, test_data):
        # Tiền xử lý dữ liệu kiểm tra
        X_test, y_test = self.preprocess_for_keras(test_data)

        # Đánh giá mô hình
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)

        return accuracy

    def predict(self, data):
        # Tiền xử lý dữ liệu
        X, _ = self.preprocess_for_keras(data)

        # Dự đoán
        predictions = self.model.predict(X)

        return predictions
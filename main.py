from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# Import các mô hình
from models.cnn_model import CNNModel
from models.rnn_model import RNNModel
from models.lstm_model import LSTMModel
from models.mlp_model import MLPModel
from utils.data_preprocessing import preprocess_data
from utils.visualization import plot_accuracy_loss, compare_models_accuracy


def create_spark_session():
    # Khởi tạo Spark session
    return SparkSession.builder \
        .appName("WebAttackDetection") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.0.1") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()


def load_data(spark):
    # Đường dẫn tới dữ liệu UNSW-NB15
    # Giả sử dữ liệu đã được tải về và lưu trữ trong thư mục data
    data_path = "data/UNSW-NB15/"

    # Đọc các file CSV từ dataset
    train_df = spark.read.csv(os.path.join(data_path, "UNSW-NB15_1.csv"), header=True, inferSchema=True)
    test_df = spark.read.csv(os.path.join(data_path, "UNSW-NB15_2.csv"), header=True, inferSchema=True)

    # Có thể đọc thêm các file khác trong dataset và kết hợp lại
    additional_data = spark.read.csv(os.path.join(data_path, "UNSW-NB15_3.csv"), header=True, inferSchema=True)
    train_df = train_df.union(additional_data)

    additional_data = spark.read.csv(os.path.join(data_path, "UNSW-NB15_4.csv"), header=True, inferSchema=True)
    train_df = train_df.union(additional_data)

    return train_df, test_df


def prepare_raw_data(df):
    # Chuẩn bị dữ liệu thô (chỉ chuyển đổi cột nhãn và tạo vector đặc trưng)
    # Giả sử cột 'attack_cat' là nhãn
    label_indexer = StringIndexer(inputCol="attack_cat", outputCol="label")

    # Lựa chọn các cột số làm đặc trưng
    numeric_cols = [col for col in df.columns if df.schema[col].dataType.typeName() in ["integer", "double"]]
    numeric_cols = [col for col in numeric_cols if col != "label"]

    # Tạo vector đặc trưng
    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")

    # Tạo pipeline
    pipeline = Pipeline(stages=[label_indexer, assembler])

    # Fit và transform dữ liệu
    model = pipeline.fit(df)
    data = model.transform(df)

    return data


def train_and_evaluate_models(raw_train_data, raw_test_data, processed_train_data, processed_test_data):
    models = {
        "CNN": CNNModel(),
        "RNN": RNNModel(),
        "LSTM": LSTMModel(),
        "MLP": MLPModel()
    }

    # Kết quả huấn luyện trên dữ liệu thô
    raw_results = {}
    print("Training on raw data...")
    for name, model in models.items():
        print(f"Training {name} model...")
        start_time = time.time()
        history = model.train(raw_train_data, raw_test_data)
        end_time = time.time()

        accuracy = model.evaluate(raw_test_data)
        raw_results[name] = {
            "history": history,
            "accuracy": accuracy,
            "time": end_time - start_time
        }
        print(f"{name} training completed. Accuracy: {accuracy:.4f}, Time: {end_time - start_time:.2f}s")

    # Kết quả huấn luyện trên dữ liệu đã xử lý
    processed_results = {}
    print("\nTraining on processed data...")
    for name, model in models.items():
        print(f"Training {name} model...")
        start_time = time.time()
        history = model.train(processed_train_data, processed_test_data)
        end_time = time.time()

        accuracy = model.evaluate(processed_test_data)
        processed_results[name] = {
            "history": history,
            "accuracy": accuracy,
            "time": end_time - start_time
        }
        print(f"{name} training completed. Accuracy: {accuracy:.4f}, Time: {end_time - start_time:.2f}s")

    return raw_results, processed_results


def visualize_results(raw_results, processed_results):
    # Vẽ đồ thị accuracy và loss cho từng mô hình trên dữ liệu thô
    print("Generating visualizations for raw data results...")
    for name, result in raw_results.items():
        plot_accuracy_loss(result["history"], name, data_type="raw")

    # Vẽ đồ thị so sánh accuracy giữa các mô hình trên dữ liệu thô
    compare_models_accuracy(raw_results, data_type="raw")

    # Vẽ đồ thị accuracy và loss cho từng mô hình trên dữ liệu đã xử lý
    print("Generating visualizations for processed data results...")
    for name, result in processed_results.items():
        plot_accuracy_loss(result["history"], name, data_type="processed")

    # Vẽ đồ thị so sánh accuracy giữa các mô hình trên dữ liệu đã xử lý
    compare_models_accuracy(processed_results, data_type="processed")


def main():
    # Tạo Spark session
    spark = create_spark_session()

    # Load dữ liệu
    print("Loading data...")
    train_df, test_df = load_data(spark)

    # Chuẩn bị dữ liệu thô
    print("Preparing raw data...")
    raw_train_data = prepare_raw_data(train_df)
    raw_test_data = prepare_raw_data(test_df)

    # Tiền xử lý dữ liệu
    print("Preprocessing data...")
    processed_train_data, processed_test_data = preprocess_data(train_df, test_df)

    # Huấn luyện và đánh giá các mô hình
    raw_results, processed_results = train_and_evaluate_models(
        raw_train_data, raw_test_data, processed_train_data, processed_test_data
    )

    # Vẽ đồ thị kết quả
    visualize_results(raw_results, processed_results)

    # Đóng Spark session
    spark.stop()


if __name__ == "__main__":
    main()
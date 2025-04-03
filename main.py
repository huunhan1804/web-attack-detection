from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import pandas as pd
# Import các mô hình
from models.cnn_model import CNNModel
from models.rnn_model import RNNModel
from models.lstm_model import LSTMModel
from models.mlp_model import MLPModel
from utils.data_preprocessing import preprocess_data
from utils.visualization import plot_accuracy_loss, compare_models_accuracy
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

def create_spark_session():
    # Khởi tạo Spark session
    return SparkSession.builder \
        .appName("WebAttackDetection") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.0.1") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()


def load_data(spark):
    try:
        csv_files = [
            'E:\\Data\\Web_attack\\OneDrive_2_3-26-2025\\UNSW-NB15_1.csv',
            'E:\\Data\\Web_attack\\OneDrive_2_3-26-2025\\UNSW-NB15_2.csv',
            'E:\\Data\\Web_attack\\OneDrive_2_3-26-2025\\UNSW-NB15_3.csv',
            'E:\\Data\\Web_attack\\OneDrive_2_3-26-2025\\UNSW-NB15_4.csv'
        ]
        
        column_names = [
            'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl',
            'dttl', 'sloss', 'dloss', 'service', 'sload', 'dload', 'spkts', 'dpkts', 'swin', 'dwin',
            'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'sjit', 'djit',
            'stime', 'ltime', 'sintpkt', 'dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports',
            'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst',
            'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
            'attack_cat', 'precision'
        ]
        
        dfs = []
        for file in csv_files:
            if os.path.exists(file):
                print(f"Loading file: {file}")
                df = spark.read.csv(file, header=False, inferSchema=True)
                if df.count() > 0:
                    # Only rename columns if we have data
                    df = df.toDF(*column_names)
                    dfs.append(df)
                else:
                    print(f"Warning: File {file} is empty")
        
        if dfs:
            full_dataset = dfs[0]
            for df in dfs[1:]:
                full_dataset = full_dataset.unionByName(df)
                
            # Handle missing values and data types
            # Convert all numeric columns to double to avoid type issues
            numeric_cols = [col for col in full_dataset.columns 
                        if full_dataset.schema[col].dataType.typeName() in ["integer", "double", "float"]]
            
            for col in numeric_cols:
                full_dataset = full_dataset.withColumn(col, F.col(col).cast(DoubleType()))
                # Replace nulls with 0
                full_dataset = full_dataset.withColumn(col, 
                                                    F.when(F.col(col).isNull(), 0.0)
                                                    .otherwise(F.col(col)))
            
            # Handle 'attack_cat' column - fix the "bengin" typo to "benign"
            full_dataset = full_dataset.withColumn("attack_cat", 
                                                F.when(F.col("attack_cat").isNull(), "benign")
                                                .otherwise(F.col("attack_cat")))
            
            print(f"Total dataset size: {full_dataset.count()} records")
            # Split data
            train_data, test_data = full_dataset.randomSplit([0.8, 0.2], seed=42)
            
            return train_data, test_data            
        else:
            raise FileNotFoundError("Không tìm thấy file dữ liệu")
    
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu: {e}")
        import traceback
        traceback.print_exc()
    
    return None, None
    
def prepare_raw_data(df):
    try:
        # Chuẩn bị dữ liệu thô (chỉ chuyển đổi cột nhãn và tạo vector đặc trưng)
        # Fix the "bengin" typo to "benign"
        df = df.withColumn("attack_cat", F.when(F.col("attack_cat").isNull(), "benign")
                                    .otherwise(F.col("attack_cat")))
    
        label_indexer = StringIndexer(inputCol="attack_cat", outputCol="label", handleInvalid="keep")
        
        # Lựa chọn các cột số làm đặc trưng
        numeric_cols = [col for col in df.columns if df.schema[col].dataType.typeName() in ["integer", "double", "float"]]
        numeric_cols = [col for col in numeric_cols if col not in ["label", "precision"]]
        
        # Handle missing values
        for col in numeric_cols:
            df = df.withColumn(col, F.col(col).cast(DoubleType()))
            df = df.withColumn(col, F.when(F.col(col).isNull(), 0.0).otherwise(F.col(col)))
        
        # Tạo vector đặc trưng
        assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features", handleInvalid="keep")
    
        # Tạo pipeline
        pipeline = Pipeline(stages=[label_indexer, assembler])
    
        # Fit và transform dữ liệu
        model = pipeline.fit(df)
        data = model.transform(df)
        
        # Select only necessary columns
        data = data.select("features", "label")
    
        return data
    except Exception as e:
        print(f"Error in prepare_raw_data: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to a simpler approach
        print("Using simple fallback for raw data preparation...")
        
        # Basic label transformation
        label_indexer = StringIndexer(inputCol="attack_cat", outputCol="label", handleInvalid="keep")
        
        # Select a subset of reliable numeric columns
        safe_cols = ["dur", "sbytes", "dbytes", "sttl", "dttl", "sload", "dload", "spkts", "dpkts"]
        safe_cols = [col for col in safe_cols if col in df.columns]
        
        # Handle missing values
        for col in safe_cols:
            df = df.withColumn(col, F.col(col).cast(DoubleType()))
            df = df.withColumn(col, F.when(F.col(col).isNull(), 0.0).otherwise(F.col(col)))
        
        # Simple assembler
        assembler = VectorAssembler(inputCols=safe_cols, outputCol="features", handleInvalid="keep")
        
        # Simple pipeline
        pipeline = Pipeline(stages=[label_indexer, assembler])
        
        # Fit and transform
        model = pipeline.fit(df)
        data = model.transform(df).select("features", "label")
        
        return data


def train_and_evaluate_models(raw_train_data, raw_test_data, processed_train_data, processed_test_data):
    models = {
        "CNN": CNNModel(),
    #    "RNN": RNNModel(),
    #    "LSTM": LSTMModel(),
    #    "MLP": MLPModel()
    }

    # Kết quả huấn luyện trên dữ liệu thô
    raw_results = {}
    print("Training on raw data...")
    for name, model in models.items():
        try:
            print(f"Training {name} model on raw data...")
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
        except Exception as e:
            print(f"Error training {name} model on raw data: {e}")
            raw_results[name] = {
                "history": None,
                "accuracy": 0,
                "time": 0
            }

    # Kết quả huấn luyện trên dữ liệu đã xử lý
    processed_results = {}
    print("\nTraining on processed data...")
    for name, model in models.items():
        try:
            print(f"Training {name} model on processed data...")
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
        except Exception as e:
            print(f"Error training {name} model on processed data: {e}")
            processed_results[name] = {
                "history": None,
                "accuracy": 0,
                "time": 0
            }

    return raw_results, processed_results


def visualize_results(raw_results, processed_results):
    try:
        # Vẽ đồ thị accuracy và loss cho từng mô hình trên dữ liệu thô
        print("Generating visualizations for raw data results...")
        for name, result in raw_results.items():
            if result["history"] is not None:
                plot_accuracy_loss(result["history"], name, data_type="raw")

        # Vẽ đồ thị so sánh accuracy giữa các mô hình trên dữ liệu thô
        compare_models_accuracy(raw_results, data_type="raw")

        # Vẽ đồ thị accuracy và loss cho từng mô hình trên dữ liệu đã xử lý
        print("Generating visualizations for processed data results...")
        for name, result in processed_results.items():
            if result["history"] is not None:
                plot_accuracy_loss(result["history"], name, data_type="processed")

        # Vẽ đồ thị so sánh accuracy giữa các mô hình trên dữ liệu đã xử lý
        compare_models_accuracy(processed_results, data_type="processed")
    except Exception as e:
        print(f"Error in visualization: {e}")
        import traceback
        traceback.print_exc()
        

def check_nan(data):
        X = np.array([np.array(x.toArray()) for x in data.select("features").collect()])
        y = np.array(data.select("label").collect()).flatten()
        if np.isnan(X).any() or np.isinf(X).any():
            print("Warning: Dữ liệu X có NaN hoặc Inf")
        if np.isnan(y).any() or np.isinf(y).any():
            print("Warning: Dữ liệu y có NaN hoặc Inf")

def main():
    try:
        # Tạo Spark session
        spark = create_spark_session()
        # Load dữ liệu
        print("Loading data...")
        train_df, test_df = load_data(spark)
        if train_df is None or test_df is None:
            print("Không thể tải dữ liệu. Kết thúc chương trình.")
            return

        # Show data summary
        print("Training data count:", train_df.count())
        print("Test data count:", test_df.count())
        
        # Chuẩn bị dữ liệu thô
        print("Preparing raw data...")
        raw_train_data = prepare_raw_data(train_df)
        raw_test_data = prepare_raw_data(test_df)
        
        check_nan(raw_train_data)
        check_nan(raw_test_data)
        
        # Tiền xử lý dữ liệu
        print("Preprocessing data...")
        processed_train_data, processed_test_data = preprocess_data(train_df, test_df)
        # Huấn luyện và đánh giá các mô hình
        raw_results, processed_results = train_and_evaluate_models(
            raw_train_data, raw_test_data, processed_train_data, processed_test_data
        )
        # Vẽ đồ thị kết quả
        visualize_results(raw_results, processed_results)
    except Exception as e:
        print(f"Lỗi trong hàm main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Chương trình đã hoàn thành.")
        # Đóng Spark session
        if 'spark' in locals():
            spark.stop()


if __name__ == "__main__":
    main()
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder, PCA
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from pyspark.ml.feature import Imputer


def preprocess_data(train_df, test_df):
    """
    Tiền xử lý dữ liệu bao gồm:
    - Xử lý giá trị thiếu
    - Mã hóa biến phân loại
    - Chuẩn hóa đặc trưng số
    - Giảm chiều dữ liệu
    - Chuẩn bị dữ liệu cho các mô hình deep learning
    """
    # Xác định các biến phân loại và số
    categorical_cols = [col for col in train_df.columns if train_df.schema[col].dataType.typeName() == "string"]
    categorical_cols = [col for col in categorical_cols if col != "attack_cat"]  # Loại bỏ cột nhãn

    numeric_cols = [col for col in train_df.columns if
                    train_df.schema[col].dataType.typeName() in ["integer", "double"]]

    # Xử lý giá trị thiếu
    imputer = Imputer(inputCols=numeric_cols, outputCols=numeric_cols)

    # Mã hóa biến phân loại
    indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_idx").fit(train_df) for col in categorical_cols]
    encoders = [OneHotEncoder(inputCol=f"{col}_idx", outputCol=f"{col}_vec") for col in categorical_cols]

    # Tạo các cột đã mã hóa
    encoded_cols = [f"{col}_vec" for col in categorical_cols]

    # Chuẩn hóa đặc trưng số
    assembler_numeric = VectorAssembler(inputCols=numeric_cols, outputCol="numeric_features")
    scaler = StandardScaler(inputCol="numeric_features", outputCol="scaled_numeric_features")

    # Mã hóa cột nhãn
    label_indexer = StringIndexer(inputCol="attack_cat", outputCol="label")

    # Tổng hợp tất cả các đặc trưng
    assembler_final = VectorAssembler(
        inputCols=["scaled_numeric_features"] + encoded_cols,
        outputCol="features_raw"
    )

    # Giảm chiều dữ liệu bằng PCA
    pca = PCA(k=20, inputCol="features_raw", outputCol="features")

    # Tạo pipeline
    stages = [imputer] + indexers + encoders + [assembler_numeric, scaler, label_indexer, assembler_final, pca]
    pipeline = Pipeline(stages=stages)

    # Fit pipeline trên tập huấn luyện
    model = pipeline.fit(train_df)

    # Transform dữ liệu
    train_data = model.transform(train_df)
    test_data = model.transform(test_df)

    # Chọn các cột cần thiết
    final_cols = ["features", "label"]
    train_data = train_data.select(final_cols)
    test_data = test_data.select(final_cols)

    return train_data, test_data
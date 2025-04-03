from pyspark.ml.feature import VectorAssembler, StandardScaler, OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

def preprocess_data(train_df, test_df):
    """
    Preprocess the data for model training.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        
    Returns:
        tuple: (processed_train_data, processed_test_data)
    """
    # Handle missing values in the attack_cat column
    train_df = train_df.withColumn("attack_cat", 
                                F.when(F.col("attack_cat").isNull(), "benign")
                                .otherwise(F.col("attack_cat")))
    
    test_df = test_df.withColumn("attack_cat", 
                                F.when(F.col("attack_cat").isNull(), "benign")
                                .otherwise(F.col("attack_cat")))

    # Convert string columns to numeric for both datasets
    string_cols = [field.name for field in train_df.schema.fields 
                if field.dataType.simpleString() == "string" and field.name != "attack_cat"]
    
    # Convert categorical features to numeric using StringIndexer and OneHotEncoder
    indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep") 
            for col in string_cols]
    
    encoders = [OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_vec") 
            for col in string_cols]
    
    # Index the label column
    label_indexer = StringIndexer(inputCol="attack_cat", outputCol="label", handleInvalid="keep")
    
    # Convert numeric columns and ensure they're of the right type
    numeric_cols = [field.name for field in train_df.schema.fields 
                if field.dataType.simpleString() in ("int", "double", "float") 
                and field.name != "label"]
    
    # Cast all numeric columns to double to avoid type issues
    for col in numeric_cols:
        train_df = train_df.withColumn(col, F.col(col).cast(DoubleType()))
        test_df = test_df.withColumn(col, F.col(col).cast(DoubleType()))
    
    # Handle null values in numeric columns
    for col in numeric_cols:
        train_df = train_df.withColumn(col, F.when(F.col(col).isNull(), 0.0).otherwise(F.col(col)))
        test_df = test_df.withColumn(col, F.when(F.col(col).isNull(), 0.0).otherwise(F.col(col)))
    
    # Create feature vector from numeric columns
    numeric_assembler = VectorAssembler(inputCols=numeric_cols, outputCol="numeric_features", handleInvalid="keep")
    
    # Create combined feature vector
    if string_cols:
        categorical_cols = [f"{col}_vec" for col in string_cols]
        combined_assembler = VectorAssembler(
            inputCols=["numeric_features"] + categorical_cols, 
            outputCol="features_unscaled",
            handleInvalid="keep"
        )
    else:
        combined_assembler = VectorAssembler(
            inputCols=["numeric_features"], 
            outputCol="features_unscaled",
            handleInvalid="keep"
        )
    
    # Scale features
    scaler = StandardScaler(inputCol="features_unscaled", outputCol="features")
    
    # Create pipeline
    stages = indexers + encoders + [label_indexer, numeric_assembler, combined_assembler, scaler]
    pipeline = Pipeline(stages=stages)
    
    # Try-except block to handle potential errors during pipeline fitting
    try:
        # Fit pipeline on training data
        print("Fitting pipeline on training data...")
        model = pipeline.fit(train_df)
        
        # Transform data
        train_data = model.transform(train_df)
        test_data = model.transform(test_df)
        
        # Select only necessary columns for modeling
        cols_to_select = ["features", "label"]
        train_data = train_data.select(cols_to_select)
        test_data = test_data.select(cols_to_select)
        
        return train_data, test_data
    
    except Exception as e:
        print(f"Error during data preprocessing: {str(e)}")
        # Provide a simpler fallback preprocessing if the full pipeline fails
        return fallback_preprocessing(train_df, test_df)

def fallback_preprocessing(train_df, test_df):
    """
    A simpler fallback preprocessing method if the main one fails.
    This only uses numeric features and basic scaling.
    """
    print("Using fallback preprocessing method...")
    
    # Handle missing values in the attack_cat column
    train_df = train_df.withColumn("attack_cat", 
                                F.when(F.col("attack_cat").isNull(), "benign")
                                .otherwise(F.col("attack_cat")))
    
    test_df = test_df.withColumn("attack_cat", 
                                F.when(F.col("attack_cat").isNull(), "benign")
                                .otherwise(F.col("attack_cat")))
    
    # Label indexer
    label_indexer = StringIndexer(inputCol="attack_cat", outputCol="label", handleInvalid="keep")
    
    # Get only numeric columns
    numeric_cols = [field.name for field in train_df.schema.fields 
                if field.dataType.simpleString() in ("int", "double", "float") 
                and field.name != "label"]
    
    # Handle missing and non-numeric values
    for col in numeric_cols:
        train_df = train_df.withColumn(col, F.col(col).cast(DoubleType()))
        test_df = test_df.withColumn(col, F.col(col).cast(DoubleType()))
        
        # Replace nulls with 0
        train_df = train_df.withColumn(col, F.when(F.col(col).isNull(), 0.0).otherwise(F.col(col)))
        test_df = test_df.withColumn(col, F.when(F.col(col).isNull(), 0.0).otherwise(F.col(col)))
    
    # Vector assembler
    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features_unscaled", handleInvalid="keep")
    
    # Scaler
    scaler = StandardScaler(inputCol="features_unscaled", outputCol="features")
    
    # Create pipeline
    pipeline = Pipeline(stages=[label_indexer, assembler, scaler])
    
    try:
        # Fit and transform
        model = pipeline.fit(train_df)
        train_data = model.transform(train_df).select("features", "label")
        test_data = model.transform(test_df).select("features", "label")
        
        return train_data, test_data
    except Exception as e:
        print(f"Fallback preprocessing also failed: {str(e)}")
        print("Using minimal preprocessing as last resort...")
        
        # As a last resort, use minimal preprocessing
        label_indexer = StringIndexer(inputCol="attack_cat", outputCol="label", handleInvalid="keep")
        
        # Get a subset of reliable numeric columns
        safe_cols = ["dur", "sbytes", "dbytes", "sttl", "dttl", "sload", "dload", "spkts", "dpkts"]
        safe_cols = [col for col in safe_cols if col in train_df.columns]
        
        # Handle missing values
        for col in safe_cols:
            train_df = train_df.withColumn(col, F.col(col).cast(DoubleType()))
            test_df = test_df.withColumn(col, F.col(col).cast(DoubleType()))
            
            train_df = train_df.withColumn(col, F.when(F.col(col).isNull(), 0.0).otherwise(F.col(col)))
            test_df = test_df.withColumn(col, F.when(F.col(col).isNull(), 0.0).otherwise(F.col(col)))
        
        # Simple pipeline
        assembler = VectorAssembler(inputCols=safe_cols, outputCol="features", handleInvalid="keep")
        pipeline = Pipeline(stages=[label_indexer, assembler])
        
        # Fit and transform
        model = pipeline.fit(train_df)
        train_data = model.transform(train_df).select("features", "label")
        test_data = model.transform(test_df).select("features", "label")
        
        return train_data, test_data
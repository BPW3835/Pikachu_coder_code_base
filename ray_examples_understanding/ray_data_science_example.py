"""
Example of using Ray with data science libraries for large-scale data processing.
This script demonstrates integration with pandas and scikit-learn.
"""

import ray
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time


# Initialize Ray
ray.init(ignore_reinit_error=True)


@ray.remote
def preprocess_partition(partition_data):
    """
    Preprocess a partition of data.
    
    Args:
        partition_data: A tuple containing (partition_df,) 
        
    Returns:
        Preprocessed DataFrame partition
    """
    partition_df = partition_data[0]  # Extract DataFrame from tuple
    
    # Example preprocessing steps
    # Fill missing values
    partition_df = partition_df.fillna(0)
    
    # Normalize numerical columns
    numerical_cols = partition_df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if col != 'target':  # Don't normalize target column
            mean = partition_df[col].mean()
            std = partition_df[col].std()
            if std > 0:
                partition_df[col] = (partition_df[col] - mean) / std
    
    return partition_df


@ray.remote
def train_model_on_partition(partition_data, model_params=None):
    """
    Train a model on a data partition.
    
    Args:
        partition_data: Tuple of (X, y) for training
        model_params: Parameters for the model
        
    Returns:
        Trained model
    """
    X, y = partition_data
    
    if model_params is None:
        model_params = {"n_estimators": 10, "random_state": 42}
    
    # Create and train model
    model = RandomForestClassifier(**model_params)
    model.fit(X, y)
    
    return model


@ray.remote
def evaluate_model(model, test_data):
    """
    Evaluate a model on test data.
    
    Args:
        model: Trained model
        test_data: Tuple of (X_test, y_test)
        
    Returns:
        Accuracy score
    """
    X_test, y_test = test_data
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy


def create_sample_dataset(n_rows=100000, n_features=20):
    """
    Create a sample dataset for demonstration.
    
    Args:
        n_rows: Number of rows
        n_features: Number of features
        
    Returns:
        Pandas DataFrame
    """
    print(f"Creating sample dataset with {n_rows:,} rows and {n_features} features")
    
    # Generate synthetic data
    X = np.random.randn(n_rows, n_features)
    # Create target variable with some relationship to features
    y = (X[:, 0] + X[:, 1] - X[:, 2] + np.random.randn(n_rows) * 0.1 > 0).astype(int)
    
    # Create DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Introduce some missing values randomly
    missing_indices = np.random.choice(df.index, size=int(0.01 * len(df)), replace=False)
    df.loc[missing_indices, 'feature_0'] = np.nan
    
    return df


def process_large_dataframe_parallel(df, n_partitions=10):
    """
    Process a large DataFrame in parallel using Ray.
    
    Args:
        df: Pandas DataFrame to process
        n_partitions: Number of partitions to split the data into
        
    Returns:
        Processed DataFrame
    """
    print(f"Processing DataFrame with shape {df.shape} using {n_partitions} partitions")
    
    # Split DataFrame into partitions
    partitions = np.array_split(df, n_partitions)
    print(f"Split data into {len(partitions)} partitions")
    
    # Submit preprocessing tasks (wrapped in tuples to avoid serialization issues)
    preprocessed_futures = [preprocess_partition.remote((partition,)) for partition in partitions]
    
    # Collect results
    preprocessed_partitions = ray.get(preprocessed_futures)
    
    # Concatenate back into a single DataFrame
    processed_df = pd.concat(preprocessed_partitions, ignore_index=True)
    
    return processed_df


def distributed_model_training(df, n_partitions=5):
    """
    Train models on partitions of data and evaluate them.
    
    Args:
        df: Processed DataFrame with features and target
        n_partitions: Number of partitions for training
        
    Returns:
        List of model accuracies
    """
    print(f"Distributed model training with {n_partitions} partitions")
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col != 'target']
    X = df[feature_cols]
    y = df['target']
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Split training data into partitions
    X_train_partitions = np.array_split(X_train, n_partitions)
    y_train_partitions = np.array_split(y_train, n_partitions)
    
    # Create training data tuples
    train_data_partitions = list(zip(X_train_partitions, y_train_partitions))
    
    # Train models on each partition
    model_futures = [
        train_model_on_partition.remote(train_data) 
        for train_data in train_data_partitions
    ]
    
    # Collect trained models
    models = ray.get(model_futures)
    
    # Evaluate each model on the test set
    test_data = (X_test, y_test)
    accuracy_futures = [
        evaluate_model.remote(model, test_data) 
        for model in models
    ]
    
    # Collect accuracy scores
    accuracies = ray.get(accuracy_futures)
    
    return accuracies


def main():
    """Main function demonstrating Ray with data science libraries."""
    print("=" * 70)
    print("Ray with Data Science Libraries Example")
    print("=" * 70)
    
    # Create sample dataset
    df = create_sample_dataset(n_rows=50000, n_features=15)
    print(f"Created dataset with shape {df.shape}")
    
    # Example 1: Parallel DataFrame processing
    print("\n1. Parallel DataFrame preprocessing:")
    start_time = time.time()
    processed_df = process_large_dataframe_parallel(df, n_partitions=8)
    preprocessing_time = time.time() - start_time
    print(f"Preprocessing took {preprocessing_time:.2f} seconds")
    print(f"Processed DataFrame shape: {processed_df.shape}")
    print("Missing values after preprocessing:")
    print(processed_df.isnull().sum().sum())
    
    # Example 2: Distributed model training
    print("\n2. Distributed model training and evaluation:")
    start_time = time.time()
    accuracies = distributed_model_training(processed_df, n_partitions=5)
    training_time = time.time() - start_time
    print(f"Training and evaluation took {training_time:.2f} seconds")
    print(f"Model accuracies: {[f'{acc:.4f}' for acc in accuracies]}")
    print(f"Average accuracy: {np.mean(accuracies):.4f}")
    print(f"Accuracy standard deviation: {np.std(accuracies):.4f}")
    
    # Shutdown Ray
    ray.shutdown()
    print("\n" + "=" * 70)
    print("Data science example completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
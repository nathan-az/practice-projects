import pyspark
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, VectorIndexer
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import when, struct, lit, coalesce, monotonically_increasing_id, row_number
from pyspark.sql.types import *
from pyspark.sql.window import Window
import numpy as np
import os
from typing import Dict, List, Tuple, Union, Text


def get_missing_counts(df: DataFrame) -> Dict[str, int]:
    missing = {col: df.where(df[col].isNull()).count() for col in df.columns}
    return missing

def print_missing(df: DataFrame) -> Dict[str, int]:
    missing = get_missing_counts(df)
    total_rows = df.count()
    print(f"Total samples: {total_rows}")
    for col, num_missing in missing.items():
        if num_missing > 0:
            print(f"Column: {col}, num missing: {num_missing} (% missing: {num_missing/total_rows*100:.1f}%)")
    return missing

def create_indexers(columns_for_encoding: DataFrame) -> List[StringIndexer]:
    indexers = [StringIndexer(inputCol=col, outputCol=col+"_index") for col in columns_for_encoding]
    return indexers

def indexers_to_encoder(columns_for_encoding: DataFrame, indexers: List[StringIndexer]) -> OneHotEncoder:
    inputs = [indexer.getOutputCol() for indexer in indexers]
    encoder = OneHotEncoder(inputCols=inputs, outputCols=[col + "_encoded" for col in columns_for_encoding])
    return encoder

def string_to_onehot_pipeline(columns_for_encoding: DataFrame) -> Pipeline:
    indexers = create_indexers(columns_for_encoding)
    encoder = indexers_to_encoder(columns_for_encoding, indexers)
    stages = indexers  + [encoder]
    pipeline = Pipeline(stages=stages)
    return pipeline

def string_to_onehot_transformer(df: DataFrame, columns_for_encoding: List[str]) -> PipelineModel:
    pipeline = string_to_onehot_pipeline(columns_for_encoding)
    transformer = pipeline.fit(df)
    return transformer

def create_feature_vectoriser(df: DataFrame, columns_to_vectorise: List[str], output_col: str) -> PipelineModel:
    vectoriser = VectorAssembler()\
            .setInputCols(columns_to_vectorise)\
            .setOutputCol(output_col)
    vectoriser_pipeline = Pipeline(stages=[vectoriser])
    transformer = vectoriser_pipeline.fit(df)
    return transformer
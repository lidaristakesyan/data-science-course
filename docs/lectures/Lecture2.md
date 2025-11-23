  

Data Science Tools and Environments

Advanced Course for Data Science Students

_What is Data and How to Use It Optimally in Analysis, ML/DL Pipelines_

**Duration: 7 Lectures**

Part I: Core Tools & Data Processing (3 Lectures) | Part II: Big Data Ecosystem (4 Lectures)

Course Overview

This advanced course assumes prior familiarity with Python programming and basic data science concepts. The curriculum bridges foundational knowledge with expert-level application, focusing on production-ready techniques, performance optimization, and industry best practices. Students gain hands-on experience with professional workflows that scale from exploratory analysis to enterprise-grade data pipelines.

Prerequisites

- Proficiency in Python programming (functions, classes, decorators, context managers)
- Basic understanding of NumPy arrays and Pandas DataFrames
- Familiarity with command-line interfaces and Git version control
- Foundational statistics knowledge (distributions, hypothesis testing, correlation)

PART I: Core Tools, Environments & Data Processing

Lecture 1: Professional Development Environments & Advanced Tool Configuration

Learning Objectives

- Design and implement reproducible virtual environments using venv, conda, and containerization
- Configure JupyterLab for production-grade data science workflows with custom kernels
- Implement environment management best practices for ML/DL pipeline reproducibility
- Master dependency management, version pinning, and conflict resolution strategies

1.1 Virtual Environment Architecture

Virtual environments provide isolated Python installations that prevent dependency conflicts between projects. Understanding the underlying architecture is essential for troubleshooting and optimization. A virtual environment creates a self-contained directory structure containing a Python interpreter symlink, a site-packages directory for installed libraries, and activation scripts that modify PATH and PYTHONPATH environment variables.

**The venv Module: Native Python Isolation**

Python's built-in venv module creates lightweight virtual environments. The key insight is that venv does not copy the Python interpreter; instead, it creates symlinks (on Unix) or copies (on Windows) to the system Python, significantly reducing disk usage while maintaining isolation.

# Create virtual environment with system site-packages access

python -m venv --system-site-packages ./venv_with_system

  

# Create completely isolated environment

python -m venv --clear ./venv_isolated

  

# Upgrade pip immediately after creation

source ./venv_isolated/bin/activate

python -m pip install --upgrade pip setuptools wheel

**Conda Environments: Beyond Python**

Conda environments differ fundamentally from venv in that they can manage non-Python dependencies (C libraries, compilers, CUDA toolkits). This is crucial for data science where packages like NumPy, SciPy, and TensorFlow depend on optimized BLAS/LAPACK implementations.

# Create environment with specific Python and dependencies

conda create -n ds_env python=3.11 numpy scipy pandas scikit-learn -c conda-forge

  

# Export environment for reproducibility

conda env export --from-history > environment.yml

  

# Create exact reproduction from lock file

conda-lock install -n ds_env_locked conda-lock.yml

1.2 Advanced JupyterLab Configuration

JupyterLab serves as the primary interactive development environment for data scientists. Advanced configuration transforms it from a notebook viewer into a comprehensive IDE with debugging, profiling, and collaboration capabilities.

**Custom Kernel Management**

# Register virtual environment as Jupyter kernel

python -m ipykernel install --user --name=ds_project --display-name="DS Project (Python 3.11)"

  

# List available kernels

jupyter kernelspec list

  

# Remove obsolete kernel

jupyter kernelspec uninstall ds_old_project

1.3 Data Types: Memory Representation and Optimization

Understanding how Python and NumPy represent data in memory is fundamental to writing efficient data science code. Memory layout directly impacts cache utilization, vectorization possibilities, and overall computational performance.

**NumPy dtype System**

import numpy as np

  

# Examine dtype properties

dt = np.dtype('float32')

print(f"Itemsize: {dt.itemsize} bytes")

print(f"Byte order: {dt.byteorder}")

  

# Memory-efficient integer selection

data = np.array([1, 2, 3, 100, 200], dtype=np.uint8)  # 1 byte per element

data_large = np.array([1, 2, 3, 100, 200], dtype=np.int64)  # 8 bytes

  

print(f"uint8 memory: {data.nbytes} bytes")

print(f"int64 memory: {data_large.nbytes} bytes")

**Pandas Memory Optimization**

import pandas as pd

  

def optimize_dataframe(df):

    """Reduce DataFrame memory footprint."""

    for col in df.columns:

        col_type = df[col].dtype

        if col_type == 'object':

            num_unique = df[col].nunique()

            if num_unique / len(df) < 0.5:  # Cardinality < 50%

                df[col] = df[col].astype('category')

        elif col_type == 'float64':

            df[col] = pd.to_numeric(df[col], downcast='float')

        elif col_type == 'int64':

            df[col] = pd.to_numeric(df[col], downcast='integer')

    return df

Recommended Resources for Lecture 1

|   |   |   |
|---|---|---|
|**Resource**|**Type**|**Coverage**|
|Python Data Science Handbook (VanderPlas)|Book, Ch. 1-2|IPython, Jupyter, NumPy fundamentals|
|Effective Python, 3rd Ed. (Slatkin)|Book, Items 73-77|Virtual environments, dependencies|
|conda.io Documentation|Official Docs|Environment management, conda-forge|
|JupyterLab Documentation|Official Docs|Extension system, configuration|
|NumPy User Guide: Data Types|Official Docs|dtype system, structured arrays|

Lecture 2: Advanced NumPy, Pandas, and SciPy for High-Performance Computing

Learning Objectives

- Master NumPy broadcasting, advanced indexing, and memory-efficient operations
- Implement high-performance Pandas operations using vectorization and method chaining
- Apply SciPy's sparse matrices and optimization routines for scientific computing
- Profile and optimize code bottlenecks in data processing pipelines

2.1 NumPy: Beyond the Basics

**Broadcasting: The Key to Vectorized Operations**

Broadcasting is NumPy's mechanism for performing operations on arrays of different shapes. Understanding broadcasting rules eliminates the need for explicit loops and enables highly optimized SIMD operations. The rules are: (1) dimensions are compared from right to left, (2) dimensions are compatible if equal or one is 1, (3) missing dimensions are treated as 1.

import numpy as np

  

# Broadcasting example: normalize features

data = np.random.randn(1000, 50)  # 1000 samples, 50 features

mean = data.mean(axis=0)  # Shape: (50,)

std = data.std(axis=0)    # Shape: (50,)

  

# Broadcasting (1000, 50) - (50,) -> (1000, 50)

normalized = (data - mean) / std

  

# Outer product via broadcasting

a = np.arange(5)[:, np.newaxis]  # Shape: (5, 1)

b = np.arange(3)                 # Shape: (3,)

outer = a * b                    # Shape: (5, 3)

**Advanced Indexing Patterns**

# Boolean indexing for conditional selection

data = np.random.randn(1000, 10)

mask = (data[:, 0] > 0) & (data[:, 1] < 0)

filtered = data[mask]  # Returns copy

  

# Fancy indexing for reordering

indices = np.argsort(data[:, 0])[::-1]  # Descending order

sorted_data = data[indices]  # Returns copy

  

# np.where for conditional element selection

result = np.where(data > 0, data, 0)  # ReLU operation

2.2 Pandas: Production-Grade Data Manipulation

**Method Chaining for Readable Pipelines**

import pandas as pd

import numpy as np

  

def calculate_metrics(df):

    return df.assign(

        total=lambda x: x['quantity'] * x['price'],

        log_total=lambda x: np.log1p(x['total'])

    )

  

result = (

    pd.read_csv('sales.csv')

    .query('date >= "2024-01-01"')

    .pipe(calculate_metrics)

    .groupby('category', as_index=False)

    .agg({'total': ['sum', 'mean'], 'quantity': 'count'})

    .sort_values('sum', ascending=False)

)

**GroupBy: Split-Apply-Combine Pattern**

# transform: broadcast aggregation back to original index

df['pct_of_category'] = (

    df.groupby('category')['sales']

    .transform(lambda x: x / x.sum() * 100)

)

  

# Multiple aggregations with named columns

agg_result = df.groupby('category').agg(

    total_sales=('sales', 'sum'),

    avg_sales=('sales', 'mean'),

    num_products=('product_id', 'nunique')

)

2.3 SciPy: Scientific Computing Toolkit

**Sparse Matrices for Memory Efficiency**

from scipy import sparse

import numpy as np

  

# Construct sparse matrix incrementally

row = np.array([0, 0, 1, 2, 2])

col = np.array([0, 2, 1, 0, 2])

data = np.array([1, 2, 3, 4, 5])

coo = sparse.coo_matrix((data, (row, col)), shape=(3, 3))

  

# Convert to CSR for efficient arithmetic

csr = coo.tocsr()

print(f"Sparse memory: {csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes} bytes")

Recommended Resources for Lecture 2

|   |   |   |
|---|---|---|
|**Resource**|**Type**|**Coverage**|
|Python Data Science Handbook (VanderPlas)|Book, Ch. 2-3|NumPy, Pandas comprehensive|
|NumPy Documentation: Broadcasting|Official Guide|Broadcasting rules, examples|
|Pandas User Guide: GroupBy|Official Docs|Split-apply-combine patterns|
|"High Performance Python" (Gorelick)|Book, Ch. 4-6|NumPy internals, profiling|
|Wes McKinney's Python for Data Analysis|Book, 3rd Ed.|Pandas creator's authoritative guide|

Lecture 3: Data Processing, Visualization, and Feature Engineering

Learning Objectives

- Implement robust data import/export pipelines supporting multiple formats
- Apply advanced missing data imputation techniques appropriate to data characteristics
- Design feature engineering pipelines with proper train/test separation
- Create publication-quality visualizations using Matplotlib and Seaborn

3.1 Multi-Format Data Import/Export

**Efficient CSV Processing**

import pandas as pd

  

# Memory-efficient CSV reading

chunk_iter = pd.read_csv(

    'large_file.csv',

    chunksize=100_000,

    dtype={'id': 'int32', 'category': 'category', 'value': 'float32'},

    usecols=['id', 'category', 'value', 'date'],

    parse_dates=['date'],

    na_values=['', 'NA', 'NULL', '-999']

)

  

# Process chunks and aggregate

results = []

for chunk in chunk_iter:

    processed = chunk.groupby('category')['value'].sum()

    results.append(processed)

final = pd.concat(results).groupby(level=0).sum()

**Parquet: The Columnar Format for Analytics**

import pandas as pd

  

# Write with compression and partitioning

df.to_parquet(

    'data/',

    engine='pyarrow',

    compression='snappy',

    partition_cols=['year', 'month'],

    index=False

)

  

# Read with column selection and filtering

df = pd.read_parquet(

    'data/',

    columns=['id', 'value', 'category'],

    filters=[('year', '>=', 2023), ('category', 'in', ['A', 'B'])]

)

3.2 Missing Data: Theory and Practice

**Advanced Imputation Strategies**

from sklearn.impute import KNNImputer, IterativeImputer

from sklearn.ensemble import RandomForestRegressor

  

# KNN Imputation - preserves multivariate relationships

knn_imputer = KNNImputer(n_neighbors=5, weights='distance')

df_imputed = pd.DataFrame(

    knn_imputer.fit_transform(df),

    columns=df.columns,

    index=df.index

)

  

# Iterative imputation with Random Forest

iter_imputer = IterativeImputer(

    estimator=RandomForestRegressor(n_estimators=100, random_state=42),

    max_iter=10,

    random_state=42

)

3.3 Data Transformation Pipeline

**Scaling and Normalization**

from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer

from sklearn.compose import ColumnTransformer

  

# Column-specific transformations

preprocessor = ColumnTransformer([

    ('num_standard', StandardScaler(), ['age', 'income']),

    ('num_robust', RobustScaler(), ['transaction_amount']),

    ('num_power', PowerTransformer(method='yeo-johnson'), ['skewed_feature']),

], remainder='passthrough')

  

# CRITICAL: fit on training data only, transform both

X_train_scaled = preprocessor.fit_transform(X_train)

X_test_scaled = preprocessor.transform(X_test)  # No fit!

3.4 Outlier Detection and Handling

from sklearn.ensemble import IsolationForest

from sklearn.neighbors import LocalOutlierFactor

  

# Isolation Forest - efficient for high dimensions

iso_forest = IsolationForest(

    contamination=0.05,

    random_state=42,

    n_jobs=-1

)

outlier_labels = iso_forest.fit_predict(X)  # -1 = outlier

  

# Local Outlier Factor - density-based

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)

lof_labels = lof.fit_predict(X)

3.5 Visualization for Analysis

**Seaborn for Statistical Visualization**

import seaborn as sns

import matplotlib.pyplot as plt

  

sns.set_theme(style='whitegrid', palette='deep', font_scale=1.1)

  

# Heatmap for correlation matrix

fig, ax = plt.subplots(figsize=(10, 8))

corr = df.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(

    corr, mask=mask, annot=True, fmt='.2f',

    cmap='RdBu_r', center=0, square=True, ax=ax

)

3.6 Feature Engineering

from sklearn.feature_selection import SelectKBest, mutual_info_classif, SelectFromModel

from sklearn.ensemble import RandomForestClassifier

  

# Mutual information (captures non-linear relationships)

mi_selector = SelectKBest(score_func=mutual_info_classif, k=20)

X_mi = mi_selector.fit_transform(X, y)

  

# Tree-based importance with threshold

rf = RandomForestClassifier(n_estimators=100, random_state=42)

model_selector = SelectFromModel(rf, threshold='median')

X_model = model_selector.fit_transform(X, y)

Recommended Resources for Lecture 3

|   |   |   |
|---|---|---|
|**Resource**|**Type**|**Coverage**|
|"Feature Engineering for ML" (Zheng)|Book|Comprehensive feature engineering|
|Pandas I/O Documentation|Official Docs|All file format support|
|Scikit-learn User Guide: Preprocessing|Official Docs|Transformers, pipelines|
|"Fundamentals of Data Visualization" (Wilke)|Book|Principles of visualization|
|Seaborn Tutorial Gallery|Official Docs|Statistical visualization patterns|

PART II: Big Data Ecosystem & Distributed Computing

Lecture 4: Big Data Foundations - Hadoop Ecosystem and Distributed Computing

Learning Objectives

- Understand theoretical foundations of distributed computing and MapReduce
- Explain HDFS architecture, data locality, and fault tolerance mechanisms
- Compare batch vs. stream processing paradigms and their use cases
- Evaluate when big data tools are necessary vs. single-machine optimization

4.1 The Big Data Problem

Big data is characterized by the "Three Vs": Volume (terabytes to petabytes), Velocity (real-time streaming), and Variety (structured, semi-structured, unstructured). Distributed computing addresses this by partitioning data and computation across clusters of commodity machines.

**When Do You Actually Need Big Data Tools?**

Modern single machines with 64-128GB RAM and NVMe storage can process datasets of 50-100GB efficiently using optimized libraries (Polars, DuckDB). The overhead of distributed systems only pays off beyond certain thresholds.

# Before reaching for Spark, try:

import polars as pl

  

df = pl.scan_csv('large_file.csv')  # Lazy - doesn't load yet

result = (

    df.filter(pl.col('date') > '2024-01-01')

    .group_by('category')

    .agg(pl.col('value').sum())

    .collect()  # Execute optimized query plan

)

  

# Or DuckDB - In-process analytical database

import duckdb

result = duckdb.execute("""

    SELECT category, SUM(value) as total

    FROM read_csv_auto('large_file.csv')

    WHERE date > '2024-01-01'

    GROUP BY category

""").df()

4.2 Hadoop Distributed File System (HDFS)

HDFS is designed for storing very large files with streaming access patterns on commodity hardware. Files are split into large blocks (default 128MB) distributed across DataNodes. The NameNode maintains the filesystem namespace and block locations. This architecture optimizes for high throughput rather than low latency.

4.3 MapReduce: The Foundation

# Conceptual MapReduce for word count

# Map phase: (line) -> [(word, 1), (word, 1), ...]

def mapper(line):

    for word in line.split():

        yield (word.lower(), 1)

  

# Reduce phase: (word, [counts]) -> (word, total)

def reducer(word, counts):

    return (word, sum(counts))

Recommended Resources for Lecture 4

|   |   |   |
|---|---|---|
|**Resource**|**Type**|**Coverage**|
|"Hadoop: The Definitive Guide" (White)|Book, Ch. 1-4|HDFS, MapReduce fundamentals|
|"Designing Data-Intensive Apps" (Kleppmann)|Book, Ch. 10|Batch processing theory|
|Google MapReduce Paper (2004)|Research Paper|Original MapReduce design|
|Google GFS Paper (2003)|Research Paper|Distributed filesystem design|

Lecture 5: Apache Spark and PySpark for Data Processing

Learning Objectives

- Understand Spark's architecture: driver, executors, and the DAG execution model
- Implement data transformations using DataFrame and SQL APIs
- Optimize Spark jobs through partitioning, caching, and broadcast variables
- Debug and profile Spark applications using the Spark UI

5.1 Spark Architecture

**The Driver-Executor Model**

Spark applications consist of a driver program that coordinates execution and executors that perform computation. The driver maintains the SparkContext, converts user code into a DAG of stages, and schedules tasks on executors. This architecture enables in-memory computation across multiple operations.

from pyspark.sql import SparkSession

  

# Create SparkSession

spark = SparkSession.builder \

    .appName('DataProcessingApp') \

    .config('spark.executor.memory', '4g') \

    .config('spark.executor.cores', '2') \

    .config('spark.sql.shuffle.partitions', '200') \

    .getOrCreate()

**Lazy Evaluation and the DAG**

# Transformations (lazy - build DAG)

df = spark.read.parquet('s3://bucket/data/')

filtered = df.filter(df['date'] > '2024-01-01')  # Not executed

aggregated = filtered.groupBy('category').sum('value')  # Not executed

  

# Action (triggers execution)

result = aggregated.collect()  # NOW the entire pipeline executes

  

# Explain the execution plan

aggregated.explain(extended=True)

5.2 DataFrame API

**Window Functions for Advanced Analytics**

from pyspark.sql.window import Window

from pyspark.sql import functions as F

  

window_category = Window.partitionBy('category').orderBy(F.desc('sales'))

window_rolling = Window.partitionBy('store').orderBy('date').rowsBetween(-6, 0)

  

df_windowed = df.withColumn(

    'rank_in_category', F.rank().over(window_category)

).withColumn(

    'rolling_7day_avg', F.avg('sales').over(window_rolling)

).withColumn(

    'prev_day_sales', F.lag('sales', 1).over(

        Window.partitionBy('store').orderBy('date'))

)

5.3 Performance Optimization

**Partitioning Strategies**

# Check current partitioning

print(f"Number of partitions: {df.rdd.getNumPartitions()}")

  

# Repartition by column (hash partitioning)

df_repartitioned = df.repartition(100, 'customer_id')

  

# Coalesce to reduce partitions (no shuffle)

df_coalesced = df.coalesce(10)

  

# Check for data skew

df.groupBy(F.spark_partition_id()).count().show()

**Broadcast Joins**

from pyspark.sql.functions import broadcast

  

# Small dimension table

dim_table = spark.read.parquet('dimensions.parquet')  # 10MB

fact_table = spark.read.parquet('facts.parquet')  # 100GB

  

# Broadcast the small table

result = fact_table.join(

    broadcast(dim_table),

    on='dimension_id',

    how='left'

)

Recommended Resources for Lecture 5

|   |   |   |
|---|---|---|
|**Resource**|**Type**|**Coverage**|
|"Learning Spark" (Damji et al., 2nd Ed.)|Book|Comprehensive PySpark guide|
|"Spark: The Definitive Guide" (Chambers)|Book|Deep Spark internals|
|Apache Spark Documentation|Official Docs|API reference, tuning guide|
|"High Performance Spark" (Karau)|Book|Performance optimization|

Lecture 6: SQL Mastery and NoSQL Databases

Learning Objectives

- Write advanced SQL queries including CTEs, window functions, and recursive queries
- Understand query execution plans and optimize SQL performance
- Design and query document databases (MongoDB) and wide-column stores (Cassandra)
- Select appropriate database technologies based on access patterns

6.1 Advanced SQL Techniques

**Common Table Expressions (CTEs)**

-- Multi-level CTE for funnel analysis

WITH user_sessions AS (

    SELECT user_id, session_id, MIN(event_time) as session_start

    FROM events

    WHERE event_date >= CURRENT_DATE - INTERVAL '30 days'

    GROUP BY user_id, session_id

),

funnel_events AS (

    SELECT 

        e.user_id, e.session_id,

        MAX(CASE WHEN event_type = 'page_view' THEN 1 ELSE 0 END) as viewed,

        MAX(CASE WHEN event_type = 'add_to_cart' THEN 1 ELSE 0 END) as added,

        MAX(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) as purchased

    FROM events e JOIN user_sessions s ON e.user_id = s.user_id

    GROUP BY e.user_id, e.session_id

)

SELECT SUM(viewed) as views, SUM(added) as adds, SUM(purchased) as purchases

FROM funnel_events;

6.2 MongoDB: Document Database

**Aggregation Pipeline**

pipeline = [

    {'$match': {'created_at': {'$gte': datetime(2024, 1, 1)}}},

    {'$unwind': '$items'},

    {'$group': {

        '_id': '$items.product_id',

        'total_quantity': {'$sum': '$items.quantity'},

        'total_revenue': {'$sum': {'$multiply': ['$items.price', '$items.quantity']}}

    }},

    {'$sort': {'total_revenue': -1}},

    {'$limit': 10}

]

results = list(db.orders.aggregate(pipeline))

6.3 Apache Cassandra

**Data Model and Partition Strategy**

-- Partition by device and day for time-bounded queries

CREATE TABLE sensor_readings (

    device_id UUID,

    date DATE,

    timestamp TIMESTAMP,

    temperature DOUBLE,

    humidity DOUBLE,

    PRIMARY KEY ((device_id, date), timestamp)

) WITH CLUSTERING ORDER BY (timestamp DESC);

  

-- Efficient query - hits single partition

SELECT * FROM sensor_readings

WHERE device_id = ? AND date = '2024-06-15'

LIMIT 1000;

Recommended Resources for Lecture 6

|   |   |   |
|---|---|---|
|**Resource**|**Type**|**Coverage**|
|"SQL Performance Explained" (Winand)|Book|Indexes, execution plans|
|"MongoDB: The Definitive Guide" (Bradshaw)|Book, 3rd Ed.|Document modeling, aggregation|
|"Cassandra: The Definitive Guide" (Carpenter)|Book, 3rd Ed.|Data modeling, operations|
|Use The Index, Luke (website)|Online Tutorial|SQL indexing deep dive|

Lecture 7: Building Production ETL Pipelines

Learning Objectives

- Design robust ETL/ELT pipelines following software engineering best practices
- Implement workflow orchestration using Apache Airflow
- Build data quality checks and monitoring into pipelines
- Apply incremental processing patterns for efficiency

7.1 ETL vs. ELT Architectures

Extract-Transform-Load (ETL) processes data before loading. Extract-Load-Transform (ELT) leverages the processing power of modern data warehouses (Snowflake, BigQuery, Redshift) where raw data is loaded first, then transformed using SQL within the warehouse. The transformation layer is typically managed by tools like dbt.

-- dbt model example (SQL-based transformation)

{{ config(materialized='incremental', unique_key='order_id') }}

  

WITH source_orders AS (

    SELECT * FROM {{ ref('stg_orders') }}

    {% if is_incremental() %}

    WHERE updated_at > (SELECT MAX(updated_at) FROM {{ this }})

    {% endif %}

)

SELECT o.order_id, o.customer_id, c.customer_segment

FROM source_orders o

LEFT JOIN {{ ref('dim_customers') }} c ON o.customer_id = c.customer_id

7.2 Apache Airflow: Workflow Orchestration

**DAG Definition**

from airflow import DAG

from airflow.operators.python import PythonOperator

from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

from datetime import timedelta

  

default_args = {

    'owner': 'data_engineering',

    'retries': 3,

    'retry_delay': timedelta(minutes=5),

}

  

with DAG(

    dag_id='daily_sales_etl',

    default_args=default_args,

    schedule_interval='0 6 * * *',

    catchup=False,

) as dag:

    validate = PythonOperator(task_id='validate', python_callable=validate_data)

    transform = SparkSubmitOperator(task_id='transform', application='/spark/transform.py')

    load = PythonOperator(task_id='load', python_callable=load_to_warehouse)

    validate >> transform >> load

7.3 Data Quality and Validation

**Great Expectations Framework**

import great_expectations as gx

from great_expectations.dataset import PandasDataset

  

def create_sales_expectations(df):

    ge_df = PandasDataset(df)

    ge_df.expect_column_values_to_not_be_null('order_id')

    ge_df.expect_column_values_to_be_unique('order_id')

    ge_df.expect_column_values_to_be_between('quantity', min_value=1, max_value=1000)

    ge_df.expect_column_values_to_be_between('price', min_value=0.01, max_value=100000)

    return ge_df.validate()

7.4 Incremental Processing Patterns

**Change Data Capture (CDC)**

# Timestamp-based incremental extraction

def extract_incremental(conn, table, last_run_time):

    query = f"""

        SELECT * FROM {table}

        WHERE updated_at > %(last_run)s

        AND updated_at <= %(current_run)s

    """

    current_run = datetime.utcnow()

    df = pd.read_sql(query, conn, params={'last_run': last_run_time, 'current_run': current_run})

    save_watermark(table, current_run)

    return df

Recommended Resources for Lecture 7

|   |   |   |
|---|---|---|
|**Resource**|**Type**|**Coverage**|
|"Fundamentals of Data Engineering" (Reis)|Book|Modern data engineering patterns|
|"Data Pipelines Pocket Reference" (Densmore)|Book|Practical pipeline patterns|
|Apache Airflow Documentation|Official Docs|DAG authoring, operators|
|dbt Documentation & Courses|Official Resources|Modern ELT transformation|
|Great Expectations Documentation|Official Docs|Data quality testing|

Comprehensive Reading List

Core Textbooks

- VanderPlas, J. (2016). Python Data Science Handbook. O'Reilly Media. [Chapters 1-4]
- McKinney, W. (2022). Python for Data Analysis, 3rd Edition. O'Reilly Media.
- Kleppmann, M. (2017). Designing Data-Intensive Applications. O'Reilly Media. [Ch. 3, 10-12]
- Damji, J. et al. (2020). Learning Spark, 2nd Edition. O'Reilly Media.
- Reis, J. & Housley, M. (2022). Fundamentals of Data Engineering. O'Reilly Media.

Supplementary Books

- Gorelick, M. & Ozsvald, I. (2020). High Performance Python, 2nd Ed. O'Reilly.
- Zheng, A. & Casari, A. (2018). Feature Engineering for Machine Learning. O'Reilly.
- White, T. (2015). Hadoop: The Definitive Guide, 4th Ed. O'Reilly.
- Chambers, B. & Zaharia, M. (2018). Spark: The Definitive Guide. O'Reilly.
- Winand, M. (2012). SQL Performance Explained. Self-published.
- Bradshaw, S. et al. (2019). MongoDB: The Definitive Guide, 3rd Ed. O'Reilly.

Research Papers

- Dean, J. & Ghemawat, S. (2004). MapReduce: Simplified Data Processing on Large Clusters. OSDI.
- Ghemawat, S. et al. (2003). The Google File System. SOSP.
- Zaharia, M. et al. (2012). Resilient Distributed Datasets. NSDI.

Online Resources

- Official Documentation: NumPy, Pandas, Scikit-learn, PySpark, Airflow
- MongoDB University (university.mongodb.com) - Free certification courses
- DataStax Academy - Cassandra training
- dbt Learn (courses.getdbt.com) - Modern data transformation
- Use The Index, Luke (use-the-index-luke.com) - SQL indexing

Appendix: Environment Setup Guide

# Create conda environment for course

conda create -n ds_tools python=3.11 -y

conda activate ds_tools

  

# Core data science libraries

pip install numpy pandas scipy scikit-learn matplotlib seaborn

pip install jupyterlab ipykernel ipywidgets

  

# Additional libraries

pip install polars duckdb pyarrow fastparquet

pip install sqlalchemy psycopg2-binary pymongo cassandra-driver

pip install great-expectations missingno category_encoders

  

# PySpark (requires Java 8/11)

pip install pyspark==3.5.0

  

# Register Jupyter kernel

python -m ipykernel install --user --name=ds_tools

_— End of Course Document —_

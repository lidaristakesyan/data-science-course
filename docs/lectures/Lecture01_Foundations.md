# Lecture 01: Data as a Building Block

## PART I: FOUNDATIONS OF DATA ENGINEERING

### Chapter 1: The Data Lifecycle — Understanding the Complete Journey

---

> *"Data is a precious thing and will last longer than the systems themselves."*
> — Tim Berners-Lee, inventor of the World Wide Web

---

## 1.1 Introduction: Why Data Engineering Matters

In the modern era of artificial intelligence and machine learning, there is a common misconception that the magic happens in the models. Data scientists spend years mastering sophisticated algorithms, neural network architectures, and statistical methods. Yet, when they enter the workforce, they discover an uncomfortable truth: **approximately 80% of their time is spent not on modeling, but on finding, cleaning, and preparing data.**

This reality has given rise to a fundamental shift in how we think about building intelligent systems. As Andrew Ng, co-founder of Google Brain and former Chief Scientist at Baidu, has emphasized through his advocacy for "data-centric AI": the quality of your data often matters more than the sophistication of your model. A simple logistic regression trained on excellent, well-curated data will frequently outperform a complex deep learning model trained on messy, poorly understood data.

The emerging field of data-centric AI "emphasizes the systematic engineering of data to build AI systems, shifting our focus from model to data. It is important to note that 'data-centric' differs fundamentally from 'data-driven', as the latter only emphasizes the use of data to guide AI development, which typically still centers on models" (Zha et al., 2023).

This chapter introduces you to the complete data lifecycle—the journey that data takes from its point of origin to its ultimate use in analysis, machine learning, and decision-making. Understanding this lifecycle transforms you from a passive consumer of datasets into an informed practitioner who can:

- Trace the provenance of any dataset and understand its limitations
- Identify where data quality issues originate
- Design better features because you understand the source systems
- Communicate effectively with data engineers and infrastructure teams
- Build your own data pipelines when necessary

---

## 1.2 The Data Lifecycle: A Conceptual Framework

Data does not simply appear in a clean CSV file ready for analysis. It flows through a series of stages, each with its own challenges, tools, and best practices. Understanding this flow—the data lifecycle—provides the mental framework necessary for working effectively with data at any scale.

### 1.2.1 The Eight Stages of the Data Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                  │
│                         THE DATA LIFECYCLE                                       │
│                                                                                  │
│     ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐               │
│     │ GENERATE │───▶│ COLLECT  │───▶│  STORE   │───▶│TRANSFORM │               │
│     │          │    │ (Ingest) │    │ (Persist)│    │ (Process)│               │
│     └──────────┘    └──────────┘    └──────────┘    └──────────┘               │
│                                                            │                    │
│                                                            ▼                    │
│     ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐               │
│     │ ARCHIVE  │◀───│ MAINTAIN │◀───│  SERVE   │◀───│ ANALYZE  │               │
│     │ (Retire) │    │ (Operate)│    │ (Deliver)│    │ (Consume)│               │
│     └──────────┘    └──────────┘    └──────────┘    └──────────┘               │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

Let us examine each stage in detail.

---

### 1.2.2 Stage 1: Generation — Where Data Originates

Every piece of data begins somewhere. Understanding the origin of data—its **provenance**—is essential for assessing its reliability, understanding its structure, and anticipating its limitations.

#### Sources of Data Generation

| Source Category | Examples | Characteristics |
|-----------------|----------|-----------------|
| **Transactional Systems** | Point-of-sale systems, banking transactions, e-commerce orders | Highly structured, real-time, mission-critical accuracy |
| **User Interactions** | Website clicks, mobile app events, search queries | Semi-structured, extremely high volume, behavioral |
| **Operational Systems** | CRM records, ERP data, inventory management | Structured, business-process driven |
| **IoT and Sensors** | Temperature readings, GPS coordinates, machine telemetry | Time-series, continuous streams, potentially noisy |
| **Third-Party Sources** | APIs (weather, financial markets, social media), purchased datasets | Various formats, rate-limited, external dependencies |
| **User-Generated Content** | Reviews, comments, uploaded files, form submissions | Unstructured or semi-structured, requires validation |
| **Machine-Generated Logs** | Application logs, server metrics, error reports | Semi-structured, high volume, essential for debugging |

#### The Importance of Understanding Data Generation

When you receive a dataset, asking "where did this data come from?" is not merely an academic exercise. The generation context determines:

- **Data quality**: Manual entry systems have different error profiles than automated sensors
- **Timeliness**: Real-time systems provide current data; batch extracts may be hours or days old
- **Completeness**: Some systems capture everything; others sample or aggregate
- **Bias**: The mechanism of data collection often introduces systematic biases

> **Key Insight**: The most sophisticated analysis cannot overcome fundamental flaws introduced at the data generation stage. Always trace your data back to its source.

---

### 1.2.3 Stage 2: Collection — Bringing Data Into Your Systems

Once data is generated, it must be captured and brought into your data infrastructure. This process—**data ingestion**—presents its own set of challenges and design decisions.

#### Batch vs. Streaming Ingestion

The two fundamental paradigms for data collection are batch processing and stream processing:

**Batch Ingestion**
```
┌─────────────┐     ┌─────────────────────┐     ┌─────────────┐
│   Source    │────▶│   Scheduled Job     │────▶│   Storage   │
│   System    │     │ (hourly/daily/weekly)│    │   System    │
└─────────────┘     └─────────────────────┘     └─────────────┘
```

Batch ingestion collects data at scheduled intervals. A nightly job might extract all new records from a production database, transform them, and load them into a data warehouse. This approach is:

- Simpler to implement and debug
- More efficient for large volumes (economies of scale)
- Acceptable when real-time data is not required
- Easier to ensure data consistency

**Stream Ingestion**
```
┌─────────────┐     ┌─────────────────────┐     ┌─────────────┐
│   Source    │────▶│   Continuous        │────▶│   Storage   │
│   System    │     │   Stream Processor  │     │   System    │
└─────────────┘     └─────────────────────┘     └─────────────┘
```

Stream ingestion processes data continuously as it arrives. User click events might flow through Apache Kafka into a real-time analytics system within milliseconds. This approach is:

- Essential when freshness matters (fraud detection, real-time recommendations)
- More complex to implement correctly
- Requires careful handling of late-arriving data
- Demands robust error handling for continuous operation

#### Common Ingestion Patterns

| Pattern | Description | Use Case |
|---------|-------------|----------|
| **Full Load** | Extract entire dataset each time | Small reference tables, initial loads |
| **Incremental Load** | Extract only new or changed records | Large transactional tables |
| **Change Data Capture (CDC)** | Capture individual insert/update/delete operations | Real-time replication, audit trails |
| **Event Streaming** | Continuous flow of discrete events | User behavior tracking, IoT data |

---

### 1.2.4 Stage 3: Storage — Persisting Data for Future Use

Once collected, data must be stored in systems appropriate for its structure, volume, and intended use. The choice of storage system has profound implications for cost, query performance, and analytical capabilities.

#### The Storage Landscape

Modern data infrastructure typically includes multiple storage systems, each optimized for different purposes:

**Databases** are optimized for transactional operations—reading and writing individual records quickly and reliably. They store the current state of operational data.

**Data Warehouses** are optimized for analytical queries—aggregating millions of records to answer business questions. They store historical data in structured, pre-defined schemas.

**Data Lakes** store raw data in its native format, without requiring a pre-defined schema. They accommodate structured, semi-structured, and unstructured data at massive scale.

As the MongoDB documentation explains: "A database stores the current data required to power an application. A data warehouse stores current and historical data for one or more systems in a predefined and fixed schema for the purpose of analyzing the data. Data lakes store data in their raw form" (MongoDB, 2024).

We will explore these storage systems in depth in Chapter 3.

---

### 1.2.5 Stage 4: Transformation — Making Data Useful

Raw data is rarely suitable for analysis. The transformation stage encompasses all the processing required to convert raw data into analysis-ready datasets.

#### The Transformation Spectrum

Transformations range from simple to complex:

| Transformation Type | Examples | Complexity |
|---------------------|----------|------------|
| **Cleaning** | Removing duplicates, handling null values, correcting typos | Basic |
| **Standardization** | Consistent date formats, normalized text case, unified units | Basic |
| **Validation** | Enforcing data types, range checks, referential integrity | Moderate |
| **Enrichment** | Adding derived fields, joining with reference data | Moderate |
| **Aggregation** | Summarizing transactions by day/week/month | Moderate |
| **Feature Engineering** | Creating ML features from raw signals | Advanced |
| **Complex Business Logic** | Multi-step calculations, conditional transformations | Advanced |

#### The ETL vs. ELT Paradigm

Historically, transformations occurred during the data movement process itself—**Extract, Transform, Load (ETL)**. Data was extracted from sources, transformed in a dedicated processing system, and then loaded into the destination.

Modern cloud data warehouses have enabled a different approach—**Extract, Load, Transform (ELT)**. Raw data is loaded directly into the destination, and transformations occur within the powerful processing engines of cloud warehouses like Snowflake, BigQuery, or Redshift.

We will examine these paradigms thoroughly in Chapter 6.

---

### 1.2.6 Stage 5: Analysis — Extracting Insights

The analysis stage is where data scientists typically enter the picture. This encompasses:

- **Exploratory Data Analysis (EDA)**: Understanding distributions, relationships, and anomalies
- **Statistical Analysis**: Hypothesis testing, significance testing, causal inference
- **Machine Learning**: Building predictive and prescriptive models
- **Business Intelligence**: Creating reports, dashboards, and visualizations

While this stage is the traditional focus of data science education, its success depends entirely on the quality of the preceding stages.

---

### 1.2.7 Stage 6: Serving — Delivering Value

Analysis has no value unless its results reach decision-makers. The serving stage delivers insights through:

- **Dashboards and Reports**: Self-service access to metrics and KPIs
- **APIs**: Programmatic access for applications
- **Embedded Analytics**: Insights integrated into operational systems
- **ML Model Predictions**: Real-time or batch scoring
- **Alerts and Notifications**: Proactive communication of important changes

---

### 1.2.8 Stage 7: Maintenance — Ensuring Reliability

Data systems require ongoing care to remain reliable and trustworthy:

- **Monitoring**: Tracking data freshness, quality metrics, and system health
- **Alerting**: Automated notification of anomalies or failures
- **Documentation**: Maintaining accurate metadata and lineage information
- **Evolution**: Updating systems as requirements change
- **Incident Response**: Diagnosing and resolving data issues

---

### 1.2.9 Stage 8: Archival — Managing the End of Life

Data does not live forever. The archival stage addresses:

- **Retention Policies**: How long different data types must be kept
- **Cold Storage**: Moving infrequently accessed data to cheaper storage tiers
- **Deletion**: Permanently removing data per policy or regulation
- **Compliance**: Meeting legal requirements like GDPR's "right to erasure"

---

## 1.3 Roles in the Data Ecosystem

The data lifecycle involves multiple specialized roles. Understanding these roles—and how they interact—improves collaboration and helps you identify when to seek expertise.

### 1.3.1 The Core Data Roles

| Role | Primary Responsibility | Key Skills | Tools |
|------|------------------------|------------|-------|
| **Data Engineer** | Build and maintain data infrastructure | Python, SQL, distributed systems | Spark, Airflow, Kafka, Cloud platforms |
| **Data Scientist** | Extract insights and build models | Statistics, ML, programming | Python, R, SQL, ML frameworks |
| **Data Analyst** | Answer business questions with data | SQL, visualization, business acumen | SQL, Tableau, Excel, Looker |
| **Analytics Engineer** | Transform data for analysis | SQL, data modeling, software engineering | dbt, SQL, Git |
| **ML Engineer** | Deploy and scale ML systems | Software engineering, MLOps | Docker, Kubernetes, MLflow |
| **Data Architect** | Design overall data strategy | Systems design, governance | Architecture tools, cloud platforms |

### 1.3.2 The Evolving Boundaries

These roles are not rigid silos. In practice:

- Data scientists increasingly need data engineering skills to be self-sufficient
- Data engineers benefit from understanding analytical use cases
- Analytics engineers emerged to bridge the gap between engineering and analysis
- Small teams often combine multiple roles into "full-stack" data practitioners

> **For Data Scientists**: Developing data engineering competency makes you dramatically more effective. You can prototype pipelines, debug data issues, and work independently on smaller projects—while collaborating more effectively with specialists on larger ones.

---

## 1.4 Event Tracking and Event-Driven Architecture

In the modern data landscape, **events** are the fundamental unit of behavioral data. Understanding event tracking is essential for any data scientist working with user behavior, product analytics, or real-time systems.

### 1.4.1 What is an Event?

An event is a record of something that happened at a specific point in time. Unlike a database record that represents the current state of an entity, an event captures a discrete occurrence.

**The anatomy of an event:**

```
EVENT = WHAT happened + WHEN it happened + WHO did it + WHERE + CONTEXT
```

Consider this example of a well-structured event:

```json
{
  "event_id": "evt_8f14e45f-ceea-367a-a27d-f0c12e45bc",
  "event_name": "product_added_to_cart",
  "timestamp": "2025-11-29T14:32:15.123Z",
  
  "user_id": "usr_abc123",
  "anonymous_id": "anon_device_fingerprint_xyz",
  "session_id": "sess_789xyz",
  
  "properties": {
    "product_id": "prod_456",
    "product_name": "Wireless Noise-Canceling Headphones",
    "price": 249.99,
    "currency": "USD",
    "quantity": 1,
    "category": "Electronics/Audio",
    "brand": "SoundMax"
  },
  
  "context": {
    "page_url": "https://shop.example.com/products/wireless-headphones",
    "page_title": "Wireless Headphones - SoundMax",
    "referrer": "https://www.google.com/search?q=best+wireless+headphones",
    "device": {
      "type": "mobile",
      "os": "iOS",
      "os_version": "17.1",
      "browser": "Safari",
      "screen_width": 390,
      "screen_height": 844
    },
    "location": {
      "country": "United States",
      "region": "California",
      "city": "San Francisco",
      "timezone": "America/Los_Angeles"
    },
    "campaign": {
      "source": "google",
      "medium": "cpc",
      "campaign": "holiday_sale_2025",
      "term": "wireless headphones"
    }
  },
  
  "metadata": {
    "sent_at": "2025-11-29T14:32:15.456Z",
    "received_at": "2025-11-29T14:32:15.789Z",
    "sdk_version": "2.5.1",
    "library": "analytics.js"
  }
}
```

### 1.4.2 The Event-Driven Architecture

Events flow through a system architecture designed to capture, transport, and process them reliably at scale.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         EVENT-DRIVEN ARCHITECTURE                                │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                          EVENT PRODUCERS                                    │ │
│  │                                                                             │ │
│  │   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐     │ │
│  │   │   Web   │   │ Mobile  │   │ Backend │   │   IoT   │   │  Third  │     │ │
│  │   │  Apps   │   │  Apps   │   │Services │   │ Devices │   │  Party  │     │ │
│  │   └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘     │ │
│  └────────┼─────────────┼─────────────┼─────────────┼─────────────┼──────────┘ │
│           │             │             │             │             │            │
│           └─────────────┴──────┬──────┴─────────────┴─────────────┘            │
│                                │                                                │
│                                ▼                                                │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                       EVENT COLLECTION LAYER                                │ │
│  │                                                                             │ │
│  │   • SDK/Tracking Libraries (Segment, Amplitude, Mixpanel, custom)          │ │
│  │   • API Gateways and Webhook Receivers                                      │ │
│  │   • Validation, Enrichment, and Routing                                     │ │
│  │                                                                             │ │
│  └────────────────────────────────────┬───────────────────────────────────────┘ │
│                                       │                                         │
│                                       ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                       EVENT STREAMING LAYER                                 │ │
│  │                                                                             │ │
│  │   ┌──────────────────────────────────────────────────────────────────────┐ │ │
│  │   │                    MESSAGE QUEUE / EVENT BUS                          │ │ │
│  │   │                                                                       │ │ │
│  │   │    Apache Kafka  •  Amazon Kinesis  •  Google Pub/Sub  •  RabbitMQ   │ │ │
│  │   │                                                                       │ │ │
│  │   │    • Durability: Events persisted to disk                            │ │ │
│  │   │    • Scalability: Partitioned for parallel processing                │ │ │
│  │   │    • Decoupling: Producers and consumers operate independently       │ │ │
│  │   └──────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────┬───────────────────────────────────────┘ │
│                                       │                                         │
│           ┌───────────────────────────┼───────────────────────────┐             │
│           │                           │                           │             │
│           ▼                           ▼                           ▼             │
│  ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐       │
│  │   REAL-TIME     │       │     BATCH       │       │    DERIVED      │       │
│  │   PROCESSING    │       │    STORAGE      │       │     DATA        │       │
│  │                 │       │                 │       │                 │       │
│  │ • Live dashboards│      │ • Data Lake     │       │ • Data Warehouse│       │
│  │ • Alerting      │       │ • Raw event     │       │ • Aggregated    │       │
│  │ • Real-time ML  │       │   archive       │       │   tables        │       │
│  │ • Fraud detection│      │ • Parquet files │       │ • ML features   │       │
│  └─────────────────┘       └─────────────────┘       └─────────────────┘       │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 1.4.3 Designing an Event Taxonomy

A well-designed event taxonomy—the organized catalog of all events your system tracks—is crucial for maintainability and analytical utility.

#### Event Naming Conventions

Consistency in naming enables self-service analytics and reduces confusion:

**Recommended Pattern: `object_action`**

```
✓ product_viewed
✓ product_added_to_cart
✓ cart_viewed
✓ checkout_started
✓ checkout_step_completed
✓ order_completed
✓ order_refunded
✓ user_signed_up
✓ user_logged_in
✓ subscription_started
✓ subscription_cancelled
```

**Patterns to Avoid:**

```
✗ click                      (too generic)
✗ event_1, event_2           (meaningless)
✗ productAddedToCart         (inconsistent casing)
✗ add_to_cart_button_clicked_on_product_detail_page  (too specific)
✗ tracking_event             (redundant)
```

#### Event Categories

Organizing events into categories improves discoverability:

| Category | Purpose | Example Events |
|----------|---------|----------------|
| **Acquisition** | How users arrive | `campaign_clicked`, `referral_received` |
| **Activation** | First value moments | `user_signed_up`, `onboarding_completed` |
| **Engagement** | Core product usage | `feature_used`, `content_viewed` |
| **Conversion** | Revenue events | `subscription_started`, `purchase_completed` |
| **Retention** | Return behavior | `user_returned`, `streak_maintained` |
| **System** | Technical events | `error_occurred`, `page_loaded` |

### 1.4.4 Event Properties and Context

The value of an event lies not just in knowing it occurred, but in the rich context captured alongside it.

#### Properties vs. Context

**Properties** are specific to the event type:

```json
// For "product_added_to_cart"
"properties": {
  "product_id": "prod_456",
  "product_name": "Wireless Headphones",
  "price": 249.99,
  "quantity": 1,
  "category": "Electronics"
}
```

**Context** is captured automatically for all events:

```json
// Same for every event
"context": {
  "page_url": "...",
  "device": { "type": "mobile", "os": "iOS" },
  "location": { "country": "US" },
  "campaign": { "source": "google", "medium": "cpc" }
}
```

#### Identity Resolution

Users interact with products across devices and sessions, often before creating an account. Event systems must handle:

- **Anonymous ID**: Device or browser identifier before login
- **User ID**: Authenticated identifier after login
- **Session ID**: Groups events within a single visit
- **Identity Stitching**: Connecting anonymous activity to known users after authentication

```json
// Before login
{
  "event_name": "product_viewed",
  "anonymous_id": "anon_device_123",
  "user_id": null
}

// After login (same session)
{
  "event_name": "user_logged_in",
  "anonymous_id": "anon_device_123",
  "user_id": "usr_abc456"
}

// Subsequent events (user identified)
{
  "event_name": "product_purchased",
  "anonymous_id": "anon_device_123",
  "user_id": "usr_abc456"
}
```

### 1.4.5 From Events to Features: The Data Science Connection

For data scientists, events are the raw material for behavioral features:

```python
import pandas as pd

# Raw events
events = pd.DataFrame([
    {"user_id": "u1", "event": "page_view", "timestamp": "2025-01-01 10:00:00", 
     "properties": {"page": "home"}},
    {"user_id": "u1", "event": "product_viewed", "timestamp": "2025-01-01 10:05:00",
     "properties": {"product_id": "p1", "price": 99.99}},
    {"user_id": "u1", "event": "add_to_cart", "timestamp": "2025-01-01 10:07:00",
     "properties": {"product_id": "p1"}},
    {"user_id": "u1", "event": "purchase", "timestamp": "2025-01-01 10:15:00",
     "properties": {"order_total": 99.99}},
    {"user_id": "u2", "event": "page_view", "timestamp": "2025-01-01 11:00:00",
     "properties": {"page": "home"}},
    {"user_id": "u2", "event": "page_view", "timestamp": "2025-01-01 11:30:00",
     "properties": {"page": "pricing"}},
])

# Transform to user-level features for ML
user_features = events.groupby('user_id').agg(
    total_events=('event', 'count'),
    unique_event_types=('event', 'nunique'),
    session_duration_minutes=('timestamp', lambda x: 
        (pd.to_datetime(x).max() - pd.to_datetime(x).min()).seconds / 60),
    viewed_products=('event', lambda x: (x == 'product_viewed').sum()),
    made_purchase=('event', lambda x: 'purchase' in x.values),
).reset_index()

print(user_features)
```

Output:
```
  user_id  total_events  unique_event_types  session_duration_minutes  viewed_products  made_purchase
0      u1             4                   4                      15.0                1           True
1      u2             2                   1                      30.0                0          False
```

---

## 1.5 Summary and Key Concepts

This chapter established the foundational concepts necessary for understanding data engineering:

### The Data Lifecycle
Data flows through eight stages: **Generation → Collection → Storage → Transformation → Analysis → Serving → Maintenance → Archival**. Understanding this complete journey enables you to trace data quality issues, design better systems, and collaborate effectively across roles.

### Roles in the Data Ecosystem
Multiple specialized roles contribute to the data lifecycle. As a data scientist, developing data engineering literacy makes you more effective and self-sufficient.

### Events as Foundational Data
Modern behavioral data is captured through events—discrete records of actions with timestamps, identities, and rich contextual properties. Well-designed event architectures enable powerful analytics and machine learning.

---

## 1.6 Further Reading and Resources

### Books
- **Reis, J. & Housley, M. (2022). *Fundamentals of Data Engineering*.** O'Reilly Media. The definitive introduction to data engineering concepts and practices.
- **Kleppmann, M. (2017). *Designing Data-Intensive Applications*.** O'Reilly Media. Deep exploration of the principles underlying modern data systems.

### Papers
- **Zha, D. et al. (2023). "Data-centric Artificial Intelligence: A Survey."** arXiv:2303.10158. Comprehensive overview of data-centric approaches to AI. https://arxiv.org/abs/2303.10158
- **Nazabal, A. et al. (2020). "Data Engineering for Data Analytics: A Classification of the Issues, and Case Studies."** arXiv:2004.12929. Classification of data engineering tasks with practical examples. https://arxiv.org/abs/2004.12929

### Online Resources
- **Data Engineering Handbook** (GitHub): Comprehensive collection of resources, tools, and learning paths. https://github.com/DataExpert-io/data-engineer-handbook
- **DataTalks.Club Data Engineering Zoomcamp**: Free, project-based course covering modern data engineering tools and practices. https://github.com/DataTalksClub/data-engineering-zoomcamp

---

## 1.7 Exercises

**Exercise 1.1: Data Lifecycle Mapping**
Choose a dataset you have worked with recently. Trace its journey through the data lifecycle:
- Where was the data originally generated?
- How was it collected and by whom?
- Where is it stored?
- What transformations were applied before you received it?
- What limitations might exist due to decisions made at each stage?

**Exercise 1.2: Event Schema Design**
Design an event tracking schema for one of the following scenarios:
- A music streaming application (like Spotify)
- A food delivery service (like DoorDash)
- An online learning platform (like Coursera)

For each scenario:
1. Identify 10-15 key events to track
2. Define properties for each event
3. Document your naming conventions
4. Consider what context should be captured automatically

**Exercise 1.3: Event to Feature Transformation**
Given the following raw events, create five user-level features that could be useful for predicting user churn:

```json
{"user_id": "u1", "event": "app_opened", "timestamp": "2025-01-01T10:00:00Z"}
{"user_id": "u1", "event": "content_viewed", "timestamp": "2025-01-01T10:05:00Z", "properties": {"content_type": "video", "duration_seconds": 300}}
{"user_id": "u1", "event": "content_liked", "timestamp": "2025-01-01T10:10:00Z"}
{"user_id": "u1", "event": "app_closed", "timestamp": "2025-01-01T10:30:00Z"}
{"user_id": "u2", "event": "app_opened", "timestamp": "2025-01-01T11:00:00Z"}
{"user_id": "u2", "event": "error_occurred", "timestamp": "2025-01-01T11:01:00Z"}
{"user_id": "u2", "event": "app_closed", "timestamp": "2025-01-01T11:02:00Z"}
```

---

# Chapter 2: Data Types and File Formats — The Building Blocks

> *"The goal is to turn data into information, and information into insight."*
> — Carly Fiorina, former CEO of Hewlett-Packard

---

## 2.1 Introduction: Why Format Matters

When you load a dataset into pandas with `pd.read_csv()`, a complex series of operations occurs behind the scenes. The file is read from disk, bytes are decoded into characters, delimiters are detected, columns are parsed, and data types are inferred. This seemingly simple operation embodies fundamental decisions about how data is represented, stored, and accessed.

The choice of data format affects:

- **Storage costs**: Some formats are 10x more compact than others
- **Query performance**: The right format can make queries 100x faster
- **Processing efficiency**: Format determines how data flows through pipelines
- **Interoperability**: Different systems support different formats
- **Schema evolution**: Some formats handle changes gracefully; others break

This chapter provides a comprehensive understanding of data types and file formats—knowledge that will inform every data engineering decision you make.

---

## 2.2 The Data Structure Spectrum

Not all data fits neatly into spreadsheet rows and columns. Understanding the spectrum of data structures helps you choose appropriate storage and processing strategies.

### 2.2.1 Structured Data

**Structured data** conforms to a fixed schema with predefined columns and data types. Every record has the same fields in the same order.

```
┌────────────┬─────────────────┬─────────┬────────────┬───────────┐
│ customer_id│ name            │ age     │ email      │ signup_dt │
├────────────┼─────────────────┼─────────┼────────────┼───────────┤
│ 1          │ Alice Smith     │ 28      │ alice@...  │ 2024-01-15│
│ 2          │ Bob Johnson     │ 35      │ bob@...    │ 2024-02-20│
│ 3          │ Carol Williams  │ 42      │ carol@...  │ 2024-03-10│
└────────────┴─────────────────┴─────────┴────────────┴───────────┘
```

**Characteristics:**
- Schema defined before data insertion (schema-on-write)
- Fixed columns with enforced data types
- Naturally maps to relational database tables
- Easily queried with SQL
- Efficient storage due to predictable structure

**Common sources:** Relational databases, ERP systems, financial records, CRM data

### 2.2.2 Semi-Structured Data

**Semi-structured data** has some organizational properties but does not conform to a rigid schema. It is self-describing, with tags or keys that identify data elements.

```json
{
  "customer_id": 1,
  "name": "Alice Smith",
  "age": 28,
  "contacts": {
    "email": "alice@example.com",
    "phone": "+1-555-0123",
    "preferences": {
      "newsletter": true,
      "sms": false
    }
  },
  "orders": [
    {"order_id": "o1", "total": 150.00, "items": 3},
    {"order_id": "o2", "total": 89.99, "items": 1}
  ],
  "tags": ["premium", "early_adopter"]
}
```

**Characteristics:**
- Flexible schema (fields can vary between records)
- Supports nested structures (objects within objects)
- Self-describing (keys explain the data)
- Schema can evolve without breaking existing data
- Requires more sophisticated querying

**Common sources:** JSON APIs, event tracking data, log files, NoSQL databases, configuration files

### 2.2.3 Unstructured Data

**Unstructured data** lacks any predefined organizational structure. It must be processed or analyzed to extract meaningful information.

**Examples:**
- **Text**: Emails, documents, social media posts, customer reviews
- **Images**: Photographs, scanned documents, medical imaging
- **Audio**: Call recordings, podcasts, voice messages
- **Video**: Surveillance footage, user-generated content, training videos

**Characteristics:**
- No inherent schema
- Requires specialized processing (NLP, computer vision, etc.)
- Often the largest volume of organizational data
- High potential value, but difficult to extract

**Processing unstructured data:**

```python
# Text analysis example
raw_text = "I absolutely love this product! Fast shipping and great quality."

# After NLP processing, we extract structured information:
processed = {
    "sentiment": "positive",
    "sentiment_score": 0.92,
    "topics": ["product_quality", "shipping"],
    "entities": [],
    "language": "en"
}
```

### 2.2.4 The Queryability Spectrum

```
STRUCTURED                SEMI-STRUCTURED              UNSTRUCTURED
    │                           │                            │
    │  SELECT * FROM users      │  $.orders[*].total         │  [Requires ML/AI]
    │  WHERE age > 25           │  WHERE name = 'Alice'      │  
    │                           │                            │
    ◀──────────────────────────────────────────────────────────▶
    Easy to Query                                    Difficult to Query
    Rigid Schema                                     No Schema
    Small Storage                                    Large Storage
```

---

## 2.3 File Formats: A Comprehensive Guide

The choice of file format is one of the most consequential decisions in data engineering. This section provides deep coverage of the formats you will encounter.

### 2.3.1 Text-Based Formats

#### CSV (Comma-Separated Values)

CSV is the oldest and most universal data exchange format. Its simplicity is both its greatest strength and its most significant limitation.

**Structure:**
```csv
customer_id,name,age,signup_date,is_active
1,Alice Smith,28,2024-01-15,true
2,"Johnson, Bob",35,2024-02-20,true
3,Carol Williams,42,2024-03-10,false
```

**Technical Details:**
- **Encoding**: Typically UTF-8, but varies (beware of Excel's default encoding)
- **Delimiter**: Comma by default, but tab (`\t`) and pipe (`|`) are common alternatives
- **Quoting**: Fields containing delimiters or newlines must be quoted
- **Headers**: First row typically contains column names (but not guaranteed)
- **Types**: All values stored as strings; types must be inferred or specified

**Advantages:**
- Universal compatibility (every tool can read CSV)
- Human readable and editable
- Simple to generate and parse
- No dependencies or special libraries required

**Disadvantages:**

As noted in technical discussions: "CSV is just a string, meaning the dataset is larger by storing all characters according to the file-encoding; there is no type-information or schema associated with the data, and it will always be parsed while deserialized" (Stack Overflow, 2022).

Additional limitations:
- No standard specification (many dialects exist)
- No support for nested or hierarchical data
- Poor compression (text is verbose)
- Ambiguous handling of null values
- Date/time formats vary wildly
- Encoding issues are common

**When to use:**
- Small datasets (under 100MB)
- Data exchange with non-technical stakeholders
- Legacy system integration
- Quick prototyping and exploration

**Python usage:**
```python
import pandas as pd

# Reading with explicit options for reliability
df = pd.read_csv(
    'data.csv',
    encoding='utf-8',
    dtype={'customer_id': 'int32', 'name': 'string'},
    parse_dates=['signup_date'],
    na_values=['', 'NULL', 'N/A']
)

# Writing with consistent formatting
df.to_csv('output.csv', index=False, date_format='%Y-%m-%d')
```

---

#### JSON (JavaScript Object Notation)

JSON emerged from JavaScript but has become the lingua franca of web APIs and configuration files. Its ability to represent nested structures makes it far more expressive than CSV.

**Structure:**
```json
{
  "customers": [
    {
      "customer_id": 1,
      "name": "Alice Smith",
      "age": 28,
      "contacts": {
        "email": "alice@example.com",
        "phone": "+1-555-0123"
      },
      "orders": [
        {"order_id": "o1", "total": 150.00},
        {"order_id": "o2", "total": 89.99}
      ]
    }
  ],
  "metadata": {
    "extracted_at": "2025-11-29T10:30:00Z",
    "record_count": 1
  }
}
```

**Technical Details:**
- **Data types**: strings, numbers, booleans, null, arrays, objects
- **Encoding**: Must be UTF-8 (per RFC 8259)
- **No comments**: JSON does not support comments (though some parsers allow them)
- **Strict syntax**: Trailing commas and single quotes are invalid

**Variants:**
- **JSON Lines (JSONL/NDJSON)**: One JSON object per line, enabling streaming
- **GeoJSON**: JSON format for geographic data
- **JSON Schema**: Specification for validating JSON structure

**Advantages:**
- Human readable (more so than CSV for complex data)
- Native support for nested structures
- Self-describing (keys provide context)
- Universal support in programming languages
- Natural fit for API responses

**Disadvantages:**
- Verbose (keys repeated for every record)
- No native date/time type (dates are strings)
- Slow for large-scale analytics
- Memory-intensive for large files (must often parse entirely)

**When to use:**
- API responses and requests
- Configuration files
- Document storage
- Data interchange between systems
- Event data (before analytical processing)

**Python usage:**
```python
import json
import pandas as pd
from pandas import json_normalize

# Reading JSON
with open('data.json', 'r') as f:
    data = json.load(f)

# Flattening nested JSON to DataFrame
df = json_normalize(
    data['customers'],
    record_path='orders',
    meta=['customer_id', 'name', ['contacts', 'email']],
    meta_prefix='customer_'
)

# Reading JSON Lines (streaming-friendly)
df = pd.read_json('data.jsonl', lines=True)

# Writing JSON
df.to_json('output.json', orient='records', indent=2, date_format='iso')
```

---

### 2.3.2 Binary Columnar Formats

Binary columnar formats represent a paradigm shift from row-based storage. They are essential for modern analytics and big data processing.

#### Understanding Columnar Storage

The fundamental insight of columnar storage is that analytical queries typically access many rows but few columns:

```sql
-- This query touches only 2 columns out of potentially dozens
SELECT AVG(price), COUNT(*) 
FROM transactions 
WHERE transaction_date >= '2025-01-01'
```

**Row-based storage (CSV, Avro):**
```
Row 1: [id=1, name="Alice", age=28, city="NYC", salary=75000, ...]
Row 2: [id=2, name="Bob", age=35, city="LA", salary=82000, ...]
Row 3: [id=3, name="Carol", age=42, city="Chicago", salary=91000, ...]

To read 'age' column: Must scan through entire rows
```

**Column-based storage (Parquet, ORC):**
```
id column:     [1, 2, 3, 4, 5, ...]
name column:   ["Alice", "Bob", "Carol", ...]
age column:    [28, 35, 42, ...]    ← Read only this!
city column:   ["NYC", "LA", "Chicago", ...]
salary column: [75000, 82000, 91000, ...]

To read 'age' column: Read only the age column block
```

**Benefits of columnar storage:**
1. **Column pruning**: Read only the columns needed for a query
2. **Better compression**: Similar values in a column compress well together
3. **Vectorized processing**: CPUs can process columns more efficiently
4. **Predicate pushdown**: Skip entire row groups based on column statistics

---

#### Parquet

Apache Parquet is the dominant columnar format for analytics and the default format for Apache Spark. Understanding Parquet is essential for any data practitioner.

**File Structure:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           PARQUET FILE                                   │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                        ROW GROUP 1                                 │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │  │
│  │  │ Column Chunk│  │ Column Chunk│  │ Column Chunk│                │  │
│  │  │   (id)      │  │   (name)    │  │   (age)     │                │  │
│  │  │             │  │             │  │             │                │  │
│  │  │ • Data Pages│  │ • Data Pages│  │ • Data Pages│                │  │
│  │  │ • Dict Page │  │ • Dict Page │  │ • Statistics│                │  │
│  │  │ • Statistics│  │ • Statistics│  │   (min/max) │                │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                        ROW GROUP 2                                 │  │
│  │  (same structure as Row Group 1)                                   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                          FOOTER                                    │  │
│  │  • File metadata                                                   │  │
│  │  • Schema definition                                               │  │
│  │  • Row group metadata                                              │  │
│  │  • Column chunk locations                                          │  │
│  │  • Key-value metadata (custom)                                     │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Concepts:**

- **Row Groups**: Horizontal partitions of the data (typically 128MB-1GB each)
- **Column Chunks**: A column's data within a row group
- **Pages**: The smallest unit of storage within a column chunk (typically 1MB)
- **Footer**: Metadata including schema, statistics, and locations

**Schema Embedded in File:**

"Parquet stores the file schema in the file metadata. CSV files don't store file metadata, so readers need to either be supplied with the schema or the schema needs to be inferred" (MrPowers, Stack Overflow).

**Compression and Encoding:**

Parquet supports multiple compression codecs:
- **Snappy**: Fast compression/decompression, moderate ratio (default)
- **Gzip**: Higher compression ratio, slower
- **LZ4**: Very fast, lower ratio
- **Zstd**: Good balance of speed and ratio

Parquet also uses encoding schemes optimized for each data type:
- **Dictionary encoding**: For columns with repeated values
- **Run-length encoding (RLE)**: For sequences of repeated values
- **Delta encoding**: For sorted or incrementing values

**Predicate Pushdown:**

Parquet's column statistics enable query engines to skip irrelevant data:

```python
# Query: SELECT * FROM sales WHERE sale_date = '2025-11-29'

# Row Group 1 statistics: sale_date min='2025-01-01', max='2025-06-30'
#   → SKIP (date not in range)

# Row Group 2 statistics: sale_date min='2025-07-01', max='2025-12-31'
#   → READ (date might be in range)
```

**When to use Parquet:**
- Analytical queries and data warehousing
- Data lake storage
- Spark, Presto, Athena, BigQuery workloads
- Any dataset over 100MB used for analysis
- Long-term data archival

"Parquet is a default data file format for Spark" (Towards Data Science, 2023).

**Python usage:**
```python
import pandas as pd
import pyarrow.parquet as pq

# Reading Parquet
df = pd.read_parquet('data.parquet')

# Reading specific columns only (column pruning)
df = pd.read_parquet('data.parquet', columns=['customer_id', 'age', 'revenue'])

# Writing Parquet with options
df.to_parquet(
    'output.parquet',
    engine='pyarrow',
    compression='snappy',
    index=False
)

# Advanced: Reading with filters (predicate pushdown)
df = pd.read_parquet(
    'data.parquet',
    filters=[('year', '=', 2025), ('country', 'in', ['US', 'UK'])]
)

# PySpark usage
spark_df = spark.read.parquet('s3://bucket/data/')
spark_df.write.parquet('s3://bucket/output/', mode='overwrite')
```

---

#### Avro

Apache Avro is a row-based binary format designed for data serialization, particularly in streaming contexts.

**Structure:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            AVRO FILE                                     │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                           HEADER                                   │  │
│  │  • Magic bytes (Obj1)                                              │  │
│  │  • Schema (JSON)                                                   │  │
│  │  • Sync marker (16 bytes)                                          │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                        DATA BLOCK 1                                │  │
│  │  • Object count                                                    │  │
│  │  • Serialized objects (compressed)                                 │  │
│  │  • Sync marker                                                     │  │
│  │                                                                    │  │
│  │  Row 1: [1, "Alice", 28, "2024-01-15"]                            │  │
│  │  Row 2: [2, "Bob", 35, "2024-02-20"]                               │  │
│  │  ...                                                               │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                        DATA BLOCK 2                                │  │
│  │  (same structure)                                                  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

**Schema Definition:**

Avro schemas are defined in JSON:

```json
{
  "type": "record",
  "name": "Customer",
  "namespace": "com.example",
  "fields": [
    {"name": "customer_id", "type": "int"},
    {"name": "name", "type": "string"},
    {"name": "age", "type": ["null", "int"], "default": null},
    {"name": "signup_date", "type": {
      "type": "int",
      "logicalType": "date"
    }},
    {"name": "tags", "type": {"type": "array", "items": "string"}}
  ]
}
```

**Schema Evolution:**

Avro's killer feature is robust schema evolution. You can:
- Add fields (with defaults) without breaking readers
- Remove fields without breaking readers (if they have defaults)
- Rename fields using aliases

This makes Avro ideal for streaming systems where producers and consumers may be updated independently.

**Compression:**

"Avro uses a binary format which benefits data compaction. Binary data is highly compact compared to JSON or XML formats. Hence the speed of serialization and deserialization also increases" (Hevo Data, 2024).

**When to use Avro:**
- Apache Kafka message serialization
- Real-time streaming pipelines
- Systems requiring schema evolution
- Write-heavy workloads
- Data interchange between services

**Python usage:**
```python
from fastavro import writer, reader, parse_schema

# Define schema
schema = {
    "type": "record",
    "name": "Customer",
    "fields": [
        {"name": "customer_id", "type": "int"},
        {"name": "name", "type": "string"},
        {"name": "age", "type": "int"}
    ]
}
parsed_schema = parse_schema(schema)

# Write Avro
records = [
    {"customer_id": 1, "name": "Alice", "age": 28},
    {"customer_id": 2, "name": "Bob", "age": 35}
]

with open('customers.avro', 'wb') as f:
    writer(f, parsed_schema, records)

# Read Avro
with open('customers.avro', 'rb') as f:
    for record in reader(f):
        print(record)
```

---

### 2.3.3 Format Comparison

**Storage Efficiency:**

Testing with identical data reveals dramatic differences:

"Both CSV and JSON are losing a lot compared to Avro and Parquet, however, this is expected because both Avro and Parquet are binary formats (they also use compression) while CSV and JSON are not compressed" (DataCrump, 2023).

| Format | Relative Size | Notes |
|--------|---------------|-------|
| JSON | 140% | Keys repeated, no compression |
| CSV | 100% | Baseline (uncompressed) |
| CSV (gzip) | 15-25% | Compressed text |
| Avro | 25-40% | Binary, row-based |
| Parquet | 10-20% | Binary, columnar, excellent compression |

**Query Performance:**

For analytical queries selecting specific columns:

| Format | Relative Query Time |
|--------|---------------------|
| CSV | 100% (baseline) |
| JSON | 120-150% |
| Avro | 50-70% |
| Parquet | 5-15% |

The dramatic improvement with Parquet comes from column pruning and predicate pushdown.

**Decision Framework:**

"CSV and JSON are suitable for small datasets (<1,000,000 rows) or quick implementations, while Parquet, Avro, or ORC are better for large datasets with specific data behaviors" (Medium, 2024).

```
                    ┌─────────────────────────────────────────┐
                    │         FORMAT SELECTION GUIDE          │
                    └─────────────────────────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │        Is the data > 100MB?             │
                    └─────────────────────────────────────────┘
                                       │
                      ┌────────────────┴────────────────┐
                      │ NO                              │ YES
                      ▼                                 ▼
        ┌──────────────────────┐          ┌──────────────────────┐
        │ Need human editing?  │          │ Streaming/real-time? │
        └──────────────────────┘          └──────────────────────┘
                      │                              │
           ┌──────────┴──────────┐        ┌─────────┴─────────┐
           │ YES            NO   │        │ YES           NO  │
           ▼                ▼    │        ▼               ▼   │
        ┌──────┐        ┌──────┐│     ┌──────┐       ┌──────┐│
        │ CSV  │        │ JSON ││     │ AVRO │       │PARQUET│
        │      │        │      ││     │      │       │      ││
        └──────┘        └──────┘│     └──────┘       └──────┘│
                                │                            │
                                └────────────────────────────┘
```

---

## 2.4 Data Types: Precision and Efficiency

Within any format, data is represented using specific types. Understanding types improves both correctness and performance.

### 2.4.1 Numeric Types

| Type | Size | Range | Use Case |
|------|------|-------|----------|
| `int8` / `tinyint` | 1 byte | -128 to 127 | Age, small counts |
| `int16` / `smallint` | 2 bytes | -32,768 to 32,767 | Year, medium counts |
| `int32` / `int` | 4 bytes | ±2.1 billion | Most integers |
| `int64` / `bigint` | 8 bytes | ±9.2 quintillion | Large IDs, timestamps |
| `float32` / `float` | 4 bytes | ~7 decimal digits | Approximate values |
| `float64` / `double` | 8 bytes | ~15 decimal digits | Scientific, financial |
| `decimal` | Variable | Exact precision | Currency, percentages |

**Choosing the right numeric type:**

```python
import pandas as pd
import numpy as np

# Memory comparison
n = 1_000_000

# Using default int64 (8 bytes × 1M = 8MB)
ages_int64 = pd.Series(np.random.randint(0, 100, n), dtype='int64')
print(f"int64: {ages_int64.memory_usage() / 1e6:.1f} MB")  # 8.0 MB

# Using int8 (1 byte × 1M = 1MB) - ages fit in 0-127
ages_int8 = ages_int64.astype('int8')
print(f"int8: {ages_int8.memory_usage() / 1e6:.1f} MB")    # 1.0 MB

# 8x memory savings!
```

### 2.4.2 String Types

Strings are often the largest memory consumers in datasets:

```python
# String optimization with categories
df = pd.DataFrame({
    'country': np.random.choice(['USA', 'UK', 'Germany', 'France'], 1_000_000)
})

# Default object dtype
print(f"Object: {df['country'].memory_usage(deep=True) / 1e6:.1f} MB")  # ~64 MB

# Category dtype (for low-cardinality strings)
df['country'] = df['country'].astype('category')
print(f"Category: {df['country'].memory_usage(deep=True) / 1e6:.1f} MB")  # ~1 MB
```

### 2.4.3 Date and Time Types

Temporal data requires careful handling:

| Type | Storage | Precision | Example |
|------|---------|-----------|---------|
| `date` | 4 bytes | Day | 2025-11-29 |
| `datetime64[ns]` | 8 bytes | Nanosecond | 2025-11-29 14:30:15.123456789 |
| `timestamp` | 8 bytes | Varies | Unix epoch milliseconds |
| `time` | Variable | Sub-second | 14:30:15.123 |
| `interval` | Variable | Duration | 3 days, 4 hours |

**Best practices:**
- Store timestamps in UTC
- Use ISO 8601 format for string representation
- Be explicit about timezone handling
- Choose appropriate precision (do you need nanoseconds?)

```python
import pandas as pd

# Parse dates explicitly
df = pd.read_csv('data.csv', parse_dates=['created_at', 'updated_at'])

# Handle timezones
df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
df['created_at_local'] = df['created_at'].dt.tz_convert('America/New_York')
```

### 2.4.4 Boolean and Null Types

- **Boolean**: `true`/`false` - ideally 1 bit, often stored as 1 byte
- **Null/None**: Represents missing data - handling varies by system

**Null handling strategies:**

| Strategy | When to Use |
|----------|-------------|
| Keep as null | Value is genuinely unknown |
| Default value | Reasonable default exists (0 for counts) |
| Sentinel value | -1, "UNKNOWN", empty string |
| Imputation | Statistical estimate appropriate |
| Exclusion | Analysis can proceed without the record |

---

## 2.5 Working with Nested Data

Semi-structured data with nested objects and arrays requires special handling to prepare for analysis.

### 2.5.1 The Nested Data Challenge

Consider this common e-commerce data structure:

```json
{
  "order_id": "ord_12345",
  "customer": {
    "id": "cust_789",
    "name": "Alice Smith",
    "address": {
      "street": "123 Main St",
      "city": "San Francisco",
      "state": "CA",
      "zip": "94102"
    }
  },
  "items": [
    {
      "product_id": "prod_001",
      "name": "Wireless Headphones",
      "price": 249.99,
      "quantity": 1
    },
    {
      "product_id": "prod_002",
      "name": "Phone Case",
      "price": 29.99,
      "quantity": 2
    }
  ],
  "totals": {
    "subtotal": 309.97,
    "tax": 27.90,
    "shipping": 0,
    "total": 337.87
  }
}
```

This cannot be directly loaded into a flat table. We need **flattening** strategies.

### 2.5.2 Flattening Strategies

**Strategy 1: Dot notation flattening (nested objects)**

```python
from pandas import json_normalize

data = {...}  # The JSON above

# Flatten nested objects
df = json_normalize(data)

# Result:
# order_id | customer.id | customer.name | customer.address.street | ... | items | totals.total
# ord_12345| cust_789    | Alice Smith   | 123 Main St             | ... | [...]  | 337.87
```

**Strategy 2: Exploding arrays**

```python
# Explode the items array
df = json_normalize(
    data,
    record_path='items',                    # Array to explode
    meta=['order_id',                       # Fields to keep from parent
          ['customer', 'id'],
          ['customer', 'name'],
          ['totals', 'total']],
    meta_prefix='order_'                    # Prefix for parent fields
)

# Result:
# product_id | name               | price  | quantity | order_order_id | order_customer.id | order_totals.total
# prod_001   | Wireless Headphones| 249.99 | 1        | ord_12345      | cust_789          | 337.87
# prod_002   | Phone Case         | 29.99  | 2        | ord_12345      | cust_789          | 337.87
```

**Strategy 3: PySpark approach**

```python
from pyspark.sql import functions as F

# Read nested JSON
df = spark.read.json("orders.json")

# Access nested fields with dot notation
df.select(
    "order_id",
    "customer.id",
    "customer.name",
    "customer.address.city"
)

# Explode arrays
df_items = df.select(
    "order_id",
    "customer.id",
    F.explode("items").alias("item")
).select(
    "order_id",
    "id",
    "item.product_id",
    "item.name",
    "item.price",
    "item.quantity"
)
```

### 2.5.3 When to Flatten vs. Keep Nested

| Keep Nested | Flatten |
|-------------|---------|
| Document databases (MongoDB) | Analytical queries |
| API responses | Machine learning features |
| Event archives | Reporting and BI |
| Schema flexibility needed | SQL-based analysis |

---

## 2.6 Summary and Key Concepts

This chapter provided comprehensive coverage of data types and file formats:

### Data Structure Spectrum
- **Structured**: Fixed schema, tables, SQL-queryable
- **Semi-structured**: Flexible, nested, self-describing (JSON, XML)
- **Unstructured**: No schema, requires ML/NLP to process

### File Format Selection
- **CSV**: Universal but limited—use for small data and interchange
- **JSON**: Flexible, nested support—use for APIs and configuration
- **Parquet**: Columnar, compressed—use for analytics and data lakes
- **Avro**: Row-based, schema evolution—use for streaming and Kafka

### Key Insight
"JSON has the largest footprint because it stores the schema attributes for each row. For this reason, I rarely store JSON or CSV formats in curated and transformed zone in a data lake" (Towards Data Science, 2023).

---

## 2.7 Further Reading and Resources

### Papers and Articles
- **Mbata, A. et al. (2024). "A Survey of Pipeline Tools for Data Engineering."** arXiv:2406.08335. Comprehensive survey of data engineering tools and formats. https://arxiv.org/abs/2406.08335
- **Wickham, H. (2014). "Tidy Data."** Journal of Statistical Software. Foundational paper on data organization principles. https://vita.had.co.nz/papers/tidy-data.pdf

### Technical Documentation
- **Apache Parquet Documentation**: https://parquet.apache.org/docs/
- **Apache Avro Specification**: https://avro.apache.org/docs/current/spec.html
- **JSON Specification (RFC 8259)**: https://datatracker.ietf.org/doc/html/rfc8259

### Practical Comparisons
- **DataCrump: CSV vs Parquet vs JSON vs Avro**: Hands-on performance comparison with benchmarks. https://datacrump.com/csv-parquet-json-avro/

### Books
- **Reis, J. & Housley, M. (2022). *Fundamentals of Data Engineering*.** O'Reilly Media. Chapter 5 covers data formats in depth.
- **Kleppmann, M. (2017). *Designing Data-Intensive Applications*.** O'Reilly Media. Chapter 4 discusses encoding and evolution.

---

## 2.8 Exercises

**Exercise 2.1: Format Conversion and Comparison**

Take a CSV file of at least 100,000 rows and:
1. Convert it to JSON, Parquet, and Avro formats
2. Compare file sizes for each format
3. Measure read times for loading into pandas
4. Apply compression (gzip for CSV/JSON, snappy for Parquet) and compare again

```python
import pandas as pd
import time
import os

# Your code here
df = pd.read_csv('your_data.csv')

# Save in different formats and measure
formats = {
    'csv': lambda: df.to_csv('output.csv', index=False),
    'json': lambda: df.to_json('output.json', orient='records'),
    'parquet': lambda: df.to_parquet('output.parquet'),
}

for name, save_func in formats.items():
    start = time.time()
    save_func()
    elapsed = time.time() - start
    size = os.path.getsize(f'output.{name}') / 1e6
    print(f"{name}: {size:.2f} MB, {elapsed:.2f} seconds")
```

**Exercise 2.2: Nested Data Transformation**

Given this nested JSON structure, create a flat DataFrame suitable for analysis:

```json
{
  "users": [
    {
      "id": 1,
      "name": "Alice",
      "profile": {
        "age": 28,
        "occupation": "Engineer"
      },
      "sessions": [
        {"date": "2025-01-01", "duration_minutes": 45, "pages_viewed": 12},
        {"date": "2025-01-02", "duration_minutes": 30, "pages_viewed": 8}
      ]
    },
    {
      "id": 2,
      "name": "Bob",
      "profile": {
        "age": 35,
        "occupation": "Designer"
      },
      "sessions": [
        {"date": "2025-01-01", "duration_minutes": 60, "pages_viewed": 20}
      ]
    }
  ]
}
```

Create two DataFrames:
1. User-level DataFrame with profile information
2. Session-level DataFrame with user context

**Exercise 2.3: Schema Design**

Design a data schema for a ride-sharing application (like Uber/Lyft). Consider:
- What events should be tracked?
- What fields should each event have?
- What data types are appropriate for each field?
- How should nested data (e.g., route waypoints) be structured?

Document your schema in JSON Schema format.

**Exercise 2.4: Type Optimization**

Given a DataFrame with default types, optimize memory usage by selecting appropriate types:

```python
import pandas as pd
import numpy as np

# Create sample data with suboptimal types
df = pd.DataFrame({
    'user_id': np.random.randint(1, 1000000, 1000000),        # Could be int32
    'age': np.random.randint(18, 80, 1000000),                # Could be int8
    'country': np.random.choice(['US', 'UK', 'DE', 'FR'], 1000000),  # Could be category
    'is_premium': np.random.choice([True, False], 1000000),   # Already boolean
    'balance': np.random.uniform(0, 10000, 1000000),          # Could be float32
})

print(f"Original memory: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

# Your optimization code here
# Target: Reduce memory by at least 50%
```

---

# End of Part I: Foundations

The next part of this course will cover **Data Collection and Storage**, including:
- Chapter 3: Data Sources and Collection Methods
- Chapter 4: Relational Databases and SQL
- Chapter 5: Data Warehouses, Data Lakes, and Modern Storage Architectures

---


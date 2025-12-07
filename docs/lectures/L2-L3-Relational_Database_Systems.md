# Complete SQL Reference Guide
## From Fundamentals to Advanced Analytics

---

# Part 1: Relational Database Fundamentals

## 1.1 What is a Relational Database?

A **relational database** is a type of database that organizes data into **tables** (also called **relations**) consisting of rows and columns. The "relational" part refers to how tables can be linked to each other through shared columns, enabling complex data relationships while minimizing redundancy.

### Core Terminology

| Term | Definition | Analogy |
|------|------------|---------|
| **Table (Relation)** | A collection of related data organized in rows and columns | A spreadsheet tab |
| **Row (Record/Tuple)** | A single entry in a table representing one entity | One line in a spreadsheet |
| **Column (Field/Attribute)** | A specific piece of data that each row contains | A spreadsheet column header |
| **Schema** | The structure/blueprint of a database (tables, columns, relationships) | The architectural plan |
| **Database** | A collection of related tables and other objects | The entire filing cabinet |

### Why Relational Databases?

Relational databases solve fundamental data management problems:

1. **Data Integrity**: Rules ensure data remains accurate and consistent
2. **Reduced Redundancy**: Information stored once, referenced many times
3. **Flexible Querying**: SQL allows complex questions to be answered easily
4. **ACID Compliance**: Transactions are reliable and predictable
5. **Scalability**: Can handle growing data volumes efficiently

---

## 1.2 Keys: The Foundation of Relationships

### Primary Key (PK)

A **Primary Key** is a column (or combination of columns) that **uniquely identifies each row** in a table.

**Rules for Primary Keys:**
- Must be **unique** — no two rows can have the same PK value
- Cannot be **NULL** — every row must have a value
- Should be **immutable** — ideally never changes once set
- Every table should have exactly one

**Types of Primary Keys:**

```
┌─────────────────────────────────────────────────────────────────┐
│                     PRIMARY KEY TYPES                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. NATURAL KEY                   2. SURROGATE KEY              │
│     (Real-world identifier)          (Artificial identifier)    │
│                                                                 │
│     Examples:                        Examples:                  │
│     • Social Security Number         • Auto-increment ID        │
│     • ISBN for books                 • UUID/GUID                │
│     • Email address                  • Sequence number          │
│                                                                 │
│     Pros: Meaningful                 Pros: Simple, stable       │
│     Cons: Can change, complex        Cons: No business meaning  │
│                                                                 │
│  3. COMPOSITE KEY (Multiple columns together)                   │
│     Example: (student_id, course_id) for enrollments            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**SQL Examples:**

```sql
-- Single column primary key (most common)
CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100)
);

-- Auto-increment primary key
CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,  -- PostgreSQL
    -- customer_id INT AUTO_INCREMENT PRIMARY KEY,  -- MySQL
    name VARCHAR(100),
    email VARCHAR(100)
);

-- Composite primary key
CREATE TABLE order_items (
    order_id INT,
    product_id INT,
    quantity INT,
    PRIMARY KEY (order_id, product_id)
);
```

### Foreign Key (FK)

A **Foreign Key** is a column that **references a Primary Key in another table**, creating a relationship between the two tables.

**What Foreign Keys Do:**
- Create relationships between tables
- Enforce **referential integrity** (can't reference non-existent records)
- Enable JOIN operations
- Prevent orphan records

```
┌─────────────────────┐              ┌─────────────────────┐
│      customers      │              │       orders        │
├─────────────────────┤              ├─────────────────────┤
│ customer_id (PK) ●──┼──────────────┼──○ customer_id (FK) │
│ name                │              │ order_id (PK)       │
│ email               │              │ order_date          │
│ city                │              │ total_amount        │
└─────────────────────┘              └─────────────────────┘

● = Primary Key (unique identifier)
○ = Foreign Key (reference to another table)
```

**SQL Example:**

```sql
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    order_date DATE,
    total_amount DECIMAL(10,2),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
```

**Referential Integrity Actions:**

```sql
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        ON DELETE CASCADE      -- Delete orders when customer deleted
        ON UPDATE CASCADE      -- Update FK when customer PK changes
);

-- Other options:
-- ON DELETE SET NULL     -- Set FK to NULL when parent deleted
-- ON DELETE RESTRICT     -- Prevent deletion if children exist
-- ON DELETE NO ACTION    -- Same as RESTRICT (default)
```

---

## 1.3 Types of Relationships

### One-to-Many (1:N) — Most Common

One record in Table A can relate to many records in Table B, but each record in Table B relates to only one record in Table A.

```
┌─────────────────────┐         ┌─────────────────────┐
│      customers      │         │       orders        │
├─────────────────────┤         ├─────────────────────┤
│ customer_id (PK)    │◄───┐    │ order_id (PK)       │
│ name                │    │    │ customer_id (FK) ───┘
│ email               │    │    │ order_date          │
└─────────────────────┘    │    │ total_amount        │
                           │    └─────────────────────┘
                           │
    One customer ──────────┴──────── Many orders
```

**Real-world examples:**
- One customer → Many orders
- One department → Many employees
- One author → Many blog posts
- One category → Many products

**Implementation:** Place the FK in the "many" side table.

### One-to-One (1:1) — Rare

Each record in Table A relates to exactly one record in Table B, and vice versa.

```
┌─────────────────────┐         ┌─────────────────────┐
│       users         │         │    user_profiles    │
├─────────────────────┤         ├─────────────────────┤
│ user_id (PK)        │◄────────│ user_id (PK, FK)    │
│ email               │         │ bio                 │
│ password_hash       │         │ avatar_url          │
└─────────────────────┘         │ preferences_json    │
                                └─────────────────────┘
```

**When to use:**
- Optional data that doesn't apply to all records
- Security separation (sensitive data in separate table)
- Very large columns that are rarely accessed
- Exceeding column limits in a table

**Implementation:** FK in either table (usually the optional one), often also the PK.

### Many-to-Many (M:N)

Many records in Table A can relate to many records in Table B.

**Problem:** Cannot be directly represented with just two tables.

**Solution:** Create a **junction table** (also called bridge/associative table).

```
┌─────────────────────┐                           ┌─────────────────────┐
│      students       │                           │       courses       │
├─────────────────────┤                           ├─────────────────────┤
│ student_id (PK)     │◄───┐                 ┌───►│ course_id (PK)      │
│ name                │    │                 │    │ course_name         │
│ email               │    │                 │    │ credits             │
└─────────────────────┘    │                 │    └─────────────────────┘
                           │                 │
                           │  ┌───────────────────────┐
                           │  │     enrollments       │
                           │  │   (junction table)    │
                           │  ├───────────────────────┤
                           └──│ student_id (FK, PK)   │
                              │ course_id (FK, PK)  ──┘
                              │ enrollment_date       │
                              │ grade                 │
                              └───────────────────────┘
```

```sql
CREATE TABLE enrollments (
    student_id INT,
    course_id INT,
    enrollment_date DATE,
    grade CHAR(2),
    PRIMARY KEY (student_id, course_id),  -- Composite PK
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    FOREIGN KEY (course_id) REFERENCES courses(course_id)
);
```

**Real-world examples:**
- Students ↔ Courses (via enrollments)
- Products ↔ Orders (via order_items)
- Users ↔ Roles (via user_roles)
- Authors ↔ Books (via book_authors)

---

## 1.4 SQL Data Types

### Numeric Types

| Type | Description | Range/Precision | Use Case |
|------|-------------|-----------------|----------|
| `TINYINT` | Very small integer | -128 to 127 | Status codes, flags |
| `SMALLINT` | Small integer | -32,768 to 32,767 | Counts, years |
| `INT`/`INTEGER` | Standard integer | ~±2.1 billion | IDs, counts |
| `BIGINT` | Large integer | ~±9.2 quintillion | Large IDs, timestamps |
| `DECIMAL(p,s)` | Fixed precision | p digits, s after decimal | Money, exact values |
| `NUMERIC(p,s)` | Same as DECIMAL | p digits, s after decimal | Money, exact values |
| `FLOAT` | Single precision | ~7 significant digits | Scientific data |
| `DOUBLE`/`REAL` | Double precision | ~15 significant digits | Scientific data |

```sql
-- DECIMAL is crucial for money (no floating point errors)
CREATE TABLE products (
    price DECIMAL(10, 2),     -- 10 digits total, 2 after decimal
    tax_rate DECIMAL(5, 4)    -- e.g., 0.0825 for 8.25%
);

-- NEVER use FLOAT for money!
-- 0.1 + 0.2 = 0.30000000000000004 in floating point
```

### String Types

| Type | Description | Max Length | Use Case |
|------|-------------|------------|----------|
| `CHAR(n)` | Fixed-length string | n characters (padded) | Country codes, status |
| `VARCHAR(n)` | Variable-length string | Up to n characters | Names, emails |
| `TEXT` | Long text | ~65KB to unlimited | Descriptions, articles |
| `MEDIUMTEXT` | Medium long text | ~16MB | Blog posts |
| `LONGTEXT` | Very long text | ~4GB | Documents |

```sql
-- CHAR vs VARCHAR
CREATE TABLE countries (
    country_code CHAR(2),       -- Always 2 chars: 'US', 'CA'
    country_name VARCHAR(100)   -- Variable: 'USA', 'United Kingdom'
);
```

### Date and Time Types

| Type | Format | Example | Use Case |
|------|--------|---------|----------|
| `DATE` | YYYY-MM-DD | 2025-11-29 | Birthdates, due dates |
| `TIME` | HH:MM:SS | 14:30:00 | Schedules, durations |
| `DATETIME` | YYYY-MM-DD HH:MM:SS | 2025-11-29 14:30:00 | Events, logs |
| `TIMESTAMP` | Unix timestamp or datetime | 2025-11-29 14:30:00 UTC | Auto-tracking, sync |
| `YEAR` | YYYY | 2025 | Years only |

```sql
CREATE TABLE events (
    event_date DATE,
    start_time TIME,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

### Other Types

| Type | Description | Use Case |
|------|-------------|----------|
| `BOOLEAN` | TRUE/FALSE | Flags, toggles |
| `ENUM` | Predefined values | Status, categories |
| `JSON` | JSON data | Flexible schemas |
| `BLOB` | Binary data | Images, files |
| `UUID` | Universally unique ID | Distributed systems |

```sql
CREATE TABLE users (
    is_active BOOLEAN DEFAULT TRUE,
    status ENUM('pending', 'active', 'suspended'),
    preferences JSON,
    avatar BLOB
);
```

---

## 1.5 ACID Properties

ACID properties ensure database transactions are processed reliably.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            ACID PROPERTIES                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ╔═══════════════╗    A transaction is an "all or nothing" operation.       │
│  ║  ATOMICITY    ║    Either ALL operations succeed, or NONE do.            │
│  ╚═══════════════╝    Example: Bank transfer - debit AND credit both        │
│                       happen, or neither happens.                           │
│                                                                             │
│  ╔═══════════════╗    Database moves from one valid state to another.       │
│  ║  CONSISTENCY  ║    All rules, constraints, and triggers are enforced.    │
│  ╚═══════════════╝    Example: Account balance can never go negative        │
│                       if that's a defined constraint.                       │
│                                                                             │
│  ╔═══════════════╗    Concurrent transactions don't interfere.              │
│  ║  ISOLATION    ║    Each transaction sees a consistent snapshot.          │
│  ╚═══════════════╝    Example: Two people buying last item - only one       │
│                       succeeds, the other sees "out of stock."              │
│                                                                             │
│  ╔═══════════════╗    Committed transactions are permanent.                 │
│  ║  DURABILITY   ║    Data survives system crashes, power failures.         │
│  ╚═══════════════╝    Example: Confirmed order stays confirmed even         │
│                       if server crashes immediately after.                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Transaction Example:**

```sql
-- Bank transfer: Move $100 from Account A to Account B
BEGIN TRANSACTION;

UPDATE accounts SET balance = balance - 100 WHERE account_id = 'A';
UPDATE accounts SET balance = balance + 100 WHERE account_id = 'B';

-- If both succeed:
COMMIT;

-- If anything fails:
ROLLBACK;  -- Undoes everything, account balances unchanged
```

---

# Part 2: Database Normalization

## 2.1 Why Normalize?

### The Problem: Non-Normalized Data

Consider storing all order information in a single table:

**orders_denormalized**
| order_id | customer_name | customer_email | city | product | price | qty |
|----------|---------------|----------------|------|---------|-------|-----|
| 101 | Alice | alice@email.com | NYC | Laptop | 999.00 | 1 |
| 102 | Alice | alice@email.com | NYC | Mouse | 29.00 | 2 |
| 103 | Alice | alice@email.com | NYC | Keyboard | 79.00 | 1 |
| 104 | Bob | bob@email.com | LA | Laptop | 999.00 | 1 |

### Problems Identified

**1. Data Redundancy**
- Alice's name, email, and city are stored 3 times
- "Laptop" price is stored twice
- Wastes storage space
- More data to maintain

**2. Update Anomaly**
If Alice changes her email, you must update EVERY row:

```sql
-- Must update 3 rows!
UPDATE orders_denormalized
SET customer_email = 'alice.new@email.com'
WHERE customer_name = 'Alice';

-- RISK: If you miss one row, data becomes inconsistent
```

**3. Insert Anomaly**
Can't add a new customer until they place an order:

```sql
-- FAILS! What values for order_id, product, price, qty?
INSERT INTO orders_denormalized (customer_name, customer_email, city)
VALUES ('Charlie', 'charlie@email.com', 'Chicago');
```

**4. Delete Anomaly**
Deleting Bob's only order loses Bob's information entirely:

```sql
-- Bob is gone from the database!
DELETE FROM orders_denormalized WHERE customer_name = 'Bob';
```

### The Solution: Normalization

**Normalization** is the process of organizing data into multiple related tables, each storing one type of entity, to eliminate redundancy and prevent anomalies.

```
                    ┌─────────────────┐
                    │    customers    │
                    ├─────────────────┤
                    │ customer_id (PK)│
                    │ name            │
                    │ email           │
                    │ city            │
                    └────────┬────────┘
                             │
                             │ 1:N
                             │
                    ┌────────▼────────┐
                    │     orders      │
                    ├─────────────────┤
                    │ order_id (PK)   │
                    │ customer_id (FK)│──────────┐
                    │ order_date      │          │
                    └────────┬────────┘          │
                             │                   │
                             │ 1:N               │
                             │                   │
                    ┌────────▼────────┐          │
                    │   order_items   │          │
                    ├─────────────────┤          │
                    │ order_id (FK,PK)│          │
                    │ product_id(FK,PK)──────────┼───┐
                    │ quantity        │          │   │
                    └─────────────────┘          │   │
                                                 │   │
                    ┌─────────────────┐          │   │
                    │    products     │◄─────────┘   │
                    ├─────────────────┤              │
                    │ product_id (PK) │◄─────────────┘
                    │ name            │
                    │ price           │
                    └─────────────────┘
```

---

## 2.2 Normal Forms

### First Normal Form (1NF)

**Requirements:**
1. Each column contains only **atomic** (indivisible) values
2. No repeating groups or arrays within a cell
3. Each row is unique (has a primary key)
4. Each column has a unique name

**Violation Example:**

| order_id | customer | products |
|----------|----------|----------|
| 101 | Alice | Laptop, Mouse, Keyboard |

**Problem:** Multiple values in the "products" cell. Cannot easily query individual products.

**1NF Solution:**

| order_id | customer | product |
|----------|----------|---------|
| 101 | Alice | Laptop |
| 101 | Alice | Mouse |
| 101 | Alice | Keyboard |

```sql
-- Non-1NF (bad - array in column)
CREATE TABLE orders_bad (
    order_id INT,
    customer VARCHAR(100),
    products TEXT  -- "Laptop, Mouse, Keyboard"
);

-- 1NF compliant
CREATE TABLE order_items (
    order_id INT,
    customer VARCHAR(100),
    product VARCHAR(100),
    PRIMARY KEY (order_id, product)
);
```

### Second Normal Form (2NF)

**Requirements:**
1. Must be in 1NF
2. No **partial dependencies** — every non-key column depends on the ENTIRE primary key, not just part of it

**Only applies to tables with composite primary keys.**

**Violation Example:**

Composite PK: (order_id, product_id)

| order_id | product_id | customer_name | product_name | quantity |
| -------- | ---------- | ------------- | ------------ | -------- |
| 101      | 1          | Alice         | Laptop       | 1        |

**Problems:**
- `customer_name` depends only on `order_id` (partial dependency)
- `product_name` depends only on `product_id` (partial dependency)

**2NF Solution:** Split into separate tables:

```sql
-- orders table (customer depends on full order)
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT
);

-- products table (product_name depends on full product)
CREATE TABLE products (
    product_id INT PRIMARY KEY,
    product_name VARCHAR(100)
);

-- order_items (quantity depends on both order AND product)
CREATE TABLE order_items (
    order_id INT,
    product_id INT,
    quantity INT,
    PRIMARY KEY (order_id, product_id)
);
```

### Third Normal Form (3NF)

**Requirements:**
1. Must be in 2NF
2. No **transitive dependencies** — non-key columns cannot depend on other non-key columns

**Violation Example:**

| customer_id | name  | city | city_zip |
| ----------- | ----- | ---- | -------- |
| 1           | Alice | NYC  | 10001    |
| 2           | Bob   | NYC  | 10001    |

**Problem:** `city_zip` depends on `city`, not directly on `customer_id`. This is a transitive dependency:
```
customer_id → city → city_zip
```

**3NF Solution:**

```sql
-- customers table
CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    name VARCHAR(100),
    city_id INT
);

-- cities table
CREATE TABLE cities (
    city_id INT PRIMARY KEY,
    city_name VARCHAR(100),
    zip_code VARCHAR(10)
);
```

### Summary of Normal Forms

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         NORMAL FORMS PROGRESSION                           │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   ┌─────────┐                                                              │
│   │   1NF   │  ► Atomic values only                                        │
│   │         │  ► No repeating groups                                       │
│   │         │  ► Has primary key                                           │
│   └────┬────┘                                                              │
│        │                                                                   │
│        ▼                                                                   │
│   ┌─────────┐                                                              │
│   │   2NF   │  ► Must be in 1NF                                            │
│   │         │  ► No partial dependencies                                   │
│   │         │  ► (All columns depend on ENTIRE PK)                         │
│   └────┬────┘                                                              │
│        │                                                                   │
│        ▼                                                                   │
│   ┌─────────┐                                                              │
│   │   3NF   │  ► Must be in 2NF                                            │
│   │         │  ► No transitive dependencies                                │
│   │         │  ► (Non-key columns don't depend on each other)              │
│   └────┬────┘                                                              │
│        │                                                                   │
│        ▼                                                                   │
│   ┌─────────┐                                                              │
│   │  BCNF   │  ► Must be in 3NF                                            │
│   │         │  ► Every determinant is a candidate key                      │
│   │         │  ► (Stricter version of 3NF)                                 │
│   └─────────┘                                                              │
│                                                                            │
│   Most production databases aim for 3NF — good balance of                  │
│   data integrity and query performance.                                    │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### Higher Normal Forms (Brief Overview)

**Boyce-Codd Normal Form (BCNF):**
- Stricter than 3NF
- Every determinant must be a candidate key
- Handles rare edge cases in 3NF

**Fourth Normal Form (4NF):**
- No multi-valued dependencies
- Example: Employee skills and languages stored separately

**Fifth Normal Form (5NF):**
- No join dependencies
- Extremely rare in practice

---

## 2.3 When to Denormalize

While normalization is generally good, **denormalization** (intentionally violating normal forms) is sometimes used for performance.

**Scenarios for Denormalization:**

| Scenario                  | Why Denormalize?                                      |
| ------------------------- | ----------------------------------------------------- |
| Read-heavy workloads      | JOINs are expensive; pre-joined data speeds up reads  |
| Data warehouses           | Analytics queries benefit from star/snowflake schemas |
| Reporting tables          | Pre-aggregate data for dashboards                     |
| Caching layers            | Store computed values for performance                 |
| High-traffic applications | Reduce query complexity at read time                  |

**Example: Storing order total instead of calculating**

```sql
-- Normalized (calculate total each time)
SELECT SUM(oi.quantity * p.price) as total
FROM order_items oi
JOIN products p ON oi.product_id = p.product_id
WHERE oi.order_id = 101;

-- Denormalized (stored total)
SELECT total_amount FROM orders WHERE order_id = 101;
```

**Trade-off:** Faster reads, but must update `total_amount` whenever items change.

---

# Part 3: SQL Queries - From Basic to Advanced

## 3.1 SELECT - Reading Data

### Basic SELECT

```sql
-- Select all columns from a table
SELECT * FROM customers;

-- Select specific columns
SELECT customer_id, name, email FROM customers;

-- Select with column aliases
SELECT 
    customer_id AS id,
    name AS customer_name,
    email AS contact_email
FROM customers;

-- Select with expressions
SELECT 
    product_name,
    price,
    price * 0.9 AS discounted_price,
    price * 1.08 AS price_with_tax
FROM products;
```

### DISTINCT - Removing Duplicates

```sql
-- Get unique cities
SELECT DISTINCT city FROM customers;

-- Get unique combinations
SELECT DISTINCT city, state FROM customers;

-- Count unique values
SELECT COUNT(DISTINCT city) AS unique_cities FROM customers;
```

---

## 3.2 WHERE - Filtering Data

### Comparison Operators

```sql
-- Equals
SELECT * FROM products WHERE price = 99.99;

-- Not equals
SELECT * FROM products WHERE category <> 'Electronics';
SELECT * FROM products WHERE category != 'Electronics';

-- Greater than / Less than
SELECT * FROM orders WHERE total_amount > 100;
SELECT * FROM orders WHERE total_amount >= 100;
SELECT * FROM orders WHERE order_date < '2025-01-01';
SELECT * FROM orders WHERE order_date <= '2025-01-01';
```

### Logical Operators

```sql
-- AND (all conditions must be true)
SELECT * FROM orders 
WHERE total_amount > 100 
  AND order_date >= '2025-01-01';

-- OR (at least one condition must be true)
SELECT * FROM customers 
WHERE city = 'NYC' 
   OR city = 'LA';

-- NOT (negates condition)
SELECT * FROM products 
WHERE NOT category = 'Electronics';

-- Combined with parentheses
SELECT * FROM orders 
WHERE (status = 'pending' OR status = 'processing')
  AND total_amount > 50;
```

### Special Operators

```sql
-- IN (match any value in list)
SELECT * FROM customers 
WHERE city IN ('NYC', 'LA', 'Chicago', 'Miami');

-- NOT IN
SELECT * FROM customers 
WHERE city NOT IN ('NYC', 'LA');

-- BETWEEN (inclusive range)
SELECT * FROM orders 
WHERE order_date BETWEEN '2025-01-01' AND '2025-12-31';

SELECT * FROM products 
WHERE price BETWEEN 10 AND 100;

-- LIKE (pattern matching)
SELECT * FROM customers WHERE email LIKE '%@gmail.com';   -- Ends with
SELECT * FROM customers WHERE name LIKE 'John%';          -- Starts with
SELECT * FROM customers WHERE name LIKE '%son%';          -- Contains
SELECT * FROM products WHERE sku LIKE 'PRD-___';          -- Exactly 3 chars after PRD-

-- IS NULL / IS NOT NULL
SELECT * FROM customers WHERE phone IS NULL;
SELECT * FROM orders WHERE shipped_date IS NOT NULL;
```

---

## 3.3 ORDER BY & LIMIT

### ORDER BY - Sorting Results

```sql
-- Sort ascending (default)
SELECT * FROM products ORDER BY price;
SELECT * FROM products ORDER BY price ASC;

-- Sort descending
SELECT * FROM products ORDER BY price DESC;

-- Multiple columns (sort by first, then by second for ties)
SELECT * FROM orders 
ORDER BY customer_id ASC, order_date DESC;

-- Sort by expression
SELECT product_name, price, price * quantity AS total
FROM order_items
ORDER BY price * quantity DESC;

-- Sort by column position (not recommended for readability)
SELECT customer_id, name, city FROM customers
ORDER BY 3;  -- Sorts by city

-- Sort with NULLs first/last
SELECT * FROM employees ORDER BY manager_id NULLS FIRST;
SELECT * FROM employees ORDER BY manager_id NULLS LAST;
```

### LIMIT - Restricting Results

```sql
-- Get first 10 rows
SELECT * FROM customers LIMIT 10;

-- Top 5 highest spending customers
SELECT customer_id, SUM(total_amount) AS total_spent
FROM orders
GROUP BY customer_id
ORDER BY total_spent DESC
LIMIT 5;

-- Pagination (OFFSET)
-- Page 1: rows 1-10
SELECT * FROM products ORDER BY product_id LIMIT 10 OFFSET 0;

-- Page 2: rows 11-20
SELECT * FROM products ORDER BY product_id LIMIT 10 OFFSET 10;

-- Page 3: rows 21-30
SELECT * FROM products ORDER BY product_id LIMIT 10 OFFSET 20;

-- Alternative syntax (MySQL)
SELECT * FROM products ORDER BY product_id LIMIT 10, 10;  -- LIMIT offset, count
```

---

## 3.4 Aggregate Functions

### Basic Aggregates

```sql
-- COUNT - number of rows
SELECT COUNT(*) AS total_rows FROM orders;
SELECT COUNT(phone) AS customers_with_phone FROM customers;  -- Excludes NULLs
SELECT COUNT(DISTINCT customer_id) AS unique_customers FROM orders;

-- SUM - total of numeric values
SELECT SUM(total_amount) AS total_revenue FROM orders;

-- AVG - average value
SELECT AVG(total_amount) AS avg_order_value FROM orders;

-- MIN / MAX
SELECT MIN(price) AS cheapest, MAX(price) AS most_expensive FROM products;
SELECT MIN(order_date) AS first_order, MAX(order_date) AS last_order FROM orders;

-- Multiple aggregates together
SELECT 
    COUNT(*) AS num_orders,
    SUM(total_amount) AS total_revenue,
    AVG(total_amount) AS avg_order,
    MIN(total_amount) AS smallest_order,
    MAX(total_amount) AS largest_order
FROM orders;
```


---

## 3.5 GROUP BY - Grouping Data

### Basic Grouping

```sql
-- Orders per customer
SELECT 
    customer_id,
    COUNT(*) AS num_orders,
    SUM(total_amount) AS total_spent,
    AVG(total_amount) AS avg_order_value
FROM orders
GROUP BY customer_id;

-- Products per category
SELECT 
    category,
    COUNT(*) AS product_count,
    AVG(price) AS avg_price,
    MIN(price) AS min_price,
    MAX(price) AS max_price
FROM products
GROUP BY category;

-- Multiple grouping columns
SELECT 
    strftime('%Y', order_date) AS year,
    strftime('%m', order_date) AS month,
    COUNT(*) AS num_orders,
    SUM(total_amount) AS monthly_revenue
FROM orders
GROUP BY strftime('%Y', order_date), strftime('%m', order_date)
ORDER BY year, month;
```


## 3.6 HAVING - Filtering Groups

**WHERE filters rows BEFORE grouping. HAVING filters groups AFTER aggregation.**

```sql
-- Customers with more than 5 orders
SELECT 
    customer_id,
    COUNT(*) AS num_orders
FROM orders
GROUP BY customer_id
HAVING COUNT(*) > 5;

-- Categories with average price over $50
SELECT 
    category,
    AVG(price) AS avg_price
FROM products
GROUP BY category
HAVING AVG(price) > 50;

-- Combined WHERE and HAVING
SELECT 
    customer_id,
    COUNT(*) AS num_orders,
    SUM(total_amount) AS total_spent
FROM orders
WHERE order_date >= '2025-01-01'    -- Filter rows first
GROUP BY customer_id
HAVING SUM(total_amount) > 1000;    -- Then filter groups
```

### WHERE vs HAVING Comparison

```
┌─────────────────────────────────────────────────────────────────────┐
│                      WHERE vs HAVING                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   WHERE                          HAVING                             │
│   ─────────────────────────      ─────────────────────────          │
│   • Filters individual ROWS      • Filters GROUPS                   │
│   • Applied BEFORE grouping      • Applied AFTER grouping           │
│   • Cannot use aggregates        • Can use aggregate functions      │
│   • Faster (reduces data early)  • Works on aggregated results      │
│                                                                     │
│   Example:                       Example:                           │
│   WHERE order_date > '2025'      HAVING COUNT(*) > 5                │
│   WHERE price < 100              HAVING SUM(amount) > 1000          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

# Part 4: JOINs - Combining Tables

## 4.1 Understanding JOINs

JOINs combine rows from two or more tables based on a related column. This is essential for working with normalized databases.

### Sample Tables for Examples

**customers**
| customer_id | name | city |
|-------------|------|------|
| 1 | Alice | NYC |
| 2 | Bob | LA |
| 3 | Carol | Chicago |
| 4 | David | Miami |

**orders**
| order_id | customer_id | amount |
|----------|-------------|--------|
| 101 | 1 | 150.00 |
| 102 | 1 | 200.00 |
| 103 | 2 | 75.00 |
| 104 | 5 | 300.00 |

Note: Customer 3, 4 have no orders. Order 104 has customer_id=5 (doesn't exist).

---

## 4.2 INNER JOIN

Returns only rows where there's a match in BOTH tables.

**Mathematical Definition:** INNER JOIN ≈ Set Intersection (for matching criteria)

```
    customers                orders               INNER JOIN Result
  ┌───────────┐          ┌───────────┐          ┌─────────────────┐
  │  1 Alice  │──────────│  1  150   │──────────│ Alice  1   150  │
  │  2 Bob    │──────────│  1  200   │──────────│ Alice  1   200  │
  │  3 Carol  │    ╳     │  2   75   │──────────│ Bob    2    75  │
  │  4 David  │    ╳     │  5  300   │    ╳     │                 │
  └───────────┘          └───────────┘          └─────────────────┘
       ╳ = No match, excluded from result
```

```sql
-- Basic INNER JOIN
SELECT 
    c.customer_id,
    c.name,
    o.order_id,
    o.amount
FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id;
```

**Result:**
| customer_id | name | order_id | amount |
|-------------|------|----------|--------|
| 1 | Alice | 101 | 150.00 |
| 1 | Alice | 102 | 200.00 |
| 2 | Bob | 103 | 75.00 |

Carol and David are excluded (no orders). Order 104 is excluded (customer doesn't exist).

---

## 4.3 LEFT JOIN (LEFT OUTER JOIN)

Returns ALL rows from the left table, plus matching rows from the right table. Non-matching right columns are NULL.

```
    customers                orders               LEFT JOIN Result
  ┌───────────┐          ┌───────────┐          ┌──────────────────┐
  │  1 Alice  │──────────│  1  150   │──────────│ Alice  1   150   │
  │  2 Bob    │──────────│  1  200   │──────────│ Alice  1   200   │
  │  3 Carol  │──NULL────│  2   75   │──────────│ Bob    2    75   │
  │  4 David  │──NULL────│  5  300   │    ╳     │ Carol  3   NULL  │
  └───────────┘          └───────────┘          │ David  4   NULL  │
                                                └──────────────────┘
```

```sql
-- LEFT JOIN
SELECT 
    c.customer_id,
    c.name,
    o.order_id,
    o.amount
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id;
```

**Result:**
| customer_id | name | order_id | amount |
|-------------|------|----------|--------|
| 1 | Alice | 101 | 150.00 |
| 1 | Alice | 102 | 200.00 |
| 2 | Bob | 103 | 75.00 |
| 3 | Carol | NULL | NULL |
| 4 | David | NULL | NULL |

**Use Cases:**
- Find all customers and their orders (including those with no orders)
- Count orders per customer (including zero)
- Find records that DON'T have matching records

```sql
-- Find customers who have never ordered
SELECT c.customer_id, c.name
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_id IS NULL;
```

---

## 4.4 RIGHT JOIN (RIGHT OUTER JOIN)

Returns ALL rows from the right table, plus matching rows from the left table. Non-matching left columns are NULL.

```
    customers                orders               RIGHT JOIN Result
  ┌───────────┐          ┌───────────┐          ┌──────────────────┐
  │  1 Alice  │──────────│  1  150   │──────────│ Alice  1   150   │
  │  2 Bob    │──────────│  1  200   │──────────│ Alice  1   200   │
  │  3 Carol  │    ╳     │  2   75   │──────────│ Bob    2    75   │
  │  4 David  │    ╳     │  5  300   │──NULL────│ NULL   5   300   │
  └───────────┘          └───────────┘          └──────────────────┘
```

```sql
-- RIGHT JOIN (less commonly used - can usually be rewritten as LEFT JOIN)
SELECT 
    c.customer_id,
    c.name,
    o.order_id,
    o.amount
FROM customers c
RIGHT JOIN orders o ON c.customer_id = o.customer_id;
```

**Result:**
| customer_id | name | order_id | amount |
|-------------|------|----------|--------|
| 1 | Alice | 101 | 150.00 |
| 1 | Alice | 102 | 200.00 |
| 2 | Bob | 103 | 75.00 |
| NULL | NULL | 104 | 300.00 |

---

## 4.5 FULL OUTER JOIN

Returns ALL rows from BOTH tables. NULL where there's no match.

```
    customers                orders               FULL OUTER Result
  ┌───────────┐          ┌───────────┐          ┌──────────────────┐
  │  1 Alice  │──────────│  1  150   │──────────│ Alice  1   150   │
  │  2 Bob    │──────────│  1  200   │──────────│ Alice  1   200   │
  │  3 Carol  │──NULL────│  2   75   │──────────│ Bob    2    75   │
  │  4 David  │──NULL────│  5  300   │──NULL────│ Carol  3   NULL  │
  └───────────┘          └───────────┘          │ David  4   NULL  │
                                                │ NULL   5   300   │
                                                └──────────────────┘
```

```sql
-- FULL OUTER JOIN
SELECT 
    c.customer_id,
    c.name,
    o.order_id,
    o.amount
FROM customers c
FULL OUTER JOIN orders o ON c.customer_id = o.customer_id;
```

**Result:**
| customer_id | name | order_id | amount |
|-------------|------|----------|--------|
| 1 | Alice | 101 | 150.00 |
| 1 | Alice | 102 | 200.00 |
| 2 | Bob | 103 | 75.00 |
| 3 | Carol | NULL | NULL |
| 4 | David | NULL | NULL |
| NULL | NULL | 104 | 300.00 |

**Note:** MySQL doesn't support FULL OUTER JOIN directly. Use UNION:

```sql
-- MySQL workaround for FULL OUTER JOIN
SELECT c.*, o.order_id, o.amount
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id

UNION

SELECT c.*, o.order_id, o.amount
FROM customers c
RIGHT JOIN orders o ON c.customer_id = o.customer_id
WHERE c.customer_id IS NULL;
```

---

## 4.6 CROSS JOIN (Cartesian Product)

Returns every combination of rows from both tables. No join condition.

**Mathematical Definition:** A × B = {(a,b) | a ∈ A and b ∈ B}

```sql
-- CROSS JOIN (explicit syntax)
SELECT c.name, p.product_name
FROM customers c
CROSS JOIN products p;

-- CROSS JOIN (implicit syntax)
SELECT c.name, p.product_name
FROM customers c, products p;
```

If customers has 4 rows and products has 10 rows, result has 4 × 10 = 40 rows.

**Use Cases:**
- Generate all possible combinations
- Create date scaffolds for reporting
- Expand data for analysis

```sql
-- Generate all date-product combinations for inventory tracking
SELECT d.date, p.product_id
FROM dates d
CROSS JOIN products p;
```

---

## 4.7 SELF JOIN

Joining a table to itself. Useful for hierarchical data.

**employees**
| emp_id | name | manager_id |
|--------|------|------------|
| 1 | Alice | NULL |
| 2 | Bob | 1 |
| 3 | Carol | 1 |
| 4 | David | 2 |

```sql
-- Find employees and their managers
SELECT 
    e.name AS employee,
    m.name AS manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.emp_id;
```

**Result:**
| employee | manager |
|----------|---------|
| Alice | NULL |
| Bob | Alice |
| Carol | Alice |
| David | Bob |

---

## 4.8 Multiple Table JOINs

Joining three or more tables:

```sql
-- Orders with customer names and product names
SELECT 
    c.name AS customer_name,
    o.order_id,
    o.order_date,
    p.product_name,
    oi.quantity,
    p.price,
    (oi.quantity * p.price) AS line_total
FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id
INNER JOIN order_items oi ON o.order_id = oi.order_id
INNER JOIN products p ON oi.product_id = p.product_id
ORDER BY o.order_date, o.order_id;
```

---

## 4.9 JOIN Visual Summary

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          JOIN TYPES SUMMARY                              │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   INNER JOIN                      LEFT JOIN                              │
│   ┌─────┬─────┐                   ┌─────┬─────┐                          │
│   │  A  │  B  │                   │  A  │  B  │                          │
│   │  ┌──┴──┐  │                   │  ████████ │                          │
│   │  │█████│  │                   │  ████████ │                          │
│   │  └──┬──┘  │                   │  └──┬──┘  │                          │
│   │     │     │                   │     │     │                          │
│   └─────┴─────┘                   └─────┴─────┘                          │
│   Only matching rows              All of A + matching B                  │
│                                                                          │
│   RIGHT JOIN                      FULL OUTER JOIN                        │
│   ┌─────┬─────┐                   ┌─────┬─────┐                          │
│   │  A  │  B  │                   │  A  │  B  │                          │
│   │  ┌──████████                  │  █████████│                          │
│   │  │  ████████                  │  █████████│                          │
│   │  └──┬──┘  │                   │  └──┬──┘  │                          │
│   │     │     │                   │     │     │                          │
│   └─────┴─────┘                   └─────┴─────┘                          │
│   All of B + matching A           All rows from both                     │
│                                                                          │
│   CROSS JOIN                      SELF JOIN                              │
│   ┌─────┬─────┐                   ┌─────────┐                            │
│   │  A  │  B  │                   │    A    │                            │
│   │  ×  │  ×  │                   │  ┌───┐  │                            │
│   │  ×  │  ×  │                   │  │ A │  │                            │
│   │  ×  │  ×  │                   │  └───┘  │                            │
│   └─────┴─────┘                   └─────────┘                            │
│   Every combination               Table joined to itself                 │
│   (A rows × B rows)               (hierarchical data)                    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

# Part 5: Subqueries and CTEs

## 5.1 Subqueries

A subquery is a query nested inside another query.

### Subquery in WHERE

```sql
-- Customers who ordered above average
SELECT customer_id, name
FROM customers
WHERE customer_id IN (
    SELECT customer_id 
    FROM orders 
    WHERE total_amount > (SELECT AVG(total_amount) FROM orders)
);

-- Products that have never been ordered
SELECT product_id, product_name
FROM products
WHERE product_id NOT IN (
    SELECT DISTINCT product_id FROM order_items
);

-- Using EXISTS (often faster than IN)
SELECT c.customer_id, c.name
FROM customers c
WHERE EXISTS (
    SELECT 1 FROM orders o 
    WHERE o.customer_id = c.customer_id 
    AND o.total_amount > 500
);
```

### Subquery in SELECT (Scalar Subquery)

```sql
-- Add a column with total order count
SELECT 
    customer_id,
    name,
    (SELECT COUNT(*) FROM orders o WHERE o.customer_id = c.customer_id) AS order_count
FROM customers c;

-- Compare each product price to category average
SELECT 
    product_name,
    price,
    (SELECT AVG(price) FROM products p2 WHERE p2.category = p1.category) AS category_avg,
    price - (SELECT AVG(price) FROM products p2 WHERE p2.category = p1.category) AS diff_from_avg
FROM products p1;
```

### Subquery in FROM (Derived Table)

```sql
-- Analyze customer spending tiers
SELECT 
    spending_tier,
    COUNT(*) AS customer_count,
    AVG(total_spent) AS avg_spent
FROM (
    SELECT 
        customer_id,
        SUM(total_amount) AS total_spent,
        CASE 
            WHEN SUM(total_amount) >= 1000 THEN 'High'
            WHEN SUM(total_amount) >= 500 THEN 'Medium'
            ELSE 'Low'
        END AS spending_tier
    FROM orders
    GROUP BY customer_id
) AS customer_spending
GROUP BY spending_tier;
```

---

## 5.2 Common Table Expressions (CTEs)

CTEs provide a cleaner, more readable way to write complex queries.

### Basic CTE Syntax

```sql
WITH cte_name AS (
    -- Your query here
    SELECT column1, column2 FROM table1
)
SELECT * FROM cte_name;
```

### Single CTE Example

```sql
-- High-value customers CTE
WITH high_value_customers AS (
    SELECT 
        customer_id,
        SUM(total_amount) AS total_spent
    FROM orders
    GROUP BY customer_id
    HAVING SUM(total_amount) > 1000
)
SELECT 
    c.name,
    hvc.total_spent
FROM customers c
INNER JOIN high_value_customers hvc ON c.customer_id = hvc.customer_id
ORDER BY hvc.total_spent DESC;
```

### Multiple CTEs

```sql
-- Multiple CTEs for complex analysis
WITH 
customer_orders AS (
    SELECT 
        customer_id,
        COUNT(*) AS order_count,
        SUM(total_amount) AS total_spent
    FROM orders
    GROUP BY customer_id
),
customer_categories AS (
    SELECT 
        customer_id,
        order_count,
        total_spent,
        CASE 
            WHEN total_spent >= 1000 THEN 'Gold'
            WHEN total_spent >= 500 THEN 'Silver'
            ELSE 'Bronze'
        END AS category
    FROM customer_orders
)
SELECT 
    c.name,
    cc.order_count,
    cc.total_spent,
    cc.category
FROM customers c
JOIN customer_categories cc ON c.customer_id = cc.customer_id
ORDER BY cc.total_spent DESC;
```

---

# Part 6: Window Functions (Analytics Functions)

## 6.1 Window Function Fundamentals

Window functions perform calculations across a set of rows related to the current row, without collapsing rows like GROUP BY does.

### Syntax Structure

```sql
function_name(expression) OVER (
    [PARTITION BY column1, column2, ...]    -- Divides rows into groups
    [ORDER BY column3, column4, ...]         -- Orders rows within partition
    [frame_clause]                           -- Defines window frame
)
```

### Key Difference: GROUP BY vs Window Functions

```sql
-- GROUP BY: Collapses rows
SELECT customer_id, SUM(amount) AS total
FROM orders
GROUP BY customer_id;
-- Result: One row per customer

-- Window Function: Keeps all rows
SELECT 
    customer_id,
    order_id,
    amount,
    SUM(amount) OVER (PARTITION BY customer_id) AS customer_total
FROM orders;
-- Result: All rows preserved, with running calculation added
```

---

## 6.2 Ranking Functions

### ROW_NUMBER()

Assigns a unique sequential number to each row within a partition.

```sql
-- Rank all orders by amount
SELECT 
    order_id,
    customer_id,
    amount,
    ROW_NUMBER() OVER (ORDER BY amount DESC) AS overall_rank
FROM orders;

-- Rank orders within each customer
SELECT 
    order_id,
    customer_id,
    amount,
    ROW_NUMBER() OVER (
        PARTITION BY customer_id 
        ORDER BY amount DESC
    ) AS rank_within_customer
FROM orders;
```

**Result Example:**
| order_id | customer_id | amount | rank_within_customer |
|----------|-------------|--------|----------------------|
| 102 | 1 | 200 | 1 |
| 101 | 1 | 150 | 2 |
| 105 | 1 | 75 | 3 |
| 103 | 2 | 300 | 1 |
| 104 | 2 | 100 | 2 |

### RANK()

Same as ROW_NUMBER, but gives same rank to ties, then skips.

```sql
SELECT 
    student_name,
    score,
    RANK() OVER (ORDER BY score DESC) AS rank
FROM exam_scores;
```

| student_name | score | rank |
|--------------|-------|------|
| Alice | 95 | 1 |
| Bob | 95 | 1 |
| Carol | 90 | 3 |  -- Skips 2!
| David | 85 | 4 |

### DENSE_RANK()

Same as RANK, but doesn't skip numbers after ties.

```sql
SELECT 
    student_name,
    score,
    DENSE_RANK() OVER (ORDER BY score DESC) AS dense_rank
FROM exam_scores;
```

| student_name | score | dense_rank |
|--------------|-------|------------|
| Alice | 95 | 1 |
| Bob | 95 | 1 |
| Carol | 90 | 2 |  -- No skip!
| David | 85 | 3 |

### NTILE(n)

Divides rows into n approximately equal groups.

```sql
-- Divide customers into 4 quartiles by total spending
SELECT 
    customer_id,
    total_spent,
    NTILE(4) OVER (ORDER BY total_spent DESC) AS spending_quartile
FROM customer_totals;
```

| customer_id | total_spent | spending_quartile |
|-------------|-------------|-------------------|
| 5 | 5000 | 1 |  -- Top 25%
| 3 | 4500 | 1 |
| 8 | 3000 | 2 |  -- Second quartile
| 1 | 2500 | 2 |
| ... | ... | ... |

### PERCENT_RANK() and CUME_DIST()

```sql
SELECT 
    product_name,
    price,
    -- Percentile rank (0 to 1)
    PERCENT_RANK() OVER (ORDER BY price) AS percent_rank,
    -- Cumulative distribution
    CUME_DIST() OVER (ORDER BY price) AS cumulative_dist
FROM products;
```

### Ranking Functions Comparison

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    RANKING FUNCTIONS COMPARISON                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Data: Scores = [100, 100, 90, 80]                                         │
│                                                                            │
│  ┌─────────┬─────────────┬────────┬────────────┬────────┐                  │
│  │  Score  │ ROW_NUMBER  │  RANK  │ DENSE_RANK │ NTILE(2)│                 │
│  ├─────────┼─────────────┼────────┼────────────┼─────────┤                 │
│  │   100   │      1      │   1    │     1      │    1    │                 │
│  │   100   │      2      │   1    │     1      │    1    │                 │
│  │    90   │      3      │   3    │     2      │    2    │                 │
│  │    80   │      4      │   4    │     3      │    2    │                 │
│  └─────────┴─────────────┴────────┴────────────┴─────────┘                 │
│                                                                            │
│  ROW_NUMBER: Always unique (1,2,3,4)                                       │
│  RANK: Same rank for ties, then skips (1,1,3,4)                            │
│  DENSE_RANK: Same rank for ties, no skip (1,1,2,3)                         │
│  NTILE(2): Divides into 2 groups (1,1,2,2)                                 │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 6.3 Aggregate Window Functions

Use aggregate functions (SUM, AVG, COUNT, etc.) as window functions.

### Running Total

```sql
-- Cumulative sum of daily revenue
SELECT 
    order_date,
    daily_revenue,
    SUM(daily_revenue) OVER (
        ORDER BY order_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_total
FROM daily_sales;
```

| order_date | daily_revenue | running_total |
| ---------- | ------------- | ------------- |
| 2025-01-01 | 100           | 100           |
| 2025-01-02 | 150           | 250           |
| 2025-01-03 | 200           | 450           |
| 2025-01-04 | 75            | 525           |

### Running Total by Partition

```sql
-- Running total per customer
SELECT 
    customer_id,
    order_date,
    amount,
    SUM(amount) OVER (
        PARTITION BY customer_id
        ORDER BY order_date
    ) AS customer_running_total
FROM orders;
```

### Moving Average

```sql
-- 7-day moving average
SELECT 
    order_date,
    daily_revenue,
    AVG(daily_revenue) OVER (
        ORDER BY order_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS moving_avg_7day
FROM daily_sales;

-- 3-period centered moving average
SELECT 
    order_date,
    daily_revenue,
    AVG(daily_revenue) OVER (
        ORDER BY order_date
        ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
    ) AS centered_avg_3day
FROM daily_sales;
```

### Count and Percentage

```sql
-- Count and percentage within partition
SELECT 
    category,
    product_name,
    price,
    COUNT(*) OVER (PARTITION BY category) AS category_count,
    price / SUM(price) OVER (PARTITION BY category) * 100 AS price_pct_of_category
FROM products;
```

---

## 6.4 Value Window Functions

Access values from other rows without self-joins.

### LAG() - Previous Row Value

```sql
-- Get previous day's revenue for comparison
SELECT 
    order_date,
    daily_revenue,
    LAG(daily_revenue, 1) OVER (ORDER BY order_date) AS prev_day_revenue,
    daily_revenue - LAG(daily_revenue, 1) OVER (ORDER BY order_date) AS day_over_day_change
FROM daily_sales;
```

| order_date | daily_revenue | prev_day_revenue | day_over_day_change |
|------------|---------------|------------------|---------------------|
| 2025-01-01 | 100 | NULL | NULL |
| 2025-01-02 | 150 | 100 | 50 |
| 2025-01-03 | 120 | 150 | -30 |

```sql
-- LAG with default value for NULLs
SELECT 
    order_date,
    daily_revenue,
    LAG(daily_revenue, 1, 0) OVER (ORDER BY order_date) AS prev_day_revenue
FROM daily_sales;
```

### LEAD() - Next Row Value

```sql
-- Get next day's revenue
SELECT 
    order_date,
    daily_revenue,
    LEAD(daily_revenue, 1) OVER (ORDER BY order_date) AS next_day_revenue
FROM daily_sales;

-- Look ahead 7 days
SELECT 
    order_date,
    daily_revenue,
    LEAD(daily_revenue, 7) OVER (ORDER BY order_date) AS revenue_in_7_days
FROM daily_sales;
```

### FIRST_VALUE() and LAST_VALUE()

```sql
-- Compare each day to first day of month
SELECT 
    order_date,
    daily_revenue,
    FIRST_VALUE(daily_revenue) OVER (
        PARTITION BY DATE_TRUNC('month', order_date)
        ORDER BY order_date
    ) AS first_day_of_month,
    daily_revenue - FIRST_VALUE(daily_revenue) OVER (
        PARTITION BY DATE_TRUNC('month', order_date)
        ORDER BY order_date
    ) AS diff_from_first_day
FROM daily_sales;

-- LAST_VALUE (requires frame specification!)
SELECT 
    order_date,
    daily_revenue,
    LAST_VALUE(daily_revenue) OVER (
        PARTITION BY DATE_TRUNC('month', order_date)
        ORDER BY order_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS last_day_of_month
FROM daily_sales;
```

### NTH_VALUE()

```sql
-- Get the 3rd highest value in each partition
SELECT 
    category,
    product_name,
    price,
    NTH_VALUE(price, 3) OVER (
        PARTITION BY category
        ORDER BY price DESC
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS third_highest_price
FROM products;
```

---

## 6.5 Window Frame Specifications

The frame clause defines exactly which rows are included in the window.

### Frame Syntax

```sql
ROWS BETWEEN start_bound AND end_bound
-- or
RANGE BETWEEN start_bound AND end_bound
```

### Bound Options

| Bound | Meaning |
|-------|---------|
| `UNBOUNDED PRECEDING` | Start of partition |
| `n PRECEDING` | n rows before current |
| `CURRENT ROW` | Current row |
| `n FOLLOWING` | n rows after current |
| `UNBOUNDED FOLLOWING` | End of partition |

### Frame Examples

```sql
-- All rows from start to current (running total)
ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW

-- Last 7 days including today
ROWS BETWEEN 6 PRECEDING AND CURRENT ROW

-- Centered window (1 before, current, 1 after)
ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING

-- Entire partition
ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING

-- Only future rows (excluding current)
ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING
```

### ROWS vs RANGE

```sql
-- ROWS: Physical row count
SELECT 
    order_date,
    amount,
    SUM(amount) OVER (
        ORDER BY order_date
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS sum_last_3_rows
FROM orders;

-- RANGE: Logical value range (useful for dates with gaps)
SELECT 
    order_date,
    amount,
    SUM(amount) OVER (
        ORDER BY order_date
        RANGE BETWEEN INTERVAL '2 days' PRECEDING AND CURRENT ROW
    ) AS sum_last_3_days
FROM orders;
```

### Visual Frame Reference

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        WINDOW FRAME VISUALIZATION                            │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Partition: [Row1] [Row2] [Row3] [Row4*] [Row5] [Row6] [Row7]               │
│                                     ↑                                        │
│                              Current Row (*)                                 │
│                                                                              │
│   ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW:                          │
│   [Row1] [Row2] [Row3] [Row4*]                                               │
│   ├──────────────────────────┤                                               │
│                                                                              │
│   ROWS BETWEEN 2 PRECEDING AND CURRENT ROW:                                  │
│                [Row2] [Row3] [Row4*]                                         │
│                ├────────────────────┤                                        │
│                                                                              │
│   ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING:                                  │
│                       [Row3] [Row4*] [Row5]                                  │
│                       ├─────────────────────┤                                │
│                                                                              │
│   ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING:                          │
│                              [Row4*] [Row5] [Row6] [Row7]                    │
│                              ├──────────────────────────┤                    │
│                                                                              │
│   ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING:                  │
│   [Row1] [Row2] [Row3] [Row4*] [Row5] [Row6] [Row7]                          │
│   ├────────────────────────────────────────────────────┤                     │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 6.6 Complete Window Function Examples

### Sales Analysis Dashboard

```sql
WITH daily_metrics AS (
    SELECT 
        order_date,
        COUNT(*) AS order_count,
        SUM(total_amount) AS daily_revenue,
        AVG(total_amount) AS avg_order_value
    FROM orders
    GROUP BY order_date
)
SELECT 
    order_date,
    order_count,
    daily_revenue,
    avg_order_value,
    
    -- Running totals
    SUM(daily_revenue) OVER (ORDER BY order_date) AS cumulative_revenue,
    SUM(order_count) OVER (ORDER BY order_date) AS cumulative_orders,
    
    -- Moving averages
    AVG(daily_revenue) OVER (
        ORDER BY order_date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS revenue_ma_7day,
    
    -- Day-over-day comparison
    LAG(daily_revenue, 1) OVER (ORDER BY order_date) AS prev_day_revenue,
    daily_revenue - LAG(daily_revenue, 1) OVER (ORDER BY order_date) AS dod_change,
    
    -- Week-over-week comparison
    LAG(daily_revenue, 7) OVER (ORDER BY order_date) AS same_day_last_week,
    
    -- Percentage of total
    daily_revenue / SUM(daily_revenue) OVER () * 100 AS pct_of_total,
    
    -- Ranking
    RANK() OVER (ORDER BY daily_revenue DESC) AS revenue_rank
    
FROM daily_metrics
ORDER BY order_date;
```

### Customer Behavior Analysis

```sql
SELECT 
    customer_id,
    order_id,
    order_date,
    total_amount,
    
    -- Order sequence for each customer
    ROW_NUMBER() OVER (
        PARTITION BY customer_id 
        ORDER BY order_date
    ) AS order_sequence,
    
    -- First and last order amounts
    FIRST_VALUE(total_amount) OVER (
        PARTITION BY customer_id 
        ORDER BY order_date
    ) AS first_order_amount,
    
    -- Days since previous order
    order_date - LAG(order_date) OVER (
        PARTITION BY customer_id 
        ORDER BY order_date
    ) AS days_since_last_order,
    
    -- Running total per customer
    SUM(total_amount) OVER (
        PARTITION BY customer_id 
        ORDER BY order_date
    ) AS customer_cumulative_spend,
    
    -- Customer's total (for % calculation)
    SUM(total_amount) OVER (PARTITION BY customer_id) AS customer_total,
    
    -- This order as % of customer total
    total_amount / SUM(total_amount) OVER (PARTITION BY customer_id) * 100 AS pct_of_customer_total

FROM orders
ORDER BY customer_id, order_date;
```

### Product Performance

```sql
SELECT 
    category,
    product_name,
    price,
    units_sold,
    revenue,
    
    -- Rank within category
    RANK() OVER (
        PARTITION BY category 
        ORDER BY revenue DESC
    ) AS category_rank,
    
    -- Top product in category
    FIRST_VALUE(product_name) OVER (
        PARTITION BY category 
        ORDER BY revenue DESC
    ) AS category_leader,
    
    -- Percentage of category revenue
    revenue / SUM(revenue) OVER (PARTITION BY category) * 100 AS pct_of_category,
    
    -- Difference from category average
    revenue - AVG(revenue) OVER (PARTITION BY category) AS diff_from_category_avg,
    
    -- Quartile within category
    NTILE(4) OVER (
        PARTITION BY category 
        ORDER BY revenue DESC
    ) AS revenue_quartile

FROM product_sales
ORDER BY category, revenue DESC;
```

---

## 6.7 Window Functions Summary Table

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      WINDOW FUNCTIONS QUICK REFERENCE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  RANKING FUNCTIONS                                                          │
│  ─────────────────────────────────────────────────────────────────────────  │
│  ROW_NUMBER()    Unique sequential number (1,2,3,4...)                      │
│  RANK()          Same rank for ties, then skip (1,1,3,4...)                 │
│  DENSE_RANK()    Same rank for ties, no skip (1,1,2,3...)                   │
│  NTILE(n)        Divide into n equal groups                                 │
│  PERCENT_RANK()  Percentile rank (0 to 1)                                   │
│  CUME_DIST()     Cumulative distribution (0 to 1)                           │
│                                                                             │
│  VALUE FUNCTIONS                                                            │
│  ─────────────────────────────────────────────────────────────────────────  │
│  LAG(col, n)     Value from n rows before                                   │
│  LEAD(col, n)    Value from n rows after                                    │
│  FIRST_VALUE()   First value in window                                      │
│  LAST_VALUE()    Last value in window (careful with frame!)                 │
│  NTH_VALUE(n)    Nth value in window                                        │
│                                                                             │
│  AGGREGATE FUNCTIONS (as window functions)                                  │
│  ─────────────────────────────────────────────────────────────────────────  │
│  SUM()           Running/cumulative sum                                     │
│  AVG()           Moving/rolling average                                     │
│  COUNT()         Running count                                              │
│  MIN() / MAX()   Running min/max                                            │
│  STDDEV()        Running standard deviation                                 │
│                                                                             │
│  FRAME CLAUSES                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│  ROWS BETWEEN ... AND ...                                                   │
│    UNBOUNDED PRECEDING     Start of partition                               │
│    n PRECEDING             n rows before                                    │
│    CURRENT ROW             Current row                                      │
│    n FOLLOWING             n rows after                                     │
│    UNBOUNDED FOLLOWING     End of partition                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# Part 7: Additional SQL Topics

## 7.1 CASE Expressions

```sql
-- Simple CASE
SELECT 
    product_name,
    price,
    CASE 
        WHEN price < 50 THEN 'Budget'
        WHEN price < 200 THEN 'Mid-range'
        WHEN price < 1000 THEN 'Premium'
        ELSE 'Luxury'
    END AS price_tier
FROM products;

-- CASE with aggregation
SELECT 
    COUNT(CASE WHEN status = 'completed' THEN 1 END) AS completed,
    COUNT(CASE WHEN status = 'pending' THEN 1 END) AS pending,
    COUNT(CASE WHEN status = 'cancelled' THEN 1 END) AS cancelled
FROM orders;

-- CASE in ORDER BY
SELECT * FROM products
ORDER BY 
    CASE category
        WHEN 'Electronics' THEN 1
        WHEN 'Clothing' THEN 2
        WHEN 'Books' THEN 3
        ELSE 4
    END;
```

## 7.2 String Functions

```sql
-- Concatenation
SELECT CONCAT(first_name, ' ', last_name) AS full_name FROM employees;
SELECT first_name || ' ' || last_name AS full_name FROM employees;  -- PostgreSQL

-- Case conversion
SELECT UPPER(name), LOWER(email) FROM customers;

-- Substring
SELECT SUBSTRING(phone, 1, 3) AS area_code FROM customers;
SELECT LEFT(name, 1) AS initial FROM customers;
SELECT RIGHT(ssn, 4) AS last_four FROM employees;

-- Trim whitespace
SELECT TRIM(name), LTRIM(name), RTRIM(name) FROM customers;

-- Replace
SELECT REPLACE(phone, '-', '') AS phone_digits FROM customers;

-- Length
SELECT name, LENGTH(name) AS name_length FROM customers;

-- Position/Find
SELECT POSITION('@' IN email) AS at_position FROM customers;
SELECT CHARINDEX('@', email) AS at_position FROM customers;  -- SQL Server
```

## 7.3 Date Functions

```sql
-- Current date/time
SELECT CURRENT_DATE, CURRENT_TIME, CURRENT_TIMESTAMP;
SELECT NOW();  -- PostgreSQL, MySQL

-- Extract parts
SELECT 
    EXTRACT(YEAR FROM order_date) AS year,
    EXTRACT(MONTH FROM order_date) AS month,
    EXTRACT(DAY FROM order_date) AS day,
    EXTRACT(DOW FROM order_date) AS day_of_week
FROM orders;

-- Date arithmetic
SELECT order_date + INTERVAL '30 days' AS due_date FROM orders;
SELECT order_date - INTERVAL '1 month' AS month_ago FROM orders;
SELECT order_date + 30 AS due_date FROM orders;  -- MySQL

-- Date truncation
SELECT DATE_TRUNC('month', order_date) AS month_start FROM orders;
SELECT DATE_TRUNC('week', order_date) AS week_start FROM orders;

-- Date difference
SELECT AGE(NOW(), created_at) AS account_age FROM users;  -- PostgreSQL
SELECT DATEDIFF(NOW(), created_at) AS days_old FROM users;  -- MySQL

-- Formatting
SELECT TO_CHAR(order_date, 'YYYY-MM-DD') FROM orders;  -- PostgreSQL
SELECT DATE_FORMAT(order_date, '%Y-%m-%d') FROM orders;  -- MySQL
```

## 7.4 NULL Handling

```sql
-- COALESCE - returns first non-NULL value
SELECT COALESCE(phone, email, 'No contact') AS contact FROM customers;

-- NULLIF - returns NULL if values are equal
SELECT NULLIF(discount, 0) AS discount FROM orders;  -- Avoids division by zero

-- IS NULL / IS NOT NULL
SELECT * FROM customers WHERE phone IS NULL;
SELECT * FROM customers WHERE phone IS NOT NULL;

-- NULL-safe comparison (MySQL)
SELECT * FROM t1 WHERE col <=> NULL;

-- IFNULL (MySQL) / NVL (Oracle)
SELECT IFNULL(phone, 'N/A') FROM customers;
```

## 7.5 Data Manipulation (INSERT, UPDATE, DELETE)

```sql
-- INSERT single row
INSERT INTO customers (name, email, city)
VALUES ('John Doe', 'john@email.com', 'NYC');

-- INSERT multiple rows
INSERT INTO customers (name, email, city)
VALUES 
    ('Jane Smith', 'jane@email.com', 'LA'),
    ('Bob Wilson', 'bob@email.com', 'Chicago');

-- INSERT from SELECT
INSERT INTO archived_orders
SELECT * FROM orders WHERE order_date < '2024-01-01';

-- UPDATE
UPDATE customers 
SET city = 'San Francisco'
WHERE customer_id = 1;

-- UPDATE with calculation
UPDATE products
SET price = price * 1.1
WHERE category = 'Electronics';

-- UPDATE with JOIN
UPDATE orders o
SET o.status = 'archived'
FROM customers c
WHERE o.customer_id = c.customer_id
  AND c.status = 'inactive';

-- DELETE
DELETE FROM orders WHERE order_id = 101;

-- DELETE with condition
DELETE FROM orders WHERE order_date < '2020-01-01';

-- DELETE all (but keep table structure)
DELETE FROM temp_table;
TRUNCATE TABLE temp_table;  -- Faster, no logging
```

## 7.6 Indexing

```sql
-- Create index
CREATE INDEX idx_customer_email ON customers(email);

-- Create composite index
CREATE INDEX idx_order_customer_date ON orders(customer_id, order_date);

-- Create unique index
CREATE UNIQUE INDEX idx_email_unique ON customers(email);

-- Create partial index (PostgreSQL)
CREATE INDEX idx_active_orders ON orders(order_date) 
WHERE status = 'active';

-- Drop index
DROP INDEX idx_customer_email;

-- Show indexes
SHOW INDEX FROM customers;  -- MySQL
\di customers  -- PostgreSQL
```

**When to Create Indexes:**
- Columns frequently used in WHERE clauses
- Columns used in JOIN conditions
- Columns used in ORDER BY
- Foreign key columns

**When NOT to Index:**
- Small tables (full scan is faster)
- Columns with low cardinality (few unique values)
- Frequently updated columns
- Wide columns (large text fields)

## 7.7 Views

```sql
-- Create view
CREATE VIEW customer_order_summary AS
SELECT 
    c.customer_id,
    c.name,
    COUNT(o.order_id) AS total_orders,
    SUM(o.total_amount) AS total_spent,
    AVG(o.total_amount) AS avg_order_value
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name;

-- Use view like a table
SELECT * FROM customer_order_summary WHERE total_orders > 10;

-- Create or replace view
CREATE OR REPLACE VIEW customer_order_summary AS ...;

-- Drop view
DROP VIEW customer_order_summary;

-- Materialized view (PostgreSQL) - physically stores results
CREATE MATERIALIZED VIEW mv_daily_sales AS
SELECT DATE(order_date) AS date, SUM(total_amount) AS revenue
FROM orders GROUP BY DATE(order_date);

-- Refresh materialized view
REFRESH MATERIALIZED VIEW mv_daily_sales;
```

## 7.8 Transactions

```sql
-- Basic transaction
BEGIN;  -- or START TRANSACTION

UPDATE accounts SET balance = balance - 100 WHERE account_id = 'A';
UPDATE accounts SET balance = balance + 100 WHERE account_id = 'B';

COMMIT;  -- Save changes

-- Rollback on error
BEGIN;

UPDATE accounts SET balance = balance - 100 WHERE account_id = 'A';
-- Something goes wrong...
ROLLBACK;  -- Undo all changes

-- Savepoints
BEGIN;

UPDATE accounts SET balance = balance - 100 WHERE account_id = 'A';
SAVEPOINT after_debit;

UPDATE accounts SET balance = balance + 100 WHERE account_id = 'B';
-- Problem with B...
ROLLBACK TO after_debit;  -- Keep A's changes, undo B

COMMIT;
```

---

# Part 8: Query Execution Order

Understanding the logical order of SQL clause execution helps write correct queries.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                      SQL LOGICAL EXECUTION ORDER                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Written Order:                  Execution Order:                           │
│   ──────────────                  ────────────────                           │
│   1. SELECT                       1. FROM / JOIN                             │
│   2. FROM                         2. WHERE                                   │
│   3. JOIN                         3. GROUP BY                                │
│   4. WHERE                        4. HAVING                                  │
│   5. GROUP BY                     5. SELECT                                  │
│   6. HAVING                       6. DISTINCT                                │
│   7. ORDER BY                     7. ORDER BY                                │
│   8. LIMIT                        8. LIMIT / OFFSET                          │
│                                                                              │
│   This is why:                                                               │
│   • Can't use SELECT aliases in WHERE (SELECT runs after)                    │
│   • CAN use SELECT aliases in ORDER BY (ORDER BY runs after SELECT)          │
│   • Can't use aggregate functions in WHERE (GROUP BY runs after)             │
│   • CAN use aggregate functions in HAVING (HAVING runs after GROUP BY)       │
│                                                                              │
│   Example:                                                                   │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │  SELECT customer_id, SUM(amount) AS total     -- Runs 5th          │    │
│   │  FROM orders                                   -- Runs 1st          │    │
│   │  WHERE order_date >= '2025-01-01'             -- Runs 2nd          │    │
│   │  GROUP BY customer_id                          -- Runs 3rd          │    │
│   │  HAVING SUM(amount) > 1000                     -- Runs 4th          │    │
│   │  ORDER BY total DESC                           -- Runs 6th          │    │
│   │  LIMIT 10;                                     -- Runs 7th          │    │
│   └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

# Quick Reference Card

## SQL Statement Templates

```sql
-- Basic Query
SELECT columns FROM table WHERE condition;

-- Aggregation
SELECT group_col, AGG(col) FROM table GROUP BY group_col HAVING condition;

-- Join
SELECT * FROM t1 [INNER|LEFT|RIGHT|FULL] JOIN t2 ON t1.key = t2.key;

-- Subquery
SELECT * FROM t1 WHERE col IN (SELECT col FROM t2);

-- CTE
WITH cte AS (SELECT ...) SELECT * FROM cte;

-- Window Function
SELECT col, FUNC() OVER (PARTITION BY col ORDER BY col ROWS BETWEEN ...) FROM t;

-- Insert
INSERT INTO table (cols) VALUES (vals);

-- Update  
UPDATE table SET col = val WHERE condition;

-- Delete
DELETE FROM table WHERE condition;
```

## Common Patterns

```sql
-- Top N per group
WITH ranked AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY sort_col DESC) AS rn
    FROM table
)
SELECT * FROM ranked WHERE rn <= N;

-- Running total
SELECT *, SUM(amount) OVER (ORDER BY date) AS running_total FROM table;

-- Year-over-year comparison
SELECT 
    current_year.*,
    LAG(metric, 12) OVER (ORDER BY month) AS same_month_last_year
FROM monthly_data current_year;

-- Percent of total
SELECT *, amount / SUM(amount) OVER () * 100 AS pct_of_total FROM table;

-- Find duplicates
SELECT col, COUNT(*) FROM table GROUP BY col HAVING COUNT(*) > 1;

-- Find gaps in sequence
SELECT id + 1 AS gap_start
FROM table t1
WHERE NOT EXISTS (SELECT 1 FROM table t2 WHERE t2.id = t1.id + 1);
```

---



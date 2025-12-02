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
|----------|------------|---------------|--------------|----------|
| 101 | 1 | Alice | Laptop | 1 |

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

| customer_id | name | city | city_zip |
|-------------|------|------|----------|
| 1 | Alice | NYC | 10001 |
| 2 | Bob | NYC | 10001 |

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

| Scenario | Why Denormalize? |
|----------|------------------|
| Read-heavy workloads | JOINs are expensive; pre-joined data speeds up reads |
| Data warehouses | Analytics queries benefit from star/snowflake schemas |
| Reporting tables | Pre-aggregate data for dashboards |
| Caching layers | Store computed values for performance |
| High-traffic applications | Reduce query complexity at read time |

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

### Statistical Aggregates

```sql
-- Standard deviation
SELECT STDDEV(price) AS price_stddev FROM products;
SELECT STDDEV_POP(price) AS population_stddev FROM products;
SELECT STDDEV_SAMP(price) AS sample_stddev FROM products;

-- Variance
SELECT VARIANCE(price) AS price_variance FROM products;
SELECT VAR_POP(price) AS population_variance FROM products;
SELECT VAR_SAMP(price) AS sample_variance FROM products;

-- Percentile (PostgreSQL)
SELECT 
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price) AS median_price,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY price) AS q1,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY price) AS q3
FROM products;
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
    EXTRACT(YEAR FROM order_date) AS year,
    EXTRACT(MONTH FROM order_date) AS month,
    COUNT(*) AS num_orders,
    SUM(total_amount) AS monthly_revenue
FROM orders
GROUP BY EXTRACT(YEAR FROM order_date), EXTRACT(MONTH FROM order_date)
ORDER BY year, month;
```

### Date Grouping

```sql
-- Group by date (daily)
SELECT 
    DATE(order_date) AS order_day,
    COUNT(*) AS num_orders,
    SUM(total_amount) AS daily_revenue
FROM orders
GROUP BY DATE(order_date)
ORDER BY order_day;

-- Group by week
SELECT 
    DATE_TRUNC('week', order_date) AS week_start,
    COUNT(*) AS num_orders
FROM orders
GROUP BY DATE_TRUNC('week', order_date);

-- Group by month
SELECT 
    DATE_TRUNC('month', order_date) AS month,
    SUM(total_amount) AS monthly_revenue
FROM orders
GROUP BY DATE_TRUNC('month', order_date)
ORDER BY month;
```

---

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


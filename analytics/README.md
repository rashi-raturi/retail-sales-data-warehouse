# Report - Data Analysis and Business Insights

This section summarizes the data analysis process and the resulting reports, providing valuable business insights.

## Data Exploration and Analysis
The EDA process was conducted using SQL queries. 
The analysis followed a structured approach, covering various aspects of the data:

1. **Database Exploration:** Understanding the structure and relationships within the database.
2. **Dimensions Exploration:** Analyzing the characteristics of the dimension tables (customers, products).
3. **Date Range Exploration:** Identifying the time period covered by the data.
4. **Measures Exploration:** Examining key metrics and their distributions.
5. **Magnitude Exploration:** Understanding the scale of different measures.
6. **Ranking Analysis:** Identifying top performers (e.g., customers, products).
7. **Change Over Time Analysis:** Tracking trends and patterns over time.
8. **Cumulative Analysis:** Examining the accumulated values of metrics.
9. **Performance Analysis:** Evaluating the performance of different aspects of the business.
10. **Data Segmentation:** Grouping data into meaningful segments for targeted analysis.
11. **Part-to-Whole Analysis:** Understanding the contribution of different parts to the overall picture.

```mermaid
flowchart TD
    subgraph Analytics["Data Analytics Process Roadmap"]
        direction TB

        subgraph EDA["Phase 1: Exploratory Data Analysis (EDA)"]
            direction TB
            E1["1. Database Exploration<br/><i>Database structure and relationships</i>"]
            E2["2. Dimensions Exploration<br/><i>Dimension tables (customers, products)</i>"]
            E3["3. Date Range Exploration<br/><i>Time horizons and boundaries</i>"]
            E4["4. Measures Exploration<br/><i>Key metric distributions and big numbers</i>"]
            E5["5. Magnitude Exploration<br/><i>Scale and volume comparison</i>"]
            E6["6. Ranking Analysis<br/><i>Top N and Bottom N performance</i>"]

            E1 --> E2 --> E3 --> E4 --> E5 --> E6
        end

        subgraph ADV["Phase 2: Advanced Analytics & Reporting"]
            direction TB
            A7["7. Change Over Time Analysis<br/><i>Trends, shifts, and patterns</i>"]
            A8["8. Cumulative Analysis<br/><i>Running totals and moving aggregations</i>"]
            A9["9. Performance Analysis<br/><i>KPI tracking and benchmarks</i>"]
            A10["10. Part-to-Whole Analysis<br/><i>Proportions and category share</i>"]
            A11["11. Data Segmentation<br/><i>Customer and product segmentation</i>"]
            A12["12. Reporting & Dashboards<br/><i>Actionable business insights</i>"]

            A7 --> A8 --> A9 --> A10 --> A11 --> A12
        end

        EDA ==>|Foundation for Advanced Insights| ADV
    end
```



## Final Reports

Two key reports were generated to provide business insights:

### 1. Customer Report (`12_report_customers.ipynb`)

This report consolidates key customer metrics and behaviors, including:

* Demographics (name, age, location)
* Transaction history (orders, sales, quantity)
* Customer segmentation (VIP, Regular, New)
* Key performance indicators (KPIs):
    * Recency (months since last order)
    * Average order value (AOV)
    * Average monthly spend

This report helps understand customer behavior, identify valuable customer segments, and personalize marketing efforts.

### 2. Product Report (`13_report_products.ipynb`)

This report summarizes key product metrics and performance, including:

* Product details (name, category, cost)
* Sales performance (orders, sales, quantity)
* Product segmentation (High-Performers, Mid-Range, Low-Performers)
* Key performance indicators (KPIs):
    * Recency (months since last sale)
    * Average order revenue (AOR)
    * Average monthly revenue

This report helps analyze product performance, identify top-selling products, and make informed decisions about product development and marketing.

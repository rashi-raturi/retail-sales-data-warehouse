"""
Notebook Test Utilities and Specifications for Databricks Medallion Migration

Zero-external-dependency test harness support library.
Provides:
  - Canonical 20-notebook specification dictionary and metadata
  - Robust JSON / nbformat loader
  - Python AST extractor and bytecode compiler
  - Spark SQL and Databricks magic syntax extractor and validator
  - Cross-feature schema contract validators
"""

import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ==============================================================================
# Canonical 20-Notebook Specification Registry
# ==============================================================================

NOTEBOOK_SPECS: List[Dict[str, Any]] = [
    # --------------------------------------------------------------------------
    # Milestone 1: Database Init & Bronze Ingestion
    # --------------------------------------------------------------------------
    {
        "id": "DWH-01",
        "name": "init_database",
        "path": "data-warehouse/init_database.ipynb",
        "milestone": "M1",
        "layer": "init",
        "category": "Data Warehousing",
        "description": "Database and catalog initialization (bronze, silver, gold)",
        "expected_databases": ["bronze", "silver", "gold"],
    },
    {
        "id": "DWH-02",
        "name": "ddl_bronze",
        "path": "data-warehouse/bronze/ddl_bronze.ipynb",
        "milestone": "M1",
        "layer": "bronze",
        "category": "Data Warehousing",
        "description": "Bronze layer Delta table DDL definitions",
        "expected_tables_created": [
            "bronze.crm_cust_info",
            "bronze.crm_prd_info",
            "bronze.crm_sales_details",
            "bronze.erp_loc_a101",
            "bronze.erp_cust_az12",
            "bronze.erp_px_cat_g1v2",
        ],
    },
    {
        "id": "DWH-03",
        "name": "proc_load_bronze",
        "path": "data-warehouse/bronze/proc_load_bronze.ipynb",
        "milestone": "M1",
        "layer": "bronze",
        "category": "Data Warehousing",
        "description": "Bronze layer CSV ingestion with schema enforcement into Delta",
        "expected_tables_loaded": [
            "bronze.crm_cust_info",
            "bronze.crm_prd_info",
            "bronze.crm_sales_details",
            "bronze.erp_loc_a101",
            "bronze.erp_cust_az12",
            "bronze.erp_px_cat_g1v2",
        ],
        "requires_dynamic_paths": True,
    },
    # --------------------------------------------------------------------------
    # Milestone 2: Silver Layer DDL & Transformations
    # --------------------------------------------------------------------------
    {
        "id": "DWH-04",
        "name": "ddl_silver",
        "path": "data-warehouse/silver/ddl_silver.ipynb",
        "milestone": "M2",
        "layer": "silver",
        "category": "Data Warehousing",
        "description": "Silver layer Delta table DDL definitions with audit columns",
        "expected_tables_created": [
            "silver.crm_cust_info",
            "silver.crm_prd_info",
            "silver.crm_sales_details",
            "silver.erp_loc_a101",
            "silver.erp_cust_az12",
            "silver.erp_px_cat_g1v2",
        ],
        "requires_audit_column": "dwh_create_date",
    },
    {
        "id": "DWH-05",
        "name": "proc_load_silver",
        "path": "data-warehouse/silver/proc_load_silver.ipynb",
        "milestone": "M2",
        "layer": "silver",
        "category": "Data Warehousing",
        "description": "Silver layer ELT transformations, deduplication, and cleaning",
        "expected_tables_loaded": [
            "silver.crm_cust_info",
            "silver.crm_prd_info",
            "silver.crm_sales_details",
            "silver.erp_loc_a101",
            "silver.erp_cust_az12",
            "silver.erp_px_cat_g1v2",
        ],
        "expected_tables_referenced": [
            "bronze.crm_cust_info",
            "bronze.crm_prd_info",
            "bronze.crm_sales_details",
            "bronze.erp_loc_a101",
            "bronze.erp_cust_az12",
            "bronze.erp_px_cat_g1v2",
        ],
    },
    # --------------------------------------------------------------------------
    # Milestone 3: Gold Layer Star Schema
    # --------------------------------------------------------------------------
    {
        "id": "DWH-06",
        "name": "ddl_gold",
        "path": "data-warehouse/gold/ddl_gold.ipynb",
        "milestone": "M3",
        "layer": "gold",
        "category": "Data Warehousing",
        "description": "Gold layer Star Schema Delta tables / views (dim_customers, dim_products, fact_sales)",
        "expected_tables_created": [
            "gold.dim_customers",
            "gold.dim_products",
            "gold.fact_sales",
        ],
        "expected_tables_referenced": [
            "silver.crm_cust_info",
            "silver.crm_prd_info",
            "silver.crm_sales_details",
            "silver.erp_loc_a101",
            "silver.erp_cust_az12",
            "silver.erp_px_cat_g1v2",
        ],
        "requires_both_cost_and_product_cost": True,
        "requires_active_products_filter": True,
    },
    # --------------------------------------------------------------------------
    # Milestone 4: Basic EDA Notebooks (00 - 06)
    # --------------------------------------------------------------------------
    {
        "id": "EDA-00",
        "name": "00_init_database",
        "path": "analytics/eda_scripts/00_init_database.ipynb",
        "milestone": "M4",
        "layer": "eda",
        "category": "Basic EDA",
        "description": "EDA environment initialization and Gold schema verification",
        "expected_tables_referenced": ["gold.dim_customers", "gold.dim_products", "gold.fact_sales"],
    },
    {
        "id": "EDA-01",
        "name": "01_database_exploration",
        "path": "analytics/eda_scripts/01_database_exploration.ipynb",
        "milestone": "M4",
        "layer": "eda",
        "category": "Basic EDA",
        "description": "Database exploration: catalogs, schemas, tables, and columns",
        "expected_tables_referenced": ["gold.dim_customers", "gold.dim_products", "gold.fact_sales"],
    },
    {
        "id": "EDA-02",
        "name": "02_dimensions_exploration",
        "path": "analytics/eda_scripts/02_dimensions_exploration.ipynb",
        "milestone": "M4",
        "layer": "eda",
        "category": "Basic EDA",
        "description": "Dimensions exploration: distinct values, categories, countries",
        "expected_tables_referenced": ["gold.dim_customers", "gold.dim_products"],
    },
    {
        "id": "EDA-03",
        "name": "03_date_range_exploration",
        "path": "analytics/eda_scripts/03_date_range_exploration.ipynb",
        "milestone": "M4",
        "layer": "eda",
        "category": "Basic EDA",
        "description": "Date range exploration: min/max dates and timespans",
        "expected_tables_referenced": ["gold.fact_sales", "gold.dim_customers"],
    },
    {
        "id": "EDA-04",
        "name": "04_measures_exploration",
        "path": "analytics/eda_scripts/04_measures_exploration.ipynb",
        "milestone": "M4",
        "layer": "eda",
        "category": "Basic EDA",
        "description": "Measures exploration: total sales, quantity, average price KPIs",
        "expected_tables_referenced": ["gold.fact_sales"],
    },
    {
        "id": "EDA-05",
        "name": "05_magnitude_analysis",
        "path": "analytics/eda_scripts/05_magnitude_analysis.ipynb",
        "milestone": "M4",
        "layer": "eda",
        "category": "Basic EDA",
        "description": "Magnitude analysis: revenue and order volume breakdowns by dimension",
        "expected_tables_referenced": ["gold.fact_sales", "gold.dim_products", "gold.dim_customers"],
    },
    {
        "id": "EDA-06",
        "name": "06_ranking_analysis",
        "path": "analytics/eda_scripts/06_ranking_analysis.ipynb",
        "milestone": "M4",
        "layer": "eda",
        "category": "Basic EDA",
        "description": "Ranking analysis: top/bottom customers and products via window functions",
        "expected_tables_referenced": ["gold.fact_sales", "gold.dim_products", "gold.dim_customers"],
    },
    # --------------------------------------------------------------------------
    # Milestone 5: Advanced EDA & Reporting Notebooks (07 - 13)
    # --------------------------------------------------------------------------
    {
        "id": "ADV-07",
        "name": "07_change_over_time_analysis",
        "path": "analytics/advanced_eda/07_change_over_time_analysis.ipynb",
        "milestone": "M5",
        "layer": "advanced_eda",
        "category": "Advanced EDA",
        "description": "Change over time analysis: YoY, MoM growth trends",
        "expected_tables_referenced": ["gold.fact_sales"],
        "requires_spark_date_format": True,
    },
    {
        "id": "ADV-08",
        "name": "08_cumulative_analysis",
        "path": "analytics/advanced_eda/08_cumulative_analysis.ipynb",
        "milestone": "M5",
        "layer": "advanced_eda",
        "category": "Advanced EDA",
        "description": "Cumulative analysis: running totals and moving averages",
        "expected_tables_referenced": ["gold.fact_sales"],
        "disallow_duplicate_select": True,
    },
    {
        "id": "ADV-09",
        "name": "09_performance_analysis",
        "path": "analytics/advanced_eda/09_performance_analysis.ipynb",
        "milestone": "M5",
        "layer": "advanced_eda",
        "category": "Advanced EDA",
        "description": "Performance analysis: sales vs target / historical YoY with LAG",
        "expected_tables_referenced": ["gold.fact_sales", "gold.dim_products"],
    },
    {
        "id": "ADV-10",
        "name": "10_data_segmentation",
        "path": "analytics/advanced_eda/10_data_segmentation.ipynb",
        "milestone": "M5",
        "layer": "advanced_eda",
        "category": "Advanced EDA",
        "description": "Data segmentation: customer spending tiers and product cost tiers",
        "expected_tables_referenced": ["gold.dim_products", "gold.fact_sales", "gold.dim_customers"],
        "requires_cost_column": True,
        "disallow_raw_age_function": True,
        "disallow_leading_space_in_segment": True,
    },
    {
        "id": "ADV-11",
        "name": "11_part_to_whole_analysis",
        "path": "analytics/advanced_eda/11_part_to_whole_analysis.ipynb",
        "milestone": "M5",
        "layer": "advanced_eda",
        "category": "Advanced EDA",
        "description": "Part-to-whole analysis: category contribution percentages and share",
        "expected_tables_referenced": ["gold.fact_sales", "gold.dim_products"],
    },
    {
        "id": "ADV-12",
        "name": "12_report_customers",
        "path": "analytics/advanced_eda/12_report_customers.ipynb",
        "milestone": "M5",
        "layer": "advanced_eda",
        "category": "Advanced EDA",
        "description": "Customer 360 analytical reporting table / view",
        "expected_tables_referenced": ["gold.fact_sales", "gold.dim_customers"],
        "expected_views_created": ["gold.report_customers"],
        "disallow_raw_age_function": True,
    },
    {
        "id": "ADV-13",
        "name": "13_report_products",
        "path": "analytics/advanced_eda/13_report_products.ipynb",
        "milestone": "M5",
        "layer": "advanced_eda",
        "category": "Advanced EDA",
        "description": "Product performance and lifecycle reporting table / view",
        "expected_tables_referenced": ["gold.fact_sales", "gold.dim_products"],
        "expected_views_created": ["gold.report_products"],
        "requires_cost_column": True,
        "disallow_raw_age_function": True,
    },
]

NOTEBOOK_BY_PATH: Dict[str, Dict[str, Any]] = {
    spec["path"].replace("\\", "/"): spec for spec in NOTEBOOK_SPECS
}

NOTEBOOK_BY_ID: Dict[str, Dict[str, Any]] = {
    spec["id"]: spec for spec in NOTEBOOK_SPECS
}


# ==============================================================================
# Project Root Discovery
# ==============================================================================

def find_project_root() -> Path:
    """
    Locates the project root directory by inspecting markers (e.g. datasets, DOC.md).
    Guarantees reliable resolution across different working directories.
    """
    candidates = [
        Path.cwd(),
        Path(__file__).resolve().parent.parent,
        Path("d:/Code/Projects/Data Engineering/Proj"),
    ]
    for candidate in candidates:
        if (candidate / "datasets").is_dir() and (candidate / "data-warehouse").is_dir():
            return candidate.resolve()
    # Fallback to current working directory
    return Path.cwd().resolve()


# ==============================================================================
# Notebook Parsing & Extraction Helpers
# ==============================================================================

def load_notebook(nb_path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Reads and parses a Jupyter notebook JSON file.
    Returns (notebook_data, error_message).
    """
    if not nb_path.exists():
        return None, f"File not found: {nb_path}"
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data, None
    except json.JSONDecodeError as e:
        return None, f"JSON Decode Error at line {e.lineno}, col {e.colno}: {e.msg}"
    except UnicodeDecodeError as e:
        return None, f"Encoding Error: File is not valid UTF-8 ({e})"
    except Exception as e:
        return None, f"Read Error: {e}"


def get_cell_source(cell: Dict[str, Any]) -> str:
    """Extracts cell source code as a single unified string."""
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    elif isinstance(source, str):
        return source
    return ""


# ==============================================================================
# AST & SQL Extractor
# ==============================================================================

class SparkSqlCallExtractor(ast.NodeVisitor):
    """AST Visitor that finds all spark.sql(...) and sql(...) string queries."""
    def __init__(self):
        self.sql_queries: List[str] = []

    def visit_Call(self, node: ast.Call):
        # Match spark.sql("...")
        if isinstance(node.func, ast.Attribute) and node.func.attr == "sql":
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                self.sql_queries.append(node.args[0].value)
        # Match standalone sql("...")
        elif isinstance(node.func, ast.Name) and node.func.id == "sql":
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                self.sql_queries.append(node.args[0].value)
        self.generic_visit(node)


def extract_queries_from_code_cell(code_text: str) -> List[Tuple[str, str]]:
    """
    Extracts SQL queries from a code cell.
    Returns list of (query_type, query_text) where query_type is 'magic_sql' or 'spark_sql'.
    """
    queries: List[Tuple[str, str]] = []
    trimmed = code_text.strip()

    if trimmed.startswith("%sql"):
        sql_content = trimmed[4:].strip()
        if sql_content:
            queries.append(("magic_sql", sql_content))
        return queries

    # If it's a Python cell, parse AST and extract spark.sql calls
    code_to_parse = trimmed
    if code_to_parse.startswith("%python"):
        lines = code_to_parse.splitlines()
        code_to_parse = "\n".join(lines[1:])

    try:
        tree = ast.parse(code_to_parse)
        extractor = SparkSqlCallExtractor()
        extractor.visit(tree)
        for q in extractor.sql_queries:
            queries.append(("spark_sql", q))
    except Exception:
        # If AST parse fails, fallback to regex search for spark.sql("""...""")
        pattern = re.compile(r'spark\.sql\s*\(\s*["\']{1,3}(.*?)["\']{1,3}\s*\)', re.DOTALL)
        for match in pattern.finditer(code_text):
            queries.append(("spark_sql", match.group(1)))

    return queries


# ==============================================================================
# Syntax & Dialect Validation Logic
# ==============================================================================

def validate_delimiter_balance(text: str) -> List[str]:
    """Checks balancing of parentheses (), square brackets [], and curly braces {}."""
    errors = []
    parens = 0
    brackets = 0
    braces = 0

    # Strip quoted strings and comments first to prevent false mismatches
    # Simple regex to neutralize strings
    neutralized = re.sub(r"'(?:\\.|[^'\\])*'", "''", text)
    neutralized = re.sub(r'"(?:\\.|[^"\\])*"', '""', neutralized)
    # Neutralize line comments
    neutralized = re.sub(r'--.*$', '', neutralized, flags=re.MULTILINE)
    neutralized = re.sub(r'#.*$', '', neutralized, flags=re.MULTILINE)

    for ch in neutralized:
        if ch == '(':
            parens += 1
        elif ch == ')':
            parens -= 1
            if parens < 0:
                errors.append("Unmatched closing parenthesis ')'")
                parens = 0
        elif ch == '[':
            brackets += 1
        elif ch == ']':
            brackets -= 1
            if brackets < 0:
                errors.append("Unmatched closing bracket ']'")
                brackets = 0
        elif ch == '{':
            braces += 1
        elif ch == '}':
            braces -= 1
            if braces < 0:
                errors.append("Unmatched closing brace '}'")
                braces = 0

    if parens > 0:
        errors.append(f"Unclosed open parenthesis '(' (count: {parens})")
    if brackets > 0:
        errors.append(f"Unclosed open bracket '[' (count: {brackets})")
    if braces > 0:
        errors.append(f"Unclosed open brace '{{' (count: {braces})")

    return errors


def validate_quote_balance(text: str) -> List[str]:
    """Validates that single and double quotes are balanced in SQL statements."""
    errors = []
    # Neutralize escaped quotes
    clean = text.replace(r"\'", "").replace(r'\"', "")
    # Remove triple quotes
    clean = clean.replace("'''", "").replace('"""', "")
    # Remove comments
    clean = re.sub(r'--.*$', '', clean, flags=re.MULTILINE)
    clean = re.sub(r'#.*$', '', clean, flags=re.MULTILINE)

    single_quotes = clean.count("'")
    if single_quotes % 2 != 0:
        errors.append(f"Unbalanced single quotes: odd count ({single_quotes})")

    double_quotes = clean.count('"')
    if double_quotes % 2 != 0:
        errors.append(f"Unbalanced double quotes: odd count ({double_quotes})")

    return errors


def validate_sql_syntax(sql_text: str) -> List[str]:
    """Validates common SQL syntax defects and dialect migration regressions."""
    errors = []
    trimmed = sql_text.strip()
    if not trimmed:
        return ["SQL statement is empty"]

    # 1. Delimiter balance
    errors.extend(validate_delimiter_balance(sql_text))
    errors.extend(validate_quote_balance(sql_text))

    # 2. Duplicate keyword detection (e.g. SELECT SELECT bug from script 08)
    if re.search(r'\bSELECT\s+SELECT\b', sql_text, re.IGNORECASE):
        errors.append("Duplicate 'SELECT SELECT' detected in query")

    # 3. Trailing comma before FROM or closing parenthesis
    if re.search(r',\s*FROM\b', sql_text, re.IGNORECASE):
        errors.append("Trailing comma directly before 'FROM' keyword")
    if re.search(r',\s*\)', sql_text):
        errors.append("Trailing comma directly before closing parenthesis ')'")

    # 4. PostgreSQL AGE() function prohibition
    if re.search(r'\bAGE\s*\(', sql_text, re.IGNORECASE):
        errors.append("PostgreSQL 'AGE()' function detected; must be translated to Spark SQL 'months_between()'")

    # 5. PostgreSQL :: casting prohibition
    if re.search(r'::\s*(text|int|integer|varchar|numeric|date|timestamp)', sql_text, re.IGNORECASE):
        errors.append("PostgreSQL '::' typecast operator detected; must use 'CAST(expr AS type)'")

    return errors


def extract_table_references(text: str) -> Set[str]:
    """Extracts table references in format schema.table from SQL or code text."""
    pattern = re.compile(r'\b(bronze|silver|gold)\.([a-zA-Z0-9_]+)\b', re.IGNORECASE)
    matches = set()
    for schema, table in pattern.findall(text):
        matches.add(f"{schema.lower()}.{table.lower()}")
    return matches

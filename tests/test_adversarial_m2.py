"""
Adversarial Verification Suite for Milestone 2 (Silver Layer DDL & ELT Transformations)
Author: worker_silver
Date: 2026-10-28

Empirically challenges:
1. Exact column parity & ordering between PostgreSQL DDL (ddl_silver.sql) and Databricks Delta DDL (ddl_silver.ipynb).
2. Delta Lake table creation, overwrite configuration, and schema enforcement behavior.
3. Every silver table defines and populates mandatory audit column `dwh_create_date TIMESTAMP`.
4. High-fidelity conversion of all ELT business rules from proc_load_silver.sql into PySpark operations:
   - crm_cust_info: null filtering, row_number deduplication, name trimming, marital status & gender normalization.
   - crm_prd_info: cat_id & clean prd_key extraction, cost imputation, line mapping, SCD Type 2 lead expiry window.
   - crm_sales_details: YYYYMMDD integer parsing, sales amount recalculation, price derivation.
   - erp_cust_az12: 'NAS' prefix stripping, future birthdate nullification, gender standardization.
   - erp_loc_a101: hyphen stripping, country standardization.
   - erp_px_cat_g1v2: metadata enrichment and direct passthrough.
5. Dialect safety: Zero unmigrated PostgreSQL functions (AGE, clock_timestamp) or typecast operators (::).
"""

import ast
import json
import re
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

class AdversarialMilestone2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = ROOT_DIR
        cls.ddl_silver_nb = cls.root / "data-warehouse" / "silver" / "ddl_silver.ipynb"
        cls.proc_load_nb = cls.root / "data-warehouse" / "silver" / "proc_load_silver.ipynb"
        cls.ddl_silver_sql = cls.root / "data-warehouse" / "silver" / "ddl_silver.sql"
        cls.proc_load_sql = cls.root / "data-warehouse" / "silver" / "proc_load_silver.sql"

        # Load notebooks
        with open(cls.ddl_silver_nb, "r", encoding="utf-8") as f:
            cls.ddl_silver_json = json.load(f)
        with open(cls.proc_load_nb, "r", encoding="utf-8") as f:
            cls.proc_load_json = json.load(f)

        # Extract code cells
        cls.ddl_silver_code = "\n".join("".join(c["source"]) for c in cls.ddl_silver_json["cells"] if c["cell_type"] == "code")
        cls.proc_load_code = "\n".join("".join(c["source"]) for c in cls.proc_load_json["cells"] if c["cell_type"] == "code")

        # Extract original SQL
        with open(cls.ddl_silver_sql, "r", encoding="utf-8") as f:
            cls.original_ddl_sql = f.read()
        with open(cls.proc_load_sql, "r", encoding="utf-8") as f:
            cls.original_proc_sql = f.read()

        # Parse DDL columns
        cls.ddl_parsed = cls._parse_ddl_tables(cls.ddl_silver_code)
        cls.pg_ddl_parsed = cls._parse_pg_ddl(cls.original_ddl_sql)

    @staticmethod
    def _parse_ddl_tables(code: str):
        tables = {}
        pattern = re.compile(r'CREATE\s+TABLE\s+(silver\.\w+)\s*\((.*?)\)\s*USING\s+DELTA', re.DOTALL | re.IGNORECASE)
        for match in pattern.finditer(code):
            tbl_name = match.group(1).lower()
            body = match.group(2)
            cols = []
            for line in body.strip().split("\n"):
                line = line.strip().rstrip(",")
                if not line or line.startswith("--"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    col_name = parts[0].strip("`\"").lower()
                    col_type = parts[1].strip(",").upper()
                    cols.append((col_name, col_type))
            tables[tbl_name] = cols
        return tables

    @staticmethod
    def _parse_pg_ddl(sql: str):
        tables = {}
        pattern = re.compile(r'CREATE\s+TABLE\s+(silver\.\w+)\s*\((.*?)\);', re.DOTALL | re.IGNORECASE)
        for match in pattern.finditer(sql):
            tbl_name = match.group(1).lower()
            body = match.group(2)
            cols = []
            for line in body.strip().split("\n"):
                line = line.strip().rstrip(",")
                if not line or line.startswith("--"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    col_name = parts[0].strip("`\"").lower()
                    col_type = parts[1].strip(",").upper()
                    cols.append((col_name, col_type))
            tables[tbl_name] = cols
        return tables

    def test_ddl_exact_table_count(self):
        """Assert exactly 6 Silver tables are parsed from DDL notebook."""
        self.assertEqual(len(self.ddl_parsed), 6)
        expected = [
            "silver.crm_cust_info",
            "silver.crm_prd_info",
            "silver.crm_sales_details",
            "silver.erp_loc_a101",
            "silver.erp_cust_az12",
            "silver.erp_px_cat_g1v2",
        ]
        self.assertCountEqual(list(self.ddl_parsed.keys()), expected)

    def test_ddl_1to1_column_name_parity_with_postgres_source(self):
        """Assert 100% column name parity between ddl_silver.sql and ddl_silver.ipynb."""
        for tbl, pg_cols in self.pg_ddl_parsed.items():
            delta_cols = self.ddl_parsed.get(tbl)
            self.assertIsNotNone(delta_cols, f"Table {tbl} missing in Delta DDL")

            pg_names = [c[0] for c in pg_cols]
            delta_names = [c[0] for c in delta_cols]

            self.assertEqual(
                delta_names, pg_names,
                f"Column names or ordering mismatch for table {tbl}:\n  PG: {pg_names}\n  Delta: {delta_names}"
            )

    def test_audit_column_presence_across_all_tables(self):
        """Verify audit column dwh_create_date TIMESTAMP exists in all DDL and load operations."""
        for tbl, cols in self.ddl_parsed.items():
            col_dict = dict(cols)
            self.assertIn("dwh_create_date", col_dict, f"Table {tbl} DDL missing audit column")
            self.assertEqual(col_dict["dwh_create_date"], "TIMESTAMP")

        # In proc_load_silver.ipynb, verify current_timestamp() is aliased to dwh_create_date 6 times
        alias_matches = re.findall(r'current_timestamp\(\)\.alias\(["\']dwh_create_date["\']\)', self.proc_load_code)
        self.assertEqual(len(alias_matches), 6, "Expected exactly 6 dwh_create_date assignments in ELT notebook")

    def test_delta_storage_and_overwrite_mode(self):
        """Ensure all 6 table writes use Delta format and overwrite mode."""
        delta_writes = len(re.findall(r'\.format\(["\']delta["\']\)', self.proc_load_code))
        overwrite_modes = len(re.findall(r'\.mode\(["\']overwrite["\']\)', self.proc_load_code))
        self.assertEqual(delta_writes, 6)
        self.assertEqual(overwrite_modes, 6)

    def test_customer_deduplication_integrity(self):
        """Empirically test that customer deduplication logic satisfies requirements."""
        self.assertIn('Window.partitionBy("cst_id").orderBy(col("cst_create_date").desc())', self.proc_load_code)
        self.assertIn('row_number().over(cust_window)', self.proc_load_code)
        self.assertIn('col("flag_last") == 1', self.proc_load_code)
        self.assertIn('col("cst_id").isNotNull()', self.proc_load_code)

    def test_product_scd_type2_window_integrity(self):
        """Verify product key decomposition and SCD Type 2 date calculations."""
        self.assertIn('regexp_replace(substring(col("prd_key"), 1, 5), "-", "_")', self.proc_load_code)
        self.assertIn('substring(col("prd_key"), 7, length(col("prd_key")))', self.proc_load_code)
        self.assertIn('coalesce(col("prd_cost"), lit(0))', self.proc_load_code)
        self.assertIn('Window.partitionBy("clean_prd_key").orderBy("prd_start_dt_date")', self.proc_load_code)
        self.assertIn('date_sub(lead(col("prd_start_dt_date")).over(prd_window), 1)', self.proc_load_code)

    def test_sales_integer_date_and_math_reconciliation(self):
        """Verify sales date parsing and sales quantity * abs(price) mathematical reconciliation."""
        self.assertIn('to_date(col("order_dt_str"), "yyyyMMdd")', self.proc_load_code)
        self.assertIn('to_date(col("ship_dt_str"), "yyyyMMdd")', self.proc_load_code)
        self.assertIn('to_date(col("due_dt_str"), "yyyyMMdd")', self.proc_load_code)
        self.assertIn('col("sls_quantity") * spark_abs(col("sls_price"))', self.proc_load_code)
        self.assertIn('(col("sls_sales") / when(col("sls_quantity") == 0, lit(None)).otherwise(col("sls_quantity"))).cast("int")', self.proc_load_code)

    def test_erp_transformations_integrity(self):
        """Verify ERP customer and location normalization."""
        # 'NAS' prefix stripping
        self.assertIn('when(col("cid").startswith("NAS"), substring(col("cid"), 4, length(col("cid"))))', self.proc_load_code)
        # Future birthdates set to null
        self.assertIn('when(col("bdate") > current_date(), lit(None))', self.proc_load_code)
        # Hyphen stripping
        self.assertIn('regexp_replace(col("cid"), "-", "")', self.proc_load_code)

    def test_dialect_cleanliness(self):
        """Verify no unmigrated PostgreSQL dialect remains."""
        for name, code in [("ddl_silver", self.ddl_silver_code), ("proc_load_silver", self.proc_load_code)]:
            self.assertFalse(re.search(r'\bAGE\s*\(', code, re.IGNORECASE), f"{name} contains AGE()")
            self.assertFalse(re.search(r'\bclock_timestamp\s*\(', code, re.IGNORECASE), f"{name} contains clock_timestamp()")
            self.assertFalse(re.search(r'::\s*(text|int|integer|varchar|date)', code, re.IGNORECASE), f"{name} contains :: typecast")


if __name__ == "__main__":
    unittest.main()

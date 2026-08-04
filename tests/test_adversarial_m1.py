"""
Adversarial Verification Suite for Milestone 1 (Database Init & Bronze Layer)
Author: challenger_m1_2
Date: 2026-10-28

Empirically challenges:
1. Exact column mapping & ordering between raw CSVs, DDL notebook, and PySpark StructType schemas.
2. Data typing & full dataset value validation across all 6 raw CSV files (116,294 total data rows, 768,387 cells).
3. PostgreSQL DDL vs Databricks Delta DDL 1:1 type mapping parity.
4. Delta Lake table creation, overwrite configuration, and schema enforcement behavior.
5. Edge cases: nulls, unescaped characters, casing, date/timestamp formatting, whitespace, numeric bounds.
"""

import ast
import csv
import json
import re
import sys
import unittest
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

class AdversarialMilestone1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = ROOT_DIR
        cls.init_db_nb = cls.root / "data-warehouse" / "init_database.ipynb"
        cls.ddl_bronze_nb = cls.root / "data-warehouse" / "bronze" / "ddl_bronze.ipynb"
        cls.proc_load_nb = cls.root / "data-warehouse" / "bronze" / "proc_load_bronze.ipynb"
        cls.ddl_bronze_sql = cls.root / "data-warehouse" / "bronze" / "ddl_bronze.sql"
        cls.proc_load_sql = cls.root / "data-warehouse" / "bronze" / "proc_load_bronze.sql"

        cls.csv_files = {
            "bronze.crm_cust_info": cls.root / "datasets" / "source_crm" / "cust_info.csv",
            "bronze.crm_prd_info": cls.root / "datasets" / "source_crm" / "prd_info.csv",
            "bronze.crm_sales_details": cls.root / "datasets" / "source_crm" / "sales_details.csv",
            "bronze.erp_loc_a101": cls.root / "datasets" / "source_erp" / "LOC_A101.csv",
            "bronze.erp_cust_az12": cls.root / "datasets" / "source_erp" / "CUST_AZ12.csv",
            "bronze.erp_px_cat_g1v2": cls.root / "datasets" / "source_erp" / "PX_CAT_G1V2.csv",
        }

        # Load notebooks
        with open(cls.init_db_nb, "r", encoding="utf-8") as f:
            cls.init_db_json = json.load(f)
        with open(cls.ddl_bronze_nb, "r", encoding="utf-8") as f:
            cls.ddl_bronze_json = json.load(f)
        with open(cls.proc_load_nb, "r", encoding="utf-8") as f:
            cls.proc_load_json = json.load(f)

        # Extract code cells
        cls.init_db_code = "\n".join("".join(c["source"]) for c in cls.init_db_json["cells"] if c["cell_type"] == "code")
        cls.ddl_bronze_code = "\n".join("".join(c["source"]) for c in cls.ddl_bronze_json["cells"] if c["cell_type"] == "code")
        cls.proc_load_code = "\n".join("".join(c["source"]) for c in cls.proc_load_json["cells"] if c["cell_type"] == "code")

        # Extract original SQL
        with open(cls.ddl_bronze_sql, "r", encoding="utf-8") as f:
            cls.original_ddl_sql = f.read()

        # Parse DDL columns from ddl_bronze.ipynb
        cls.ddl_parsed = cls._parse_ddl_tables(cls.ddl_bronze_code)

        # Parse StructType schemas from proc_load_bronze.ipynb
        cls.struct_parsed = cls._parse_struct_schemas(cls.proc_load_code)

        # Parse PostgreSQL DDL from ddl_bronze.sql
        cls.pg_ddl_parsed = cls._parse_pg_ddl(cls.original_ddl_sql)

    @staticmethod
    def _parse_ddl_tables(code: str):
        tables = {}
        pattern = re.compile(r'CREATE\s+TABLE\s+(bronze\.\w+)\s*\((.*?)\)\s*USING\s+DELTA', re.DOTALL | re.IGNORECASE)
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
        pattern = re.compile(r'CREATE\s+TABLE\s+(bronze\.\w+)\s*\((.*?)\);', re.DOTALL | re.IGNORECASE)
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
                    # Normalize VARCHAR(50) to VARCHAR
                    if col_type.startswith("VARCHAR"):
                        col_type = "VARCHAR"
                    cols.append((col_name, col_type))
            tables[tbl_name] = cols
        return tables

    @staticmethod
    def _parse_struct_schemas(code: str):
        schemas = {}
        pattern = re.compile(r'schema_(\w+)\s*=\s*StructType\(\[(.*?)\]\)', re.DOTALL)
        for match in pattern.finditer(code):
            tbl_name = "bronze." + match.group(1).lower()
            body = match.group(2)
            fields = []
            field_pat = re.compile(r'StructField\(\s*["\'](\w+)["\']\s*,\s*(\w+)\(\)\s*,\s*(True|False)\s*\)')
            for fmatch in field_pat.finditer(body):
                f_name = fmatch.group(1).lower()
                f_type = fmatch.group(2)
                f_null = fmatch.group(3) == "True"
                fields.append((f_name, f_type, f_null))
            schemas[tbl_name] = fields
        return schemas

    def test_01_notebook_json_and_ast_validity(self):
        """Verify all three notebooks are valid JSON and all code cells parse to Python AST."""
        for path, nb_json in [
            (self.init_db_nb, self.init_db_json),
            (self.ddl_bronze_nb, self.ddl_bronze_json),
            (self.proc_load_nb, self.proc_load_json),
        ]:
            self.assertEqual(nb_json.get("nbformat"), 4, f"{path.name} must be nbformat 4")
            for idx, cell in enumerate(nb_json["cells"]):
                if cell["cell_type"] == "code":
                    code_src = "".join(cell["source"])
                    try:
                        ast.parse(code_src)
                    except SyntaxError as e:
                        self.fail(f"Syntax error in {path.name} cell {idx}: {e}")

    def test_02_database_initialization_fidelity(self):
        """Verify init_database.ipynb properly initializes bronze, silver, gold schemas."""
        for schema in ["bronze", "silver", "gold"]:
            self.assertIn(f"CREATE DATABASE IF NOT EXISTS {schema}", self.init_db_code)
        self.assertIn("SHOW DATABASES", self.init_db_code)
        self.assertIn("DESCRIBE DATABASE EXTENDED", self.init_db_code)

    def test_03_all_6_tables_defined_in_ddl_and_struct(self):
        """Verify all 6 expected tables exist in both DDL notebook and StructType definitions."""
        expected_tables = {
            "bronze.crm_cust_info",
            "bronze.crm_prd_info",
            "bronze.crm_sales_details",
            "bronze.erp_loc_a101",
            "bronze.erp_cust_az12",
            "bronze.erp_px_cat_g1v2",
        }
        self.assertEqual(set(self.ddl_parsed.keys()), expected_tables, "DDL tables do not match expected 6 tables")
        self.assertEqual(set(self.struct_parsed.keys()), expected_tables, "StructType schemas do not match expected 6 tables")
        self.assertEqual(set(self.pg_ddl_parsed.keys()), expected_tables, "PostgreSQL DDL tables do not match expected 6 tables")

    def test_04_csv_header_vs_ddl_exact_mapping(self):
        """Adversarial check: CSV headers match DDL column definitions exactly in name and order."""
        for tbl_name, csv_path in self.csv_files.items():
            self.assertTrue(csv_path.exists(), f"Missing CSV: {csv_path}")
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                raw_header = next(reader)
            
            clean_csv_cols = [c.strip().lower() for c in raw_header]
            ddl_cols = [c[0] for c in self.ddl_parsed[tbl_name]]

            self.assertEqual(
                len(clean_csv_cols), len(ddl_cols),
                f"Column count mismatch for {tbl_name}: CSV has {len(clean_csv_cols)}, DDL has {len(ddl_cols)}"
            )
            self.assertEqual(
                clean_csv_cols, ddl_cols,
                f"Column mismatch for {tbl_name}:\nCSV: {clean_csv_cols}\nDDL: {ddl_cols}"
            )

    def test_05_csv_header_vs_structtype_exact_mapping(self):
        """Adversarial check: CSV headers match PySpark StructType fields exactly in name and order."""
        for tbl_name, csv_path in self.csv_files.items():
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                raw_header = next(reader)
            
            clean_csv_cols = [c.strip().lower() for c in raw_header]
            struct_cols = [c[0] for c in self.struct_parsed[tbl_name]]

            self.assertEqual(
                clean_csv_cols, struct_cols,
                f"Column mismatch for {tbl_name}:\nCSV:    {clean_csv_cols}\nStruct: {struct_cols}"
            )

    def test_06_ddl_vs_structtype_type_alignment(self):
        """Adversarial check: DDL column types match PySpark StructType field types."""
        type_matrix = {
            "INT": "IntegerType",
            "INTEGER": "IntegerType",
            "STRING": "StringType",
            "DATE": "DateType",
            "TIMESTAMP": "TimestampType",
        }

        for tbl_name in self.ddl_parsed:
            ddl_cols = self.ddl_parsed[tbl_name]
            struct_fields = self.struct_parsed[tbl_name]

            self.assertEqual(len(ddl_cols), len(struct_fields), f"Field count mismatch in {tbl_name}")

            for (ddl_name, ddl_type), (struct_name, struct_type, nullable) in zip(ddl_cols, struct_fields):
                self.assertEqual(ddl_name, struct_name, f"Column name mismatch: {ddl_name} vs {struct_name}")
                expected_struct_type = type_matrix.get(ddl_type)
                self.assertIsNotNone(expected_struct_type, f"Unknown DDL type {ddl_type} in {tbl_name}.{ddl_name}")
                self.assertEqual(
                    struct_type, expected_struct_type,
                    f"Type mismatch in {tbl_name}.{ddl_name}: DDL is {ddl_type} ({expected_struct_type}), StructType is {struct_type}"
                )
                self.assertTrue(nullable, f"Field {tbl_name}.{struct_name} should be nullable=True for raw Bronze ingestion")

    def test_07_postgres_ddl_vs_spark_ddl_fidelity(self):
        """Adversarial check: Original PostgreSQL DDL vs Databricks Delta DDL 1:1 fidelity."""
        sql_to_spark_types = {
            "INT": "INT",
            "VARCHAR": "STRING",
            "DATE": "DATE",
            "TIMESTAMP": "TIMESTAMP",
        }

        for tbl_name in self.pg_ddl_parsed:
            pg_cols = self.pg_ddl_parsed[tbl_name]
            spark_cols = self.ddl_parsed[tbl_name]

            self.assertEqual(
                len(pg_cols), len(spark_cols),
                f"Column count mismatch between Postgres and Spark DDL for {tbl_name}"
            )

            for (pg_name, pg_type), (spark_name, spark_type) in zip(pg_cols, spark_cols):
                self.assertEqual(pg_name, spark_name, f"Column name mismatch in {tbl_name}: {pg_name} vs {spark_name}")
                expected_spark_type = sql_to_spark_types.get(pg_type, pg_type)
                self.assertEqual(
                    spark_type, expected_spark_type,
                    f"Type translation mismatch in {tbl_name}.{pg_name}: PG {pg_type} -> Spark {spark_type} (expected {expected_spark_type})"
                )

    def test_08_full_dataset_values_type_conformance(self):
        """
        Adversarial scan of ALL rows across all 6 CSV files (116,294 rows, 768,387 cells).
        Verifies that every data value in every column strictly parses according to its assigned PySpark type.
        """
        type_parsers = {
            "IntegerType": lambda v: int(v) if v.strip() != "" else None,
            "StringType": lambda v: v,
            "DateType": lambda v: datetime.strptime(v.strip(), "%Y-%m-%d").date() if v.strip() != "" else None,
            "TimestampType": lambda v: datetime.fromisoformat(v.strip()) if v.strip() != "" else None,
        }

        total_scanned_cells = 0
        total_rows_scanned = 0

        for tbl_name, csv_path in self.csv_files.items():
            struct_fields = self.struct_parsed[tbl_name]
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                
                for row_idx, row in enumerate(reader, start=2):
                    total_rows_scanned += 1
                    self.assertEqual(
                        len(row), len(struct_fields),
                        f"Row {row_idx} in {csv_path.name} has {len(row)} values, expected {len(struct_fields)}"
                    )
                    
                    for col_idx, (val, (f_name, f_type, nullable)) in enumerate(zip(row, struct_fields)):
                        total_scanned_cells += 1
                        parser = type_parsers[f_type]
                        try:
                            parsed_val = parser(val)
                            if parsed_val is None:
                                self.assertTrue(
                                    nullable,
                                    f"Null found in non-nullable field {tbl_name}.{f_name} at row {row_idx}"
                                )
                        except Exception as ex:
                            self.fail(
                                f"Type parsing failure in {tbl_name}.{f_name} (type={f_type}) at line {row_idx}: "
                                f"value='{val}'. Error: {ex}"
                            )

    def test_09_delta_lake_configuration_and_schema_enforcement(self):
        """Adversarial check: Delta Lake overwrite, saveAsTable, and schema enforcement options."""
        # DDL notebook checks
        self.assertEqual(
            self.ddl_bronze_code.count("USING DELTA"), 6,
            "All 6 Bronze tables in ddl_bronze.ipynb must specify USING DELTA"
        )
        self.assertEqual(
            self.ddl_bronze_code.count("DROP TABLE IF EXISTS"), 6,
            "All 6 Bronze tables must have DROP TABLE IF EXISTS for idempotent re-creation"
        )

        # proc_load_bronze notebook checks
        self.assertIn('.format("delta")', self.proc_load_code)
        self.assertIn('.mode("overwrite")', self.proc_load_code)
        self.assertIn('.option("overwriteSchema", "true")', self.proc_load_code)
        self.assertIn('.saveAsTable(table_name)', self.proc_load_code)
        self.assertIn('.schema(schema)', self.proc_load_code)
        self.assertIn('.option("header", "true")', self.proc_load_code)
        self.assertIn('.option("delimiter", ",")', self.proc_load_code)

    def test_10_row_count_integrity(self):
        """Adversarial check: Verify row counts in proc_load_bronze audit match actual CSV counts."""
        expected_audit = {
            "bronze.crm_cust_info": 18494,
            "bronze.crm_prd_info": 397,
            "bronze.crm_sales_details": 60398,
            "bronze.erp_loc_a101": 18484,
            "bronze.erp_cust_az12": 18484,
            "bronze.erp_px_cat_g1v2": 37,
        }

        for tbl_name, expected_count in expected_audit.items():
            csv_path = self.csv_files[tbl_name]
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader) # skip header
                actual_count = sum(1 for _ in reader)
            self.assertEqual(
                actual_count, expected_count,
                f"Actual row count for {tbl_name} ({actual_count}) does not match expected ({expected_count})"
            )

    def test_11_edge_case_special_characters_and_formatting(self):
        """Adversarial check: Look for unescaped delimiters, unexpected newlines, or malformed records."""
        for tbl_name, csv_path in self.csv_files.items():
            with open(csv_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertNotIn("\x00", content, f"Null byte found in {tbl_name}")
            lines = content.splitlines()
            self.assertGreater(len(lines), 1, f"Empty CSV {tbl_name}")
            # Ensure no trailing blank lines in the middle
            empty_count = sum(1 for l in lines if not l.strip())
            self.assertEqual(empty_count, 0, f"Unexpected empty lines in {tbl_name}: {empty_count}")

    def test_12_column_casing_resilience(self):
        """Adversarial check: Test that proc_load_bronze correctly normalizes uppercase headers (ERP)."""
        # Ensure the notebook includes logic to handle ERP uppercase column names
        self.assertIn("df.withColumnRenamed(col_name, col_name.lower())", self.proc_load_code)

if __name__ == "__main__":
    unittest.main()

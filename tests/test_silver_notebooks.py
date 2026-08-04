"""
Unit and Static Verification Tests for Silver Layer Databricks Notebooks

Covers:
- data-warehouse/silver/ddl_silver.ipynb
- data-warehouse/silver/proc_load_silver.ipynb
"""

import ast
import json
import re
import unittest
from pathlib import Path

class TestSilverNotebooks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root_dir = Path(__file__).resolve().parent.parent
        cls.ddl_silver_path = cls.root_dir / "data-warehouse" / "silver" / "ddl_silver.ipynb"
        cls.proc_load_path = cls.root_dir / "data-warehouse" / "silver" / "proc_load_silver.ipynb"

    def _load_notebook(self, path: Path):
        self.assertTrue(path.exists(), f"Notebook file {path} does not exist")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def test_ddl_silver_schema_and_ast(self):
        """Verify ddl_silver.ipynb format, metadata, AST compilation, and table definitions."""
        data = self._load_notebook(self.ddl_silver_path)
        self.assertEqual(data.get("nbformat"), 4)
        self.assertIn(data.get("nbformat_minor"), [4, 5])
        self.assertEqual(data["metadata"]["language_info"]["name"], "python")

        code_text = ""
        has_markdown = False
        has_code = False

        for idx, cell in enumerate(data["cells"]):
            cell_type = cell.get("cell_type")
            if cell_type == "markdown":
                has_markdown = True
            elif cell_type == "code":
                has_code = True
                source = "".join(cell.get("source", []))
                self.assertTrue(source.strip(), f"Cell {idx} is empty")
                code_text += source + "\n"
                # Ensure every code cell compiles cleanly
                ast.parse(source)

        self.assertTrue(has_markdown, "ddl_silver.ipynb missing markdown documentation")
        self.assertTrue(has_code, "ddl_silver.ipynb missing code cells")

        # Verify silver database creation
        self.assertIn("CREATE DATABASE IF NOT EXISTS silver", code_text)

        # Expected 6 Silver Delta tables
        expected_tables = [
            "silver.crm_cust_info",
            "silver.crm_prd_info",
            "silver.crm_sales_details",
            "silver.erp_loc_a101",
            "silver.erp_cust_az12",
            "silver.erp_px_cat_g1v2"
        ]

        for tbl in expected_tables:
            with self.subTest(table=tbl):
                self.assertIn(tbl, code_text)
                self.assertIn(f"DROP TABLE IF EXISTS {tbl}", code_text)
                self.assertIn(f"CREATE TABLE {tbl}", code_text)

        # Ensure all 6 tables use Delta Lake format
        self.assertEqual(code_text.count("USING DELTA"), 6)

        # Ensure audit column dwh_create_date TIMESTAMP is defined in all tables
        self.assertEqual(code_text.count("dwh_create_date TIMESTAMP"), 6)

    def test_proc_load_silver_schema_and_ast(self):
        """Verify proc_load_silver.ipynb format, metadata, and Python AST compilation."""
        data = self._load_notebook(self.proc_load_path)
        self.assertEqual(data.get("nbformat"), 4)
        self.assertIn(data.get("nbformat_minor"), [4, 5])
        self.assertEqual(data["metadata"]["language_info"]["name"], "python")

        code_text = ""
        has_markdown = False
        has_code = False

        for idx, cell in enumerate(data["cells"]):
            cell_type = cell.get("cell_type")
            if cell_type == "markdown":
                has_markdown = True
            elif cell_type == "code":
                has_code = True
                source = "".join(cell.get("source", []))
                self.assertTrue(source.strip(), f"Cell {idx} is empty")
                code_text += source + "\n"
                # Check Python AST syntax
                ast.parse(source)

        self.assertTrue(has_markdown, "proc_load_silver.ipynb missing markdown documentation")
        self.assertTrue(has_code, "proc_load_silver.ipynb missing code cells")

    def test_proc_load_silver_elt_business_logic(self):
        """Verify all data cleaning, deduplication, and transformation business rules in proc_load_silver."""
        data = self._load_notebook(self.proc_load_path)
        code_text = "\n".join("".join(c["source"]) for c in data["cells"] if c["cell_type"] == "code")

        # 1. Source tables read from bronze layer
        bronze_tables = [
            "bronze.crm_cust_info",
            "bronze.crm_prd_info",
            "bronze.crm_sales_details",
            "bronze.erp_loc_a101",
            "bronze.erp_cust_az12",
            "bronze.erp_px_cat_g1v2"
        ]
        for tbl in bronze_tables:
            with self.subTest(source_table=tbl):
                self.assertIn(f'spark.table("{tbl}")', code_text)

        # 2. Target tables written to silver layer with Delta overwrite
        silver_tables = [
            "silver.crm_cust_info",
            "silver.crm_prd_info",
            "silver.crm_sales_details",
            "silver.erp_loc_a101",
            "silver.erp_cust_az12",
            "silver.erp_px_cat_g1v2"
        ]
        for tbl in silver_tables:
            with self.subTest(target_table=tbl):
                self.assertIn(f'saveAsTable("{tbl}")', code_text)

        self.assertEqual(code_text.count('.format("delta")'), 6)
        self.assertEqual(code_text.count('.mode("overwrite")'), 6)

        # 3. Mandatory audit column in all 6 loads
        self.assertGreaterEqual(code_text.count('current_timestamp().alias("dwh_create_date")'), 6)

        # 4. Customer deduplication & cleaning checks
        self.assertIn('Window.partitionBy("cst_id").orderBy(col("cst_create_date").desc())', code_text)
        self.assertIn('col("cst_id").isNotNull()', code_text)
        self.assertIn('row_number().over(cust_window)', code_text)
        self.assertIn('col("flag_last") == 1', code_text)
        self.assertIn('trim(col("cst_firstname"))', code_text)
        self.assertIn('trim(col("cst_lastname"))', code_text)
        self.assertIn('"Married"', code_text)
        self.assertIn('"Single"', code_text)
        self.assertIn('"Female"', code_text)
        self.assertIn('"Male"', code_text)

        # 5. Product transformation checks
        self.assertIn('regexp_replace(substring(col("prd_key"), 1, 5), "-", "_")', code_text)
        self.assertIn('substring(col("prd_key"), 7, length(col("prd_key")))', code_text)
        self.assertIn('coalesce(col("prd_cost"), lit(0))', code_text)
        self.assertIn('"Mountain"', code_text)
        self.assertIn('"Road"', code_text)
        self.assertIn('"Other Sales"', code_text)
        self.assertIn('"Touring"', code_text)
        self.assertIn('lead(col("prd_start_dt_date"))', code_text)
        self.assertIn('date_sub(', code_text)

        # 6. Sales details validation & math checks
        self.assertIn('to_date(col("order_dt_str"), "yyyyMMdd")', code_text)
        self.assertIn('to_date(col("ship_dt_str"), "yyyyMMdd")', code_text)
        self.assertIn('to_date(col("due_dt_str"), "yyyyMMdd")', code_text)
        self.assertIn('col("sls_quantity") * spark_abs(col("sls_price"))', code_text)
        self.assertIn('when(col("sls_quantity") == 0, lit(None)).otherwise(col("sls_quantity"))', code_text)

        # 7. ERP customer normalization checks
        self.assertIn('col("cid").startswith("NAS")', code_text)
        self.assertIn('col("bdate") > current_date()', code_text)

        # 8. ERP location normalization checks
        self.assertIn('regexp_replace(col("cid"), "-", "")', code_text)
        self.assertIn('"Germany"', code_text)
        self.assertIn('"United States"', code_text)

        # 9. No forbidden PostgreSQL dialect remnants
        self.assertFalse(re.search(r'\bAGE\s*\(', code_text, re.IGNORECASE))
        self.assertFalse(re.search(r'::\s*(text|int|integer|varchar|date)', code_text, re.IGNORECASE))


if __name__ == "__main__":
    unittest.main()

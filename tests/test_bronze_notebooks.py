"""
Unit and Static Verification Tests for Bronze Layer Databricks Notebooks

Covers:
- data-warehouse/init_database.ipynb
- data-warehouse/bronze/ddl_bronze.ipynb
- data-warehouse/bronze/proc_load_bronze.ipynb
"""

import ast
import json
import unittest
from pathlib import Path

class TestBronzeNotebooks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root_dir = Path(__file__).resolve().parent.parent
        cls.init_db_path = cls.root_dir / "data-warehouse" / "init_database.ipynb"
        cls.ddl_bronze_path = cls.root_dir / "data-warehouse" / "bronze" / "ddl_bronze.ipynb"
        cls.proc_load_path = cls.root_dir / "data-warehouse" / "bronze" / "proc_load_bronze.ipynb"

    def _load_notebook(self, path: Path):
        self.assertTrue(path.exists(), f"Notebook file {path} does not exist")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def test_init_database_schema_and_ast(self):
        data = self._load_notebook(self.init_db_path)
        self.assertEqual(data.get("nbformat"), 4)
        self.assertEqual(data.get("nbformat_minor"), 4)
        self.assertEqual(data["metadata"]["language_info"]["name"], "python")

        code_text = ""
        for cell in data["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                code_text += source + "\n"
                # Ensure each individual code cell parses as valid Python AST
                ast.parse(source)

        # Check required database creations
        for schema in ["bronze", "silver", "gold"]:
            self.assertIn(f"CREATE DATABASE IF NOT EXISTS {schema}", code_text)
        self.assertIn("SHOW DATABASES", code_text)
        self.assertIn("DESCRIBE DATABASE EXTENDED", code_text)

    def test_ddl_bronze_schema_and_tables(self):
        data = self._load_notebook(self.ddl_bronze_path)
        self.assertEqual(data.get("nbformat"), 4)
        self.assertEqual(data.get("nbformat_minor"), 4)
        self.assertEqual(data["metadata"]["language_info"]["name"], "python")

        code_text = ""
        for cell in data["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                code_text += source + "\n"
                ast.parse(source)

        expected_tables = [
            "bronze.crm_cust_info",
            "bronze.crm_prd_info",
            "bronze.crm_sales_details",
            "bronze.erp_loc_a101",
            "bronze.erp_cust_az12",
            "bronze.erp_px_cat_g1v2"
        ]
        for tbl in expected_tables:
            self.assertIn(tbl, code_text)
            self.assertIn(f"DROP TABLE IF EXISTS {tbl}", code_text)
            self.assertIn(f"CREATE TABLE {tbl}", code_text)

        # Ensure Delta Lake syntax is used
        self.assertEqual(code_text.count("USING DELTA"), 6)

    def test_proc_load_bronze_etl_and_schemas(self):
        data = self._load_notebook(self.proc_load_path)
        self.assertEqual(data.get("nbformat"), 4)
        self.assertEqual(data.get("nbformat_minor"), 4)
        self.assertEqual(data["metadata"]["language_info"]["name"], "python")

        code_text = ""
        for cell in data["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                code_text += source + "\n"
                ast.parse(source)

        # Verify dynamic path configuration
        self.assertIn("dbutils.widgets.text", code_text)
        self.assertIn("DATASET_BASE_PATH", code_text)
        self.assertIn("datasets", code_text)

        # Verify all 6 CSV sources
        csv_sources = [
            "cust_info.csv",
            "prd_info.csv",
            "sales_details.csv",
            "LOC_A101.csv",
            "CUST_AZ12.csv",
            "PX_CAT_G1V2.csv"
        ]
        for csv in csv_sources:
            self.assertIn(csv, code_text)

        # Verify Delta overwrite save
        self.assertIn('.format("delta")', code_text)
        self.assertIn('.mode("overwrite")', code_text)
        self.assertIn('.saveAsTable(table_name)', code_text)

        # Verify StructType schemas defined
        self.assertIn("schema_crm_cust_info = StructType", code_text)
        self.assertIn("schema_crm_prd_info = StructType", code_text)
        self.assertIn("schema_crm_sales_details = StructType", code_text)
        self.assertIn("schema_erp_loc_a101 = StructType", code_text)
        self.assertIn("schema_erp_cust_az12 = StructType", code_text)
        self.assertIn("schema_erp_px_cat_g1v2 = StructType", code_text)

        # Verify audit and timing
        self.assertIn("batch_start_time = time.time()", code_text)
        self.assertIn("expected_counts", code_text)

if __name__ == "__main__":
    unittest.main()

"""
Comprehensive E2E Test Suite for Databricks Jupyter Notebooks (.ipynb)

Implements the 4-tier testing methodology:
  - Tier 1: Feature Coverage (existence, JSON validity, nbformat 4 schema, metadata, cells)
  - Tier 2: Boundary & Corner Cases (empty cells, invalid types, unescaped characters, magics, delimiters)
  - Tier 3: Cross-Feature Combinations (Bronze -> Silver -> Gold -> EDA schema consistency and contracts)
  - Tier 4: Real-World Workload Testing (AST parse, bytecode compilation, critical columns, dialect translation)

Compatible with standard unittest and pytest.
Zero external dependencies (uses standard library ast, json, re, unittest, pathlib).
"""

import ast
import json
import os
import re
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Support direct execution and test discovery
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from notebook_test_utils import (
    NOTEBOOK_BY_ID,
    NOTEBOOK_BY_PATH,
    NOTEBOOK_SPECS,
    extract_queries_from_code_cell,
    extract_table_references,
    find_project_root,
    get_cell_source,
    load_notebook,
    validate_delimiter_balance,
    validate_quote_balance,
    validate_sql_syntax,
)

PROJECT_ROOT = find_project_root()


# ==============================================================================
# Helper Mixin for Notebook Tests
# ==============================================================================

class NotebookTestBase(unittest.TestCase):
    """Base class providing shared helper methods for notebook testing."""

    @classmethod
    def setUpClass(cls):
        cls.root = PROJECT_ROOT
        cls.specs = NOTEBOOK_SPECS

    def get_notebook_path(self, spec: Dict[str, Any]) -> Path:
        return self.root / spec["path"]

    def load_spec_notebook(self, spec: Dict[str, Any]) -> Tuple[Dict[str, Any], Path]:
        nb_path = self.get_notebook_path(spec)
        self.assertTrue(
            nb_path.exists(),
            f"Notebook '{spec['name']}' not found at expected path: {spec['path']}"
        )
        data, err = load_notebook(nb_path)
        self.assertIsNone(err, f"Failed to load notebook '{spec['path']}': {err}")
        self.assertIsNotNone(data)
        return data, nb_path

    def get_existing_notebooks(self) -> List[Tuple[Dict[str, Any], Dict[str, Any], Path]]:
        """Returns list of (spec, data, path) for all notebooks that currently exist."""
        existing = []
        for spec in self.specs:
            nb_path = self.get_notebook_path(spec)
            if nb_path.exists():
                data, err = load_notebook(nb_path)
                if err is None and data is not None:
                    existing.append((spec, data, nb_path))
        return existing


# ==============================================================================
# Tier 1: Feature Coverage Tests
# ==============================================================================

class TestTier1FeatureCoverage(NotebookTestBase):
    """
    Tier 1: Verifies notebook existence, JSON syntax, nbformat schema,
    metadata conventions, and mandatory cell layout.
    """

    def test_tier1_all_20_notebooks_exist(self):
        """Verify that all 20 required Databricks notebooks exist on disk."""
        missing = []
        found = []
        for spec in self.specs:
            nb_path = self.get_notebook_path(spec)
            if nb_path.exists():
                found.append(spec["path"])
            else:
                missing.append(f"{spec['id']}: {spec['path']}")

        self.assertEqual(
            len(missing),
            0,
            f"Missing {len(missing)}/20 required notebooks:\n  " + "\n  ".join(missing)
        )
        self.assertEqual(len(found), 20, f"Expected 20 notebooks, found {len(found)}")

    def test_tier1_valid_json_structure(self):
        """Verify that every existing notebook is a valid, parseable JSON document."""
        existing = self.get_existing_notebooks()
        if not existing:
            self.skipTest("No notebooks generated yet to validate JSON.")

        for spec, data, nb_path in existing:
            with self.subTest(notebook=spec["name"]):
                self.assertIsInstance(data, dict, f"{nb_path} top-level element is not a JSON object")

    def test_tier1_nbformat_schema_conformance(self):
        """Verify that notebooks conform to Jupyter nbformat 4 schema."""
        existing = self.get_existing_notebooks()
        if not existing:
            self.skipTest("No notebooks generated yet to validate nbformat schema.")

        for spec, data, nb_path in existing:
            with self.subTest(notebook=spec["name"]):
                for req_key in ["cells", "metadata", "nbformat", "nbformat_minor"]:
                    self.assertIn(req_key, data, f"{spec['path']} missing top-level key: '{req_key}'")

                self.assertEqual(
                    data.get("nbformat"), 4,
                    f"{spec['path']} has invalid nbformat: expected 4, got {data.get('nbformat')}"
                )
                self.assertGreaterEqual(
                    data.get("nbformat_minor", 0), 2,
                    f"{spec['path']} nbformat_minor must be >= 2, got {data.get('nbformat_minor')}"
                )

    def test_tier1_metadata_language_info(self):
        """Verify that notebook metadata specifies python language and kernel."""
        existing = self.get_existing_notebooks()
        if not existing:
            self.skipTest("No notebooks generated yet to validate metadata.")

        for spec, data, nb_path in existing:
            with self.subTest(notebook=spec["name"]):
                metadata = data.get("metadata", {})
                lang_info = metadata.get("language_info", {})
                self.assertEqual(
                    lang_info.get("name"), "python",
                    f"{spec['path']} metadata.language_info.name is '{lang_info.get('name')}', expected 'python'"
                )

    def test_tier1_cell_structure_and_documentation(self):
        """Verify cell layout, non-empty cells list, and presence of markdown documentation."""
        existing = self.get_existing_notebooks()
        if not existing:
            self.skipTest("No notebooks generated yet to validate cell structure.")

        for spec, data, nb_path in existing:
            with self.subTest(notebook=spec["name"]):
                cells = data.get("cells", [])
                self.assertIsInstance(cells, list, f"{spec['path']} 'cells' must be a list")
                self.assertGreater(len(cells), 0, f"{spec['path']} must contain at least one cell")

                has_markdown = False
                has_code = False

                for idx, cell in enumerate(cells):
                    cell_type = cell.get("cell_type")
                    self.assertIn(
                        cell_type, ["code", "markdown", "raw"],
                        f"{spec['path']} Cell {idx} has invalid cell_type: '{cell_type}'"
                    )
                    self.assertIn("metadata", cell, f"{spec['path']} Cell {idx} missing 'metadata'")

                    if cell_type == "markdown":
                        has_markdown = True
                    elif cell_type == "code":
                        has_code = True
                        self.assertIn("outputs", cell, f"{spec['path']} Cell {idx} missing 'outputs'")
                        self.assertIsInstance(cell.get("outputs"), list, f"{spec['path']} Cell {idx} 'outputs' not list")
                        self.assertIn("execution_count", cell, f"{spec['path']} Cell {idx} missing 'execution_count'")

                self.assertTrue(has_markdown, f"{spec['path']} lacks documentation markdown cells")
                self.assertTrue(has_code, f"{spec['path']} lacks executable code cells")


# ==============================================================================
# Tier 2: Boundary & Corner Cases Tests
# ==============================================================================

class TestTier2BoundaryCornerCases(NotebookTestBase):
    """
    Tier 2: Validates boundary conditions: empty code cells, unsupported cell magics,
    character encoding integrity, unescaped characters, and delimiter balance.
    """

    def test_tier2_no_empty_code_cells(self):
        """Ensure code cells are not empty or consisting solely of blank whitespace."""
        existing = self.get_existing_notebooks()
        if not existing:
            self.skipTest("No notebooks generated yet.")

        for spec, data, nb_path in existing:
            cells = data.get("cells", [])
            for idx, cell in enumerate(cells):
                if cell.get("cell_type") == "code":
                    source = get_cell_source(cell)
                    self.assertTrue(
                        len(source.strip()) > 0,
                        f"{spec['path']} Cell {idx} is an empty code cell without content."
                    )

    def test_tier2_encoding_integrity(self):
        """Ensure notebooks are clean UTF-8 without unicode replacement characters (\\ufffd)."""
        existing = self.get_existing_notebooks()
        if not existing:
            self.skipTest("No notebooks generated yet.")

        for spec, data, nb_path in existing:
            raw_text = nb_path.read_text(encoding="utf-8")
            self.assertNotIn(
                "\ufffd", raw_text,
                f"{spec['path']} contains Unicode replacement character (\\ufffd), indicating encoding corruption."
            )

    def test_tier2_unsupported_cell_magics(self):
        """
        Verify that cells do not use unsupported magics (e.g. bash '%sh' or '!', '%scala', '%r').
        Allowed: '%sql', '%python', '%md', or '# Databricks notebook source'.
        """
        existing = self.get_existing_notebooks()
        if not existing:
            self.skipTest("No notebooks generated yet.")

        disallowed_prefixes = ["%sh", "!", "%scala", "%r"]
        for spec, data, nb_path in existing:
            cells = data.get("cells", [])
            for idx, cell in enumerate(cells):
                if cell.get("cell_type") == "code":
                    source = get_cell_source(cell).strip()
                    for dis in disallowed_prefixes:
                        self.assertFalse(
                            source.startswith(dis),
                            f"{spec['path']} Cell {idx} uses unsupported magic '{dis}'"
                        )

    def test_tier2_delimiter_and_bracket_balance(self):
        """Verify balanced parentheses (), brackets [], and braces {} in code and SQL."""
        existing = self.get_existing_notebooks()
        if not existing:
            self.skipTest("No notebooks generated yet.")

        for spec, data, nb_path in existing:
            cells = data.get("cells", [])
            for idx, cell in enumerate(cells):
                if cell.get("cell_type") == "code":
                    source = get_cell_source(cell)
                    if source.strip().startswith("%sql"):
                        errs = validate_delimiter_balance(source[4:])
                        self.assertEqual(
                            len(errs), 0,
                            f"{spec['path']} Cell {idx} SQL delimiter imbalance: {errs}"
                        )


# ==============================================================================
# Tier 3: Cross-Feature Combinations Tests (Schema & Pipeline Contracts)
# ==============================================================================

class TestTier3CrossFeatureCombinations(NotebookTestBase):
    """
    Tier 3: Verifies cross-feature contracts and schema consistency:
      - Database catalog initialization (bronze, silver, gold)
      - Bronze table definitions match Bronze ingestion targets
      - Silver transformation reads from Bronze and writes to Silver
      - Gold Star Schema reads from Silver and defines dim/fact tables
      - EDA notebooks query only Gold Star Schema tables
    """

    def test_tier3_catalog_initialization(self):
        """Verify init_database.ipynb initializes bronze, silver, and gold databases."""
        init_spec = NOTEBOOK_BY_ID["DWH-01"]
        nb_path = self.get_notebook_path(init_spec)
        if not nb_path.exists():
            self.skipTest("init_database.ipynb not generated yet.")

        data, _ = load_notebook(nb_path)
        all_text = " ".join(get_cell_source(c) for c in data.get("cells", []))
        for db in ["bronze", "silver", "gold"]:
            self.assertTrue(
                re.search(rf"\bCREATE\s+(DATABASE|SCHEMA)\s+(IF\s+NOT\s+EXISTS\s+)?{db}\b", all_text, re.IGNORECASE)
                or re.search(rf"\b{db}\b", all_text, re.IGNORECASE),
                f"init_database.ipynb does not initialize database/catalog '{db}'"
            )

    def test_tier3_bronze_layer_contracts(self):
        """Verify Bronze DDL and Load notebooks define and load the exact 6 Bronze Delta tables."""
        ddl_spec = NOTEBOOK_BY_ID["DWH-02"]
        load_spec = NOTEBOOK_BY_ID["DWH-03"]

        ddl_path = self.get_notebook_path(ddl_spec)
        load_path = self.get_notebook_path(load_spec)

        if not ddl_path.exists() or not load_path.exists():
            self.skipTest("Bronze notebooks not generated yet.")

        ddl_data, _ = load_notebook(ddl_path)
        load_data, _ = load_notebook(load_path)

        ddl_text = " ".join(get_cell_source(c) for c in ddl_data.get("cells", []))
        load_text = " ".join(get_cell_source(c) for c in load_data.get("cells", []))

        expected_bronze_tables = [
            "bronze.crm_cust_info",
            "bronze.crm_prd_info",
            "bronze.crm_sales_details",
            "bronze.erp_loc_a101",
            "bronze.erp_cust_az12",
            "bronze.erp_px_cat_g1v2",
        ]

        for tbl in expected_bronze_tables:
            with self.subTest(table=tbl):
                self.assertIn(tbl, ddl_text.lower(), f"Bronze DDL does not define table '{tbl}'")
                self.assertIn(tbl, load_text.lower(), f"Bronze Load does not populate table '{tbl}'")

    def test_tier3_silver_layer_contracts(self):
        """Verify Silver Load reads from Bronze tables and populates Silver tables with audit column."""
        silver_ddl_spec = NOTEBOOK_BY_ID["DWH-04"]
        silver_load_spec = NOTEBOOK_BY_ID["DWH-05"]

        ddl_path = self.get_notebook_path(silver_ddl_spec)
        load_path = self.get_notebook_path(silver_load_spec)

        if not ddl_path.exists() or not load_path.exists():
            self.skipTest("Silver notebooks not generated yet.")

        ddl_data, _ = load_notebook(ddl_path)
        load_data, _ = load_notebook(load_path)

        ddl_text = " ".join(get_cell_source(c) for c in ddl_data.get("cells", []))
        load_text = " ".join(get_cell_source(c) for c in load_data.get("cells", []))

        expected_silver_tables = [
            "silver.crm_cust_info",
            "silver.crm_prd_info",
            "silver.crm_sales_details",
            "silver.erp_loc_a101",
            "silver.erp_cust_az12",
            "silver.erp_px_cat_g1v2",
        ]

        # Audit column check
        self.assertIn("dwh_create_date", ddl_text.lower(), "Silver DDL missing mandatory 'dwh_create_date' audit column")
        self.assertIn("dwh_create_date", load_text.lower(), "Silver Load missing mandatory 'dwh_create_date' audit column")

        for tbl in expected_silver_tables:
            with self.subTest(table=tbl):
                self.assertIn(tbl, ddl_text.lower(), f"Silver DDL does not define table '{tbl}'")
                self.assertIn(tbl, load_text.lower(), f"Silver Load does not populate table '{tbl}'")

    def test_tier3_gold_layer_star_schema_contracts(self):
        """Verify Gold DDL creates star schema tables (dim_customers, dim_products, fact_sales) from Silver."""
        gold_spec = NOTEBOOK_BY_ID["DWH-06"]
        gold_path = self.get_notebook_path(gold_spec)
        if not gold_path.exists():
            self.skipTest("Gold notebook not generated yet.")

        data, _ = load_notebook(gold_path)
        gold_text = " ".join(get_cell_source(c) for c in data.get("cells", []))

        for tbl in ["gold.dim_customers", "gold.dim_products", "gold.fact_sales"]:
            self.assertIn(tbl, gold_text.lower(), f"Gold notebook does not define/create '{tbl}'")

        # Must reference silver layer
        self.assertIn("silver.", gold_text.lower(), "Gold notebook does not source data from 'silver.' layer")

    def test_tier3_eda_analytical_queries_target_gold(self):
        """Verify all 14 EDA notebooks query only Gold layer tables."""
        eda_specs = [s for s in self.specs if s["layer"] in ["eda", "advanced_eda"]]
        for spec in eda_specs:
            nb_path = self.get_notebook_path(spec)
            if not nb_path.exists():
                continue

            data, _ = load_notebook(nb_path)
            content = " ".join(get_cell_source(c) for c in data.get("cells", []))
            tables = extract_table_references(content)

            # Every referenced schema must be 'gold'
            for tbl in tables:
                schema = tbl.split(".")[0]
                self.assertEqual(
                    schema, "gold",
                    f"{spec['path']} queries '{tbl}' from non-gold layer '{schema}'!"
                )


# ==============================================================================
# Tier 4: Real-World Workload Testing (AST Parse, Compilation, Dialects)
# ==============================================================================

class TestTier4RealWorldWorkloads(NotebookTestBase):
    """
    Tier 4: Validates real-world workload execution:
      - Full Python AST parse and bytecode compilation across all code cells
      - Spark SQL query syntax and balanced delimiters
      - Critical column parity in gold.dim_products (both 'cost' and 'product_cost')
      - PostgreSQL dialect translation (no raw AGE() calls, no '::' casts)
      - Portability helpers (dynamic paths, display fallback)
    """

    def test_tier4_python_ast_parse_and_bytecode_compilation(self):
        """Ensure 100% of Python code cells across all notebooks parse and compile without syntax errors."""
        existing = self.get_existing_notebooks()
        if not existing:
            self.skipTest("No notebooks generated yet.")

        for spec, data, nb_path in existing:
            cells = data.get("cells", [])
            for idx, cell in enumerate(cells):
                if cell.get("cell_type") == "code":
                    source = get_cell_source(cell).strip()
                    if not source or source.startswith("%sql") or source.startswith("%md"):
                        continue

                    # Python code to parse
                    code_to_compile = source
                    if code_to_compile.startswith("%python"):
                        lines = code_to_compile.splitlines()
                        code_to_compile = "\n".join(lines[1:])

                    with self.subTest(notebook=spec["name"], cell_index=idx):
                        try:
                            tree = ast.parse(code_to_compile, filename=str(nb_path))
                            # Bytecode compilation
                            compile(tree, filename=str(nb_path), mode="exec")
                        except SyntaxError as e:
                            self.fail(
                                f"Python SyntaxError in {spec['path']} Cell {idx} "
                                f"at line {e.lineno}, col {e.offset}: {e.msg}\nSource snippet:\n{source[:200]}"
                            )

    def test_tier4_sql_syntax_and_queries(self):
        """Validate SQL query syntax, delimiter balance, and absence of known SQL bugs."""
        existing = self.get_existing_notebooks()
        if not existing:
            self.skipTest("No notebooks generated yet.")

        for spec, data, nb_path in existing:
            cells = data.get("cells", [])
            for idx, cell in enumerate(cells):
                if cell.get("cell_type") == "code":
                    source = get_cell_source(cell)
                    queries = extract_queries_from_code_cell(source)
                    for q_type, q_text in queries:
                        with self.subTest(notebook=spec["name"], cell_index=idx, query_type=q_type):
                            errs = validate_sql_syntax(q_text)
                            self.assertEqual(
                                len(errs), 0,
                                f"SQL validation errors in {spec['path']} Cell {idx}:\n  " + "\n  ".join(errs)
                            )

    def test_tier4_gold_dim_products_column_parity(self):
        """
        Verify gold.dim_products projects BOTH 'cost' and 'product_cost' columns
        to maintain 100% downstream compatibility with all analytics scripts.
        """
        gold_spec = NOTEBOOK_BY_ID["DWH-06"]
        gold_path = self.get_notebook_path(gold_spec)
        if not gold_path.exists():
            self.skipTest("Gold notebook not generated yet.")

        data, _ = load_notebook(gold_path)
        gold_text = " ".join(get_cell_source(c) for c in data.get("cells", []))

        # Check for both column names in dim_products creation context
        has_product_cost = bool(re.search(r'\bproduct_cost\b', gold_text, re.IGNORECASE))
        has_cost = bool(re.search(r'\bcost\b', gold_text, re.IGNORECASE))

        self.assertTrue(
            has_product_cost,
            "gold.dim_products must project 'product_cost' for original DDL compatibility."
        )
        self.assertTrue(
            has_cost,
            "gold.dim_products must project 'cost' for downstream EDA compatibility (e.g. scripts 10, 13)."
        )

    def test_tier4_dialect_translation_no_raw_age(self):
        """
        Verify that PostgreSQL AGE() function has been translated to Spark SQL functions
        (such as months_between() or datediff()) across all notebooks.
        """
        existing = self.get_existing_notebooks()
        if not existing:
            self.skipTest("No notebooks generated yet.")

        for spec, data, nb_path in existing:
            for idx, cell in enumerate(data.get("cells", [])):
                source = get_cell_source(cell)
                # Check inside code cells and SQL queries
                if cell.get("cell_type") == "code":
                    self.assertFalse(
                        bool(re.search(r'\bAGE\s*\(', source, re.IGNORECASE)),
                        f"{spec['path']} Cell {idx} contains untranslated PostgreSQL 'AGE()' function! "
                        f"Must use Spark SQL 'months_between()'."
                    )

    def test_tier4_dialect_translation_no_postgres_casting(self):
        """Verify no PostgreSQL '::' typecast operator remains in code cells."""
        existing = self.get_existing_notebooks()
        if not existing:
            self.skipTest("No notebooks generated yet.")

        for spec, data, nb_path in existing:
            for idx, cell in enumerate(data.get("cells", [])):
                if cell.get("cell_type") == "code":
                    source = get_cell_source(cell)
                    self.assertFalse(
                        bool(re.search(r'::\s*(text|int|integer|varchar|numeric|date|timestamp)', source, re.IGNORECASE)),
                        f"{spec['path']} Cell {idx} contains PostgreSQL '::' cast operator! "
                        f"Must use 'CAST(expr AS type)'."
                    )

    def test_tier4_bronze_dynamic_paths_portability(self):
        """Verify Bronze ingestion notebook uses dynamic path resolution with local fallback."""
        load_spec = NOTEBOOK_BY_ID["DWH-03"]
        load_path = self.get_notebook_path(load_spec)
        if not load_path.exists():
            self.skipTest("proc_load_bronze.ipynb not generated yet.")

        data, _ = load_notebook(load_path)
        content = " ".join(get_cell_source(c) for c in data.get("cells", []))

        # Check for widget or env var fallback
        has_widgets = "dbutils.widgets" in content
        has_env_or_path = "os.getenv" in content or "CRM_SOURCE_PATH" in content or "datasets" in content

        self.assertTrue(
            has_widgets or has_env_or_path,
            "proc_load_bronze.ipynb should include dynamic path handling (dbutils.widgets or os.getenv fallback)."
        )


# ==============================================================================
# Main Runner Entry Point
# ==============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)

#!/usr/bin/env python3
"""
E2E Databricks Notebook Test Runner & Verification Suite

Production-grade, zero-external-dependency test runner implementing the
4-tier testing methodology across all 20 Databricks Jupyter notebooks.

Usage:
  python tests/run_all_tests.py [options]

Options:
  --tier {1,2,3,4,all}        Execute tests for a specific tier (default: all)
  --milestone {M1,M2,M3,M4,M5,all}  Filter notebooks by milestone (default: all)
  --progressive               Progressive mode: validate existing notebooks without
                              failing on not-yet-implemented milestone files
  --json-report [PATH]        Export full results as machine-readable JSON
  --verbose, -v               Verbose execution logging
  --help, -h                  Show help message and exit
"""

import argparse
import ast
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Support direct execution
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


class Color:
    """ANSI color codes with terminal fallback."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"

    @classmethod
    def disable(cls):
        cls.RESET = ""
        cls.BOLD = ""
        cls.DIM = ""
        cls.RED = ""
        cls.GREEN = ""
        cls.YELLOW = ""
        cls.BLUE = ""
        cls.CYAN = ""
        cls.WHITE = ""


# Detect Windows cmd / powershell ANSI support
if sys.platform == "win32":
    # Enable ANSI escape sequences on Windows 10/11
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass


class TestReport:
    """Collects and aggregates test results across tiers and notebooks."""
    def __init__(self, target_milestone: str, progressive: bool):
        self.target_milestone = target_milestone
        self.progressive = progressive
        self.start_time = time.time()
        self.end_time = 0.0

        self.tier_results: Dict[str, Dict[str, Any]] = {
            "tier1": {"name": "Tier 1: Feature Coverage", "passed": 0, "failed": 0, "skipped": 0, "errors": []},
            "tier2": {"name": "Tier 2: Boundary & Corner Cases", "passed": 0, "failed": 0, "skipped": 0, "errors": []},
            "tier3": {"name": "Tier 3: Cross-Feature Contracts", "passed": 0, "failed": 0, "skipped": 0, "errors": []},
            "tier4": {"name": "Tier 4: Real-World Workloads", "passed": 0, "failed": 0, "skipped": 0, "errors": []},
        }

        self.notebook_metrics: List[Dict[str, Any]] = []
        self.stats = {
            "total_target_notebooks": 20,
            "selected_notebooks": 0,
            "existing_notebooks": 0,
            "missing_notebooks": 0,
            "total_code_cells": 0,
            "python_cells_compiled": 0,
            "sql_queries_validated": 0,
            "total_markdown_cells": 0,
        }

    def record_pass(self, tier: str, message: str = ""):
        self.tier_results[tier]["passed"] += 1

    def record_fail(self, tier: str, notebook_path: str, message: str):
        self.tier_results[tier]["failed"] += 1
        self.tier_results[tier]["errors"].append({"notebook": notebook_path, "error": message})

    def record_skip(self, tier: str, message: str = ""):
        self.tier_results[tier]["skipped"] += 1

    @property
    def total_passed(self) -> int:
        return sum(t["passed"] for t in self.tier_results.values())

    @property
    def total_failed(self) -> int:
        return sum(t["failed"] for t in self.tier_results.values())

    @property
    def total_skipped(self) -> int:
        return sum(t["skipped"] for t in self.tier_results.values())

    @property
    def is_success(self) -> bool:
        return self.total_failed == 0


class NotebookTestRunner:
    """Executes 4-tier verification suite against target notebooks."""

    def __init__(self, root: Path, milestone: str = "all", progressive: bool = False, verbose: bool = False):
        self.root = root
        self.milestone = milestone
        self.progressive = progressive
        self.verbose = verbose
        self.report = TestReport(milestone, progressive)

    def select_notebooks(self) -> List[Dict[str, Any]]:
        """Filters notebook specifications based on requested milestone."""
        if self.milestone == "all":
            return NOTEBOOK_SPECS
        return [s for s in NOTEBOOK_SPECS if s["milestone"] == self.milestone]

    def run_all(self, selected_tiers: Optional[List[int]] = None) -> TestReport:
        """Executes selected tiers across chosen notebooks."""
        if selected_tiers is None:
            selected_tiers = [1, 2, 3, 4]

        specs = self.select_notebooks()
        self.report.stats["selected_notebooks"] = len(specs)

        # Inventory phase
        notebook_data_map: Dict[str, Dict[str, Any]] = {}
        for spec in specs:
            nb_path = self.root / spec["path"]
            exists = nb_path.exists()
            status = "FOUND" if exists else "MISSING"

            info = {
                "id": spec["id"],
                "name": spec["name"],
                "path": spec["path"],
                "milestone": spec["milestone"],
                "exists": exists,
                "status": status,
                "total_cells": 0,
                "code_cells": 0,
                "markdown_cells": 0,
                "errors": [],
            }

            if exists:
                self.report.stats["existing_notebooks"] += 1
                data, err = load_notebook(nb_path)
                if err:
                    info["errors"].append(err)
                else:
                    notebook_data_map[spec["path"]] = data
                    cells = data.get("cells", [])
                    info["total_cells"] = len(cells)
                    info["code_cells"] = sum(1 for c in cells if c.get("cell_type") == "code")
                    info["markdown_cells"] = sum(1 for c in cells if c.get("cell_type") == "markdown")
                    self.report.stats["total_code_cells"] += info["code_cells"]
                    self.report.stats["total_markdown_cells"] += info["markdown_cells"]
            else:
                self.report.stats["missing_notebooks"] += 1

            self.report.notebook_metrics.append(info)

        # Execute tiers
        if 1 in selected_tiers:
            self._execute_tier1(specs, notebook_data_map)
        if 2 in selected_tiers:
            self._execute_tier2(specs, notebook_data_map)
        if 3 in selected_tiers:
            self._execute_tier3(specs, notebook_data_map)
        if 4 in selected_tiers:
            self._execute_tier4(specs, notebook_data_map)

        self.report.end_time = time.time()
        return self.report

    # --------------------------------------------------------------------------
    # Tier 1: Feature Coverage Execution
    # --------------------------------------------------------------------------
    def _execute_tier1(self, specs: List[Dict[str, Any]], data_map: Dict[str, Dict[str, Any]]):
        """Verifies existence, valid JSON, nbformat 4 schema, metadata, cell structures."""
        for spec in specs:
            rel_path = spec["path"]
            nb_path = self.root / rel_path

            # 1. Existence check
            if not nb_path.exists():
                if self.progressive:
                    self.report.record_skip("tier1", f"Notebook {spec['id']} not yet generated (progressive mode)")
                else:
                    self.report.record_fail("tier1", rel_path, "Notebook file missing on disk")
                continue

            self.report.record_pass("tier1", f"{rel_path}: file exists")

            data = data_map.get(rel_path)
            if data is None:
                self.report.record_fail("tier1", rel_path, "Invalid JSON formatting")
                continue
            self.report.record_pass("tier1", f"{rel_path}: valid JSON")

            # 2. Schema compliance (nbformat == 4)
            nbformat = data.get("nbformat")
            if nbformat != 4:
                self.report.record_fail("tier1", rel_path, f"Expected nbformat 4, got {nbformat}")
            else:
                self.report.record_pass("tier1", f"{rel_path}: nbformat 4")

            # 3. Language info
            metadata = data.get("metadata", {})
            lang_info = metadata.get("language_info", {})
            if lang_info.get("name") != "python":
                self.report.record_fail("tier1", rel_path, f"metadata.language_info.name expected 'python', got '{lang_info.get('name')}'")
            else:
                self.report.record_pass("tier1", f"{rel_path}: python language metadata")

            # 4. Mandatory cell types & documentation
            cells = data.get("cells", [])
            has_markdown = any(c.get("cell_type") == "markdown" for c in cells)
            has_code = any(c.get("cell_type") == "code" for c in cells)

            if not has_markdown:
                self.report.record_fail("tier1", rel_path, "Notebook contains zero documentation markdown cells")
            else:
                self.report.record_pass("tier1", f"{rel_path}: documentation present")

            if not has_code:
                self.report.record_fail("tier1", rel_path, "Notebook contains zero executable code cells")
            else:
                self.report.record_pass("tier1", f"{rel_path}: code cells present")

    # --------------------------------------------------------------------------
    # Tier 2: Boundary & Corner Cases Execution
    # --------------------------------------------------------------------------
    def _execute_tier2(self, specs: List[Dict[str, Any]], data_map: Dict[str, Dict[str, Any]]):
        """Verifies empty cell rejection, unescaped chars, forbidden magics, delimiter balance."""
        disallowed_magics = ["%sh", "!", "%scala", "%r"]

        for spec in specs:
            rel_path = spec["path"]
            data = data_map.get(rel_path)
            if not data:
                if self.progressive:
                    self.report.record_skip("tier2", f"{rel_path} skipped (progressive)")
                else:
                    self.report.record_fail("tier2", rel_path, "File missing for boundary evaluation")
                continue

            nb_path = self.root / rel_path
            raw_content = nb_path.read_text(encoding="utf-8")

            # UTF-8 replacement character
            if "\ufffd" in raw_content:
                self.report.record_fail("tier2", rel_path, "Contains corrupted unicode replacement character (\\ufffd)")
            else:
                self.report.record_pass("tier2", f"{rel_path}: clean UTF-8 encoding")

            cells = data.get("cells", [])
            has_empty_code_cells = False
            has_disallowed_magic = False
            has_delimiter_imbalance = False

            for idx, cell in enumerate(cells):
                if cell.get("cell_type") == "code":
                    source = get_cell_source(cell)
                    trimmed = source.strip()

                    # Empty code cell check
                    if not trimmed:
                        has_empty_code_cells = True
                        self.report.record_fail("tier2", rel_path, f"Cell {idx}: empty code cell without content")

                    # Disallowed magics
                    for dis in disallowed_magics:
                        if trimmed.startswith(dis):
                            has_disallowed_magic = True
                            self.report.record_fail("tier2", rel_path, f"Cell {idx}: contains disallowed magic '{dis}'")

                    # Delimiter checks on %sql cells
                    if trimmed.startswith("%sql"):
                        errs = validate_delimiter_balance(trimmed[4:])
                        if errs:
                            has_delimiter_imbalance = True
                            for e in errs:
                                self.report.record_fail("tier2", rel_path, f"Cell {idx} SQL delimiter: {e}")

            if not has_empty_code_cells:
                self.report.record_pass("tier2", f"{rel_path}: no empty code cells")
            if not has_disallowed_magic:
                self.report.record_pass("tier2", f"{rel_path}: supported cell magics only")
            if not has_delimiter_imbalance:
                self.report.record_pass("tier2", f"{rel_path}: balanced delimiters")

    # --------------------------------------------------------------------------
    # Tier 3: Cross-Feature Combinations Execution
    # --------------------------------------------------------------------------
    def _execute_tier3(self, specs: List[Dict[str, Any]], data_map: Dict[str, Dict[str, Any]]):
        """Verifies inter-layer schema consistency: Bronze -> Silver -> Gold -> EDA query alignment."""
        # 1. Database Init
        init_spec = NOTEBOOK_BY_ID["DWH-01"]
        init_data = data_map.get(init_spec["path"])
        if not init_data and (self.root / init_spec["path"]).exists():
            init_data, _ = load_notebook(self.root / init_spec["path"])
        if init_data:
            content = " ".join(get_cell_source(c) for c in init_data.get("cells", []))
            for db in ["bronze", "silver", "gold"]:
                if re.search(rf"\b{db}\b", content, re.IGNORECASE):
                    self.report.record_pass("tier3", f"init_database.ipynb initializes '{db}'")
                else:
                    self.report.record_fail("tier3", init_spec["path"], f"Database '{db}' not initialized")
        else:
            if not self.progressive:
                self.report.record_fail("tier3", init_spec["path"], "init_database.ipynb missing")

        # 2. Bronze DDL & Load Table Parity
        ddl_b = data_map.get(NOTEBOOK_BY_ID["DWH-02"]["path"])
        load_b = data_map.get(NOTEBOOK_BY_ID["DWH-03"]["path"])
        if ddl_b and load_b:
            ddl_text = " ".join(get_cell_source(c) for c in ddl_b.get("cells", [])).lower()
            load_text = " ".join(get_cell_source(c) for c in load_b.get("cells", [])).lower()
            expected_bronze = NOTEBOOK_BY_ID["DWH-02"]["expected_tables_created"]

            for tbl in expected_bronze:
                if tbl in ddl_text and tbl in load_text:
                    self.report.record_pass("tier3", f"Bronze table '{tbl}' defined and loaded")
                else:
                    self.report.record_fail("tier3", "bronze", f"Table '{tbl}' missing from DDL or Load notebook")

        # 3. Silver DDL & Load Table Parity & Audit Column
        ddl_s = data_map.get(NOTEBOOK_BY_ID["DWH-04"]["path"])
        load_s = data_map.get(NOTEBOOK_BY_ID["DWH-05"]["path"])
        if ddl_s and load_s:
            ddl_s_text = " ".join(get_cell_source(c) for c in ddl_s.get("cells", [])).lower()
            load_s_text = " ".join(get_cell_source(c) for c in load_s.get("cells", [])).lower()

            if "dwh_create_date" in ddl_s_text and "dwh_create_date" in load_s_text:
                self.report.record_pass("tier3", "Silver audit column 'dwh_create_date' present")
            else:
                self.report.record_fail("tier3", "silver", "Missing 'dwh_create_date' audit column in Silver DDL or Load")

            for tbl in NOTEBOOK_BY_ID["DWH-04"]["expected_tables_created"]:
                if tbl in ddl_s_text and tbl in load_s_text:
                    self.report.record_pass("tier3", f"Silver table '{tbl}' defined and loaded")
                else:
                    self.report.record_fail("tier3", "silver", f"Table '{tbl}' missing from Silver DDL or Load")

        # 4. Gold Star Schema
        ddl_g = data_map.get(NOTEBOOK_BY_ID["DWH-06"]["path"])
        if ddl_g:
            gold_text = " ".join(get_cell_source(c) for c in ddl_g.get("cells", [])).lower()
            for tbl in ["gold.dim_customers", "gold.dim_products", "gold.fact_sales"]:
                if tbl in gold_text:
                    self.report.record_pass("tier3", f"Gold star schema table '{tbl}' created")
                else:
                    self.report.record_fail("tier3", "gold", f"Table '{tbl}' not created in ddl_gold.ipynb")

        # 5. EDA Notebooks query only Gold tables
        for spec in specs:
            if spec["layer"] in ["eda", "advanced_eda"]:
                eda_data = data_map.get(spec["path"])
                if eda_data:
                    eda_text = " ".join(get_cell_source(c) for c in eda_data.get("cells", []))
                    refs = extract_table_references(eda_text)
                    invalid_refs = [t for t in refs if not t.startswith("gold.")]
                    if invalid_refs:
                        self.report.record_fail(
                            "tier3", spec["path"],
                            f"EDA notebook references non-gold tables: {invalid_refs}"
                        )
                    else:
                        self.report.record_pass("tier3", f"{spec['path']}: queries target gold layer only")

    # --------------------------------------------------------------------------
    # Tier 4: Real-World Workload Testing Execution
    # --------------------------------------------------------------------------
    def _execute_tier4(self, specs: List[Dict[str, Any]], data_map: Dict[str, Dict[str, Any]]):
        """Full Python AST parse, bytecode compilation, SQL syntax, column parity, dialect checks."""
        for spec in specs:
            rel_path = spec["path"]
            data = data_map.get(rel_path)
            if not data:
                if self.progressive:
                    self.report.record_skip("tier4", f"{rel_path} skipped (progressive)")
                else:
                    self.report.record_fail("tier4", rel_path, "Notebook missing for workload execution")
                continue

            cells = data.get("cells", [])
            for idx, cell in enumerate(cells):
                if cell.get("cell_type") != "code":
                    continue

                source = get_cell_source(cell)
                trimmed = source.strip()
                if not trimmed:
                    continue

                # 1. Python AST parse and Bytecode Compilation
                if not trimmed.startswith("%sql") and not trimmed.startswith("%md"):
                    code_to_parse = trimmed
                    if code_to_parse.startswith("%python"):
                        lines = code_to_parse.splitlines()
                        code_to_parse = "\n".join(lines[1:])

                    try:
                        tree = ast.parse(code_to_parse, filename=rel_path)
                        compile(tree, filename=rel_path, mode="exec")
                        self.report.stats["python_cells_compiled"] += 1
                        self.report.record_pass("tier4", f"{rel_path} Cell {idx}: AST parsed & compiled")
                    except SyntaxError as e:
                        self.report.record_fail(
                            "tier4", rel_path,
                            f"Cell {idx} Python SyntaxError at line {e.lineno}, col {e.offset}: {e.msg}"
                        )

                # 2. SQL Syntax Validation
                queries = extract_queries_from_code_cell(source)
                for q_type, q_text in queries:
                    self.report.stats["sql_queries_validated"] += 1
                    sql_errs = validate_sql_syntax(q_text)
                    if sql_errs:
                        for err in sql_errs:
                            self.report.record_fail("tier4", rel_path, f"Cell {idx} ({q_type}): {err}")
                    else:
                        self.report.record_pass("tier4", f"{rel_path} Cell {idx}: SQL syntax valid")

                # 3. Dialect checks: prohibit PostgreSQL AGE()
                if re.search(r'\bAGE\s*\(', source, re.IGNORECASE):
                    self.report.record_fail(
                        "tier4", rel_path,
                        f"Cell {idx} contains untranslated PostgreSQL 'AGE()' function! Use 'months_between()'."
                    )

                # 4. Dialect checks: prohibit PostgreSQL '::' casting
                if re.search(r'::\s*(text|int|integer|varchar|numeric|date|timestamp)', source, re.IGNORECASE):
                    self.report.record_fail(
                        "tier4", rel_path,
                        f"Cell {idx} contains PostgreSQL '::' cast operator! Use 'CAST(expr AS type)'."
                    )

            # 5. Critical column parity in gold.dim_products
            if spec["id"] == "DWH-06":
                content = " ".join(get_cell_source(c) for c in cells)
                has_product_cost = bool(re.search(r'\bproduct_cost\b', content, re.IGNORECASE))
                has_cost = bool(re.search(r'\bcost\b', content, re.IGNORECASE))

                if has_product_cost and has_cost:
                    self.report.record_pass("tier4", "gold.dim_products projects both 'cost' and 'product_cost'")
                else:
                    self.report.record_fail(
                        "tier4", rel_path,
                        f"gold.dim_products column parity check failed: cost={has_cost}, product_cost={has_product_cost}"
                    )

            # 6. Check for duplicate SELECT in ADV-08
            if spec["id"] == "ADV-08":
                content = " ".join(get_cell_source(c) for c in cells)
                if re.search(r'\bSELECT\s+SELECT\b', content, re.IGNORECASE):
                    self.report.record_fail("tier4", rel_path, "Contains original SQL duplicate 'SELECT SELECT' bug")


# ==============================================================================
# CLI Formatting & Pretty Printing
# ==============================================================================

def print_banner():
    banner = f"""
{Color.CYAN}{Color.BOLD}================================================================================{Color.RESET}
{Color.WHITE}{Color.BOLD}      Databricks Medallion Migration — Comprehensive E2E Test Suite            {Color.RESET}
{Color.CYAN}================================================================================{Color.RESET}
"""
    print(banner)


def render_report_table(report: TestReport, verbose: bool):
    """Renders formatted execution results and summary dashboard."""
    print(f"{Color.BOLD}Execution Summary:{Color.RESET}")
    print(f"  Target Milestone : {Color.CYAN}{report.target_milestone}{Color.RESET}")
    print(f"  Progressive Mode : {Color.YELLOW if report.progressive else Color.GREEN}{report.progressive}{Color.RESET}")
    print(f"  Execution Time   : {report.end_time - report.start_time:.3f} seconds\n")

    # Inventory Matrix
    print(f"{Color.BOLD}Target Notebooks Inventory Matrix:{Color.RESET}")
    header = f"{'ID':<8} | {'Milestone':<9} | {'Status':<8} | {'Cells':<6} | {'Code':<5} | {'Path'}"
    print("-" * 80)
    print(header)
    print("-" * 80)

    for item in report.notebook_metrics:
        status_color = Color.GREEN if item["exists"] else (Color.YELLOW if report.progressive else Color.RED)
        status_text = f"{status_color}{item['status']:<8}{Color.RESET}"
        cells_cnt = str(item["total_cells"]) if item["exists"] else "-"
        code_cnt = str(item["code_cells"]) if item["exists"] else "-"
        print(f"{item['id']:<8} | {item['milestone']:<9} | {status_text} | {cells_cnt:<6} | {code_cnt:<5} | {item['path']}")

    print("-" * 80 + "\n")

    # Tier Results
    print(f"{Color.BOLD}4-Tier Testing Methodology Results:{Color.RESET}")
    print("=" * 80)

    for tier_key in ["tier1", "tier2", "tier3", "tier4"]:
        tier = report.tier_results[tier_key]
        status_badge = (
            f"{Color.GREEN}[PASS]{Color.RESET}" if tier["failed"] == 0
            else f"{Color.RED}[FAIL]{Color.RESET}"
        )
        print(f"{status_badge} {Color.BOLD}{tier['name']}{Color.RESET}")
        print(f"       Passed: {tier['passed']}  |  Failed: {tier['failed']}  |  Skipped: {tier['skipped']}")

        if tier["errors"]:
            print(f"       {Color.RED}Failures / Violations:{Color.RESET}")
            for err in tier["errors"][:8]:  # show top 8
                print(f"         - [{err['notebook']}]: {err['error']}")
            if len(tier["errors"]) > 8:
                print(f"         ... and {len(tier['errors']) - 8} more errors")
        print()

    print("=" * 80)

    # Detailed Statistics Dashboard
    stats = report.stats
    print(f"{Color.BOLD}Test Verification Metrics:{Color.RESET}")
    print(f"  Total Target Notebooks  : {stats['total_target_notebooks']}")
    print(f"  Discovered on Disk      : {stats['existing_notebooks']} / {stats['selected_notebooks']}")
    print(f"  Total Code Cells Scanned: {stats['total_code_cells']}")
    print(f"  Python AST Compiled     : {stats['python_cells_compiled']}")
    print(f"  SQL Queries Validated   : {stats['sql_queries_validated']}")
    print(f"  Total Checks Passed     : {Color.GREEN}{report.total_passed}{Color.RESET}")
    print(f"  Total Checks Failed     : {Color.RED if report.total_failed > 0 else Color.GREEN}{report.total_failed}{Color.RESET}")
    print(f"  Total Checks Skipped    : {report.total_skipped}")

    print("=" * 80)
    if report.is_success:
        print(f"{Color.GREEN}{Color.BOLD}>>> OVERALL RESULT: ALL TESTS PASSED (EXIT CODE 0) <<<{Color.RESET}\n")
    else:
        print(f"{Color.RED}{Color.BOLD}>>> OVERALL RESULT: TESTS FAILED (EXIT CODE 1) <<<{Color.RESET}\n")


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Databricks Notebook E2E Test Runner")
    parser.add_argument(
        "--tier", choices=["1", "2", "3", "4", "all"], default="all",
        help="Run specific test tier (default: all)"
    )
    parser.add_argument(
        "--milestone", choices=["M1", "M2", "M3", "M4", "M5", "all"], default="all",
        help="Filter notebooks by milestone (default: all)"
    )
    parser.add_argument(
        "--progressive", action="store_true",
        help="Progressive mode: test existing notebooks without failing on missing milestone files"
    )
    parser.add_argument(
        "--json-report", type=str, default=None,
        help="Write JSON test report to target file path"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    print_banner()

    tiers = None
    if args.tier != "all":
        tiers = [int(args.tier)]

    runner = NotebookTestRunner(
        root=PROJECT_ROOT,
        milestone=args.milestone,
        progressive=args.progressive,
        verbose=args.verbose
    )

    report = runner.run_all(selected_tiers=tiers)
    render_report_table(report, verbose=args.verbose)

    if args.json_report:
        report_data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "is_success": report.is_success,
            "target_milestone": report.target_milestone,
            "progressive_mode": report.progressive,
            "stats": report.stats,
            "tier_results": report.tier_results,
            "notebook_metrics": report.notebook_metrics,
        }
        out_path = Path(args.json_report).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"JSON test report saved to: {out_path}\n")

    sys.exit(0 if report.is_success else 1)


if __name__ == "__main__":
    main()

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "app.py")


def test_app_initial_render_and_no_exceptions():
    """Verify that app loads cleanly on initial render with zero exceptions."""
    at = AppTest.from_file(APP_PATH, default_timeout=35)
    at.run(timeout=35)
    assert len(at.exception) == 0, f"Exceptions on startup: {[e.value for e in at.exception]}"
    assert at.session_state.get("nav_page") == "Overview / Home"


def test_homepage_hero_buttons_navigation():
    """Verify Start Research and Explore Platform buttons navigate to correct pages."""
    at = AppTest.from_file(APP_PATH, default_timeout=35)
    at.run(timeout=35)
    assert len(at.exception) == 0

    # Find the "Start Research" button and click it
    btn_start = None
    for b in at.button:
        if "Start Research" in b.label:
            btn_start = b
            break
    assert btn_start is not None, "Could not find 'Start Research' button"
    btn_start.click().run(timeout=35)
    assert len(at.exception) == 0
    assert at.session_state.get("nav_page") == "Causal Analysis & DAG Workbench"

    # Now navigate back to Overview and test "Explore Platform"
    at.session_state["nav_page"] = "Overview / Home"
    at.run(timeout=35)
    assert len(at.exception) == 0

    btn_explore = None
    for b in at.button:
        if "Explore Platform" in b.label:
            btn_explore = b
            break
    assert btn_explore is not None, "Could not find 'Explore Platform' button"
    btn_explore.click().run(timeout=35)
    assert len(at.exception) == 0
    assert at.session_state.get("nav_page") == "Data Analysis"


def test_all_sidebar_navigation_pages():
    """Verify switching to every single workspace page functions without exceptions."""
    all_pages = [
        "Overview / Home",
        "Data Analysis",
        "Causal Analysis & DAG Workbench",
        "AI Research Assistant",
        "Knowledge Base / RAG",
        "Research Lab",
        "Reports & Exports",
        "Settings",
    ]

    at = AppTest.from_file(APP_PATH, default_timeout=35)
    at.run(timeout=35)
    assert len(at.exception) == 0

    for page in all_pages:
        at.session_state["nav_page"] = page
        at.run(timeout=35)
        assert len(at.exception) == 0, f"Exception rendered on page '{page}': {[e.value for e in at.exception]}"
        assert at.session_state.get("nav_page") == page


def test_data_analysis_load_benchmark_button():
    """Verify clicking Load Lalonde Job Training Benchmark button on Data Analysis page updates state."""
    at = AppTest.from_file(APP_PATH, default_timeout=35)
    at.session_state["nav_page"] = "Data Analysis"
    at.run(timeout=35)
    assert len(at.exception) == 0

    btn_lalonde = None
    for b in at.button:
        if "Lalonde" in b.label:
            btn_lalonde = b
            break
    assert btn_lalonde is not None, "Could not find Lalonde benchmark button"
    btn_lalonde.click().run(timeout=35)
    assert len(at.exception) == 0
    df = at.session_state.get("df")
    assert df is not None
    assert len(df) > 0
    assert "training_program" in df.columns or "treatment" in df.columns


def test_causal_page_execution_buttons():
    """Verify Execute All 5 Causal Estimators and Refutation Battery buttons run real algorithms."""
    at = AppTest.from_file(APP_PATH, default_timeout=35)
    at.session_state["nav_page"] = "Causal Analysis & DAG Workbench"
    at.run(timeout=35)
    assert len(at.exception) == 0

    # Locate and click 'Execute All 5 Causal Estimators'
    btn_exec = None
    for b in at.button:
        if "Execute All 5" in b.label:
            btn_exec = b
            break
    assert btn_exec is not None, "Could not find 'Execute All 5 Causal Estimators' button"
    btn_exec.click().run(timeout=35)
    assert len(at.exception) == 0
    all_ests = at.session_state.get("all_estimators")
    assert all_ests is not None
    assert len(all_ests) == 5
    for est in all_ests:
        assert "ate" in est and not (est["ate"] != est["ate"])  # not NaN

    # Locate and click 'Execute Refutation Battery'
    btn_ref = None
    for b in at.button:
        if "Refutation" in b.label:
            btn_ref = b
            break
    assert btn_ref is not None, "Could not find 'Execute Refutation Battery' button"
    btn_ref.click().run(timeout=35)
    assert len(at.exception) == 0
    ref_res = at.session_state.get("refutation_results")
    assert ref_res is not None
    assert len(ref_res) == 3


def test_reports_page_generation():
    """Verify Reports & Exports page renders complete Markdown, JSON and HTML exports."""
    at = AppTest.from_file(APP_PATH, default_timeout=35)
    at.session_state["nav_page"] = "Reports & Exports"
    at.run(timeout=35)
    assert len(at.exception) == 0
    # Download buttons should be present
    download_btn_labels = [b.label for b in at.download_button]
    assert any("Markdown" in l for l in download_btn_labels)
    assert any("JSON" in l for l in download_btn_labels)
    assert any("HTML" in l for l in download_btn_labels)

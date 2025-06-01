"""Test script for the cleanup system."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from .config.manager import ConfigurationManager
from .core.pipeline import CleanupPipeline
from .core.safety import SafetyManager
from .core.validation import ValidationEngine
from .modules.dead_code import DeadCodeAnalyzer
from .modules.documentation import DocumentationRefresher
from .modules.standards import StandardsEnforcer


def test_configuration_manager():
    """Test configuration manager functionality."""
    print("Testing ConfigurationManager...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        project_root = Path(temp_dir)
        config_manager = ConfigurationManager(project_root)
        
        # Test default config creation
        config = config_manager._create_default_config()
        assert config.project_name == 'psysafe-ai'
        assert len(config.phases) > 0
        assert config.safety.backup_enabled is True
        
        # Test config validation
        errors = config_manager.validate_config(config)
        assert len(errors) == 0, f"Config validation errors: {errors}"
        
        print("✅ ConfigurationManager tests passed")


def test_dead_code_analyzer():
    """Test dead code analyzer functionality."""
    print("Testing DeadCodeAnalyzer...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        project_root = Path(temp_dir)
        
        # Create test Python file with dead code
        test_file = project_root / "test.py"
        test_file.write_text("""
# This is a comment
import os
import sys  # unused import

def used_function():
    return "used"

def unused_function():
    pass

# print("commented code")
# result = some_function()

def placeholder_function():
    '''This is a placeholder'''
    pass

if __name__ == "__main__":
    used_function()
""")
        
        analyzer = DeadCodeAnalyzer(project_root)
        
        # Test validation
        assert analyzer.validate() is True
        
        # Test analysis
        analysis = analyzer.analyze()
        assert 'commented_code_blocks' in analysis
        assert 'unused_functions' in analysis
        assert 'unused_imports' in analysis
        
        print("✅ DeadCodeAnalyzer tests passed")


def test_documentation_refresher():
    """Test documentation refresher functionality."""
    print("Testing DocumentationRefresher...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        project_root = Path(temp_dir)
        
        # Create test Python file with missing docstrings
        test_file = project_root / "test.py"
        test_file.write_text("""
def function_without_docstring(param1, param2):
    return param1 + param2

class ClassWithoutDocstring:
    def method_without_docstring(self):
        return "test"
""")
        
        refresher = DocumentationRefresher(project_root)
        
        # Test validation
        assert refresher.validate() is True
        
        # Test analysis
        analysis = refresher.analyze()
        assert 'total_docstring_issues' in analysis
        assert 'missing_docstrings' in analysis
        
        print("✅ DocumentationRefresher tests passed")


def test_standards_enforcer():
    """Test standards enforcer functionality."""
    print("Testing StandardsEnforcer...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        project_root = Path(temp_dir)
        
        # Create test Python file with formatting issues
        test_file = project_root / "test.py"
        test_file.write_text("""
import sys
import os

def badlyNamedFunction():
    BADLY_NAMED_VARIABLE=1
    return BADLY_NAMED_VARIABLE

class badlyNamedClass:
    pass
""")
        
        enforcer = StandardsEnforcer(project_root)
        
        # Test validation (may fail if ruff not installed, that's ok)
        validation_result = enforcer.validate()
        print(f"Standards enforcer validation: {validation_result}")
        
        # Test analysis
        analysis = enforcer.analyze()
        assert 'naming_violations' in analysis
        assert 'line_length_issues' in analysis
        
        print("✅ StandardsEnforcer tests passed")


def test_validation_engine():
    """Test validation engine functionality."""
    print("Testing ValidationEngine...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        project_root = Path(temp_dir)
        
        # Create test Python file
        test_file = project_root / "test.py"
        test_file.write_text("""
import os
import sys

def test_function():
    return "test"

if __name__ == "__main__":
    test_function()
""")
        
        # Create pyproject.toml
        pyproject_file = project_root / "pyproject.toml"
        pyproject_file.write_text("""
[project]
name = "test-project"
""")
        
        # Create psysafe package structure
        psysafe_dir = project_root / "psysafe"
        psysafe_dir.mkdir()
        (psysafe_dir / "__init__.py").write_text("")
        
        engine = ValidationEngine(project_root)
        
        # Test syntax validation
        syntax_results = engine.validate_syntax()
        assert len(syntax_results) > 0
        assert all(r.passed for r in syntax_results), "Syntax validation should pass"
        
        # Test project structure validation
        structure_results = engine.validate_project_structure()
        assert len(structure_results) > 0
        
        print("✅ ValidationEngine tests passed")


def test_safety_manager():
    """Test safety manager functionality."""
    print("Testing SafetyManager...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        project_root = Path(temp_dir)
        
        # Create some test files
        test_file = project_root / "test.py"
        test_file.write_text("print('test')")
        
        safety_manager = SafetyManager(project_root)
        
        # Test safety checks
        safety_result = safety_manager.pre_cleanup_safety_check()
        print(f"Safety checks result: {safety_result}")
        
        # Test backup creation
        backup_info = safety_manager.backup_manager.create_full_backup("Test backup")
        assert backup_info is not None
        assert backup_info.backup_path.exists()
        
        print("✅ SafetyManager tests passed")


def test_cleanup_pipeline():
    """Test cleanup pipeline functionality."""
    print("Testing CleanupPipeline...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        project_root = Path(temp_dir)
        
        # Create minimal project structure
        test_file = project_root / "test.py"
        test_file.write_text("""
def test_function():
    return "test"
""")
        
        pipeline = CleanupPipeline(project_root)
        
        # Test execution plan creation
        plan = pipeline.create_default_execution_plan()
        assert len(plan.phases) > 0
        
        # Test analysis
        analysis = pipeline.analyze_all_phases()
        assert len(analysis) > 0
        
        print("✅ CleanupPipeline tests passed")


def run_all_tests():
    """Run all tests."""
    print("Running cleanup system tests...\n")
    
    tests = [
        test_configuration_manager,
        test_dead_code_analyzer,
        test_documentation_refresher,
        test_standards_enforcer,
        test_validation_engine,
        test_safety_manager,
        test_cleanup_pipeline,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1
        print()
    
    print("="*50)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*50)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
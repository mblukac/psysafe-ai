"""Validation engine for cleanup operations."""

import ast
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .base import CleanupReport


@dataclass
class ValidationResult:
    """Result of a validation check."""
    
    check_name: str
    passed: bool
    message: str
    severity: str  # 'info', 'warning', 'error', 'critical'
    details: Dict[str, Any]
    timestamp: datetime


class ValidationEngine:
    """Comprehensive validation engine for cleanup operations."""
    
    def __init__(self, project_root: Path, config: Optional[Dict[str, Any]] = None):
        self.project_root = project_root
        self.config = config or {}
        self.validation_results: List[ValidationResult] = []
    
    def validate_syntax(self) -> List[ValidationResult]:
        """Validate Python syntax for all files."""
        results = []
        
        for file_path in self._get_python_files():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                ast.parse(content, filename=str(file_path))
                
                results.append(ValidationResult(
                    check_name='syntax_check',
                    passed=True,
                    message=f"Syntax valid: {file_path.relative_to(self.project_root)}",
                    severity='info',
                    details={'file': str(file_path)},
                    timestamp=datetime.now()
                ))
            
            except SyntaxError as e:
                results.append(ValidationResult(
                    check_name='syntax_check',
                    passed=False,
                    message=f"Syntax error in {file_path.relative_to(self.project_root)}: {e.msg}",
                    severity='error',
                    details={
                        'file': str(file_path),
                        'line': e.lineno,
                        'column': e.offset,
                        'error': str(e)
                    },
                    timestamp=datetime.now()
                ))
            
            except Exception as e:
                results.append(ValidationResult(
                    check_name='syntax_check',
                    passed=False,
                    message=f"Error parsing {file_path.relative_to(self.project_root)}: {str(e)}",
                    severity='error',
                    details={'file': str(file_path), 'error': str(e)},
                    timestamp=datetime.now()
                ))
        
        return results
    
    def validate_imports(self) -> List[ValidationResult]:
        """Validate that all imports can be resolved."""
        results = []
        
        for file_path in self._get_python_files():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content, filename=str(file_path))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            result = self._validate_import(alias.name, file_path)
                            if result:
                                results.append(result)
                    
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            result = self._validate_import(node.module, file_path)
                            if result:
                                results.append(result)
            
            except Exception as e:
                results.append(ValidationResult(
                    check_name='import_validation',
                    passed=False,
                    message=f"Error validating imports in {file_path.relative_to(self.project_root)}: {str(e)}",
                    severity='warning',
                    details={'file': str(file_path), 'error': str(e)},
                    timestamp=datetime.now()
                ))
        
        return results
    
    def _validate_import(self, module_name: str, file_path: Path) -> Optional[ValidationResult]:
        """Validate a single import."""
        try:
            # Try to import the module
            __import__(module_name)
            return None  # Import successful, no result needed
        
        except ImportError as e:
            # Check if it's a local import
            if module_name.startswith('psysafe') or module_name.startswith('.'):
                # Local import - check if file exists
                if self._check_local_import(module_name, file_path):
                    return None
            
            return ValidationResult(
                check_name='import_validation',
                passed=False,
                message=f"Import error in {file_path.relative_to(self.project_root)}: {module_name}",
                severity='warning',
                details={
                    'file': str(file_path),
                    'module': module_name,
                    'error': str(e)
                },
                timestamp=datetime.now()
            )
        
        except Exception as e:
            return ValidationResult(
                check_name='import_validation',
                passed=False,
                message=f"Unexpected error importing {module_name}: {str(e)}",
                severity='warning',
                details={
                    'file': str(file_path),
                    'module': module_name,
                    'error': str(e)
                },
                timestamp=datetime.now()
            )
    
    def _check_local_import(self, module_name: str, file_path: Path) -> bool:
        """Check if a local import exists."""
        # This is a simplified check
        # In production, implement proper module resolution
        if module_name.startswith('psysafe'):
            parts = module_name.split('.')
            check_path = self.project_root
            for part in parts:
                check_path = check_path / part
            
            return (check_path.with_suffix('.py').exists() or 
                   (check_path / '__init__.py').exists())
        
        return True  # Assume relative imports are valid
    
    def validate_tests(self) -> List[ValidationResult]:
        """Validate that tests can run."""
        results = []
        
        # Check if pytest is available
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pytest', '--version'],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                results.append(ValidationResult(
                    check_name='test_framework',
                    passed=True,
                    message="pytest is available",
                    severity='info',
                    details={'version': result.stdout.strip()},
                    timestamp=datetime.now()
                ))
            else:
                results.append(ValidationResult(
                    check_name='test_framework',
                    passed=False,
                    message="pytest is not available",
                    severity='warning',
                    details={'error': result.stderr},
                    timestamp=datetime.now()
                ))
        
        except Exception as e:
            results.append(ValidationResult(
                check_name='test_framework',
                passed=False,
                message=f"Error checking pytest: {str(e)}",
                severity='warning',
                details={'error': str(e)},
                timestamp=datetime.now()
            ))
        
        # Run a quick test to check basic functionality
        test_dirs = [self.project_root / 'tests', self.project_root / 'test']
        
        for test_dir in test_dirs:
            if test_dir.exists():
                try:
                    result = subprocess.run(
                        [sys.executable, '-m', 'pytest', str(test_dir), '--collect-only', '-q'],
                        capture_output=True,
                        text=True,
                        cwd=self.project_root,
                        timeout=30
                    )
                    
                    if result.returncode == 0:
                        results.append(ValidationResult(
                            check_name='test_collection',
                            passed=True,
                            message=f"Tests can be collected from {test_dir.name}",
                            severity='info',
                            details={'test_dir': str(test_dir)},
                            timestamp=datetime.now()
                        ))
                    else:
                        results.append(ValidationResult(
                            check_name='test_collection',
                            passed=False,
                            message=f"Test collection failed in {test_dir.name}",
                            severity='warning',
                            details={
                                'test_dir': str(test_dir),
                                'error': result.stderr
                            },
                            timestamp=datetime.now()
                        ))
                
                except subprocess.TimeoutExpired:
                    results.append(ValidationResult(
                        check_name='test_collection',
                        passed=False,
                        message=f"Test collection timed out in {test_dir.name}",
                        severity='warning',
                        details={'test_dir': str(test_dir)},
                        timestamp=datetime.now()
                    ))
                
                except Exception as e:
                    results.append(ValidationResult(
                        check_name='test_collection',
                        passed=False,
                        message=f"Error collecting tests from {test_dir.name}: {str(e)}",
                        severity='warning',
                        details={'test_dir': str(test_dir), 'error': str(e)},
                        timestamp=datetime.now()
                    ))
        
        return results
    
    def validate_examples(self) -> List[ValidationResult]:
        """Validate that examples can run."""
        results = []
        
        examples_dir = self.project_root / 'examples'
        if not examples_dir.exists():
            results.append(ValidationResult(
                check_name='examples_validation',
                passed=True,
                message="No examples directory found",
                severity='info',
                details={},
                timestamp=datetime.now()
            ))
            return results
        
        for example_file in examples_dir.glob('*.py'):
            try:
                # Try to parse the example
                with open(example_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                ast.parse(content, filename=str(example_file))
                
                results.append(ValidationResult(
                    check_name='example_syntax',
                    passed=True,
                    message=f"Example syntax valid: {example_file.name}",
                    severity='info',
                    details={'file': str(example_file)},
                    timestamp=datetime.now()
                ))
                
                # Check for basic structure
                if 'import' in content and ('psysafe' in content or 'from psysafe' in content):
                    results.append(ValidationResult(
                        check_name='example_structure',
                        passed=True,
                        message=f"Example has proper imports: {example_file.name}",
                        severity='info',
                        details={'file': str(example_file)},
                        timestamp=datetime.now()
                    ))
                else:
                    results.append(ValidationResult(
                        check_name='example_structure',
                        passed=False,
                        message=f"Example missing psysafe imports: {example_file.name}",
                        severity='warning',
                        details={'file': str(example_file)},
                        timestamp=datetime.now()
                    ))
            
            except SyntaxError as e:
                results.append(ValidationResult(
                    check_name='example_syntax',
                    passed=False,
                    message=f"Syntax error in example {example_file.name}: {e.msg}",
                    severity='error',
                    details={
                        'file': str(example_file),
                        'line': e.lineno,
                        'error': str(e)
                    },
                    timestamp=datetime.now()
                ))
            
            except Exception as e:
                results.append(ValidationResult(
                    check_name='example_validation',
                    passed=False,
                    message=f"Error validating example {example_file.name}: {str(e)}",
                    severity='warning',
                    details={'file': str(example_file), 'error': str(e)},
                    timestamp=datetime.now()
                ))
        
        return results
    
    def validate_project_structure(self) -> List[ValidationResult]:
        """Validate project structure integrity."""
        results = []
        
        # Check for essential files
        essential_files = [
            'pyproject.toml',
            'README.md',
            'psysafe/__init__.py',
        ]
        
        for file_name in essential_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                results.append(ValidationResult(
                    check_name='project_structure',
                    passed=True,
                    message=f"Essential file exists: {file_name}",
                    severity='info',
                    details={'file': file_name},
                    timestamp=datetime.now()
                ))
            else:
                results.append(ValidationResult(
                    check_name='project_structure',
                    passed=False,
                    message=f"Essential file missing: {file_name}",
                    severity='error',
                    details={'file': file_name},
                    timestamp=datetime.now()
                ))
        
        # Check for proper package structure
        psysafe_dir = self.project_root / 'psysafe'
        if psysafe_dir.exists():
            expected_modules = ['core', 'catalog', 'drivers', 'evaluation']
            for module in expected_modules:
                module_path = psysafe_dir / module
                if module_path.exists() and (module_path / '__init__.py').exists():
                    results.append(ValidationResult(
                        check_name='package_structure',
                        passed=True,
                        message=f"Package module exists: psysafe.{module}",
                        severity='info',
                        details={'module': module},
                        timestamp=datetime.now()
                    ))
                else:
                    results.append(ValidationResult(
                        check_name='package_structure',
                        passed=False,
                        message=f"Package module missing or invalid: psysafe.{module}",
                        severity='warning',
                        details={'module': module},
                        timestamp=datetime.now()
                    ))
        
        return results
    
    def run_comprehensive_validation(self) -> List[ValidationResult]:
        """Run all validation checks."""
        print("Running comprehensive validation...")
        
        all_results = []
        
        # Run all validation checks
        validation_methods = [
            self.validate_syntax,
            self.validate_imports,
            self.validate_tests,
            self.validate_examples,
            self.validate_project_structure,
        ]
        
        for method in validation_methods:
            try:
                results = method()
                all_results.extend(results)
            except Exception as e:
                all_results.append(ValidationResult(
                    check_name=method.__name__,
                    passed=False,
                    message=f"Validation method failed: {str(e)}",
                    severity='error',
                    details={'error': str(e)},
                    timestamp=datetime.now()
                ))
        
        self.validation_results = all_results
        return all_results
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of validation results."""
        if not self.validation_results:
            return {"status": "not_run"}
        
        total_checks = len(self.validation_results)
        passed_checks = sum(1 for r in self.validation_results if r.passed)
        failed_checks = total_checks - passed_checks
        
        # Group by severity
        by_severity = {}
        for result in self.validation_results:
            if result.severity not in by_severity:
                by_severity[result.severity] = 0
            by_severity[result.severity] += 1
        
        # Group by check type
        by_check = {}
        for result in self.validation_results:
            if result.check_name not in by_check:
                by_check[result.check_name] = {'passed': 0, 'failed': 0}
            if result.passed:
                by_check[result.check_name]['passed'] += 1
            else:
                by_check[result.check_name]['failed'] += 1
        
        return {
            'status': 'completed',
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': failed_checks,
            'success_rate': passed_checks / total_checks if total_checks > 0 else 0,
            'by_severity': by_severity,
            'by_check_type': by_check,
            'critical_failures': [
                r.message for r in self.validation_results 
                if not r.passed and r.severity == 'critical'
            ]
        }
    
    def _get_python_files(self) -> List[Path]:
        """Get all Python files in the project."""
        python_files = []
        
        for file_path in self.project_root.rglob("*.py"):
            # Skip certain directories
            if any(part in str(file_path) for part in ['.git', '__pycache__', '.pytest_cache', '.mypy_cache', 'cleanup_system']):
                continue
            python_files.append(file_path)
        
        return sorted(python_files)
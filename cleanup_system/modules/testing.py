"""Test validation module for ensuring tests pass after cleanup."""

import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..core.base import CleanupPhase, CleanupReport, CodeCleanupBase, FileChange


@dataclass
class TestResult:
    """Represents the result of running a test."""
    
    test_name: str
    status: str  # 'passed', 'failed', 'skipped', 'error'
    duration: float
    error_message: Optional[str] = None
    traceback: Optional[str] = None


@dataclass
class TestSuite:
    """Represents a test suite."""
    
    name: str
    test_files: List[Path]
    results: List[TestResult]
    total_tests: int
    passed: int
    failed: int
    skipped: int
    duration: float


@dataclass
class TestBaseline:
    """Represents baseline test behavior before cleanup."""
    
    timestamp: datetime
    suites: List[TestSuite]
    total_tests: int
    total_passed: int
    total_failed: int
    examples_working: Dict[str, bool]


class TestValidator(CodeCleanupBase):
    """Ensures tests pass after each cleanup phase."""
    
    def __init__(self, project_root: Path, config: Optional[Dict[str, Any]] = None):
        super().__init__(project_root, config)
        self.baseline_results: Optional[TestBaseline] = None
        self.test_command = config.get('test_command', 'python -m pytest') if config else 'python -m pytest'
        self.examples_dir = self.project_root / 'examples'
    
    def get_phase(self) -> CleanupPhase:
        """Get the cleanup phase."""
        return CleanupPhase.TEST_VALIDATION
    
    def capture_baseline(self) -> TestBaseline:
        """Capture baseline test behavior before cleanup."""
        print("Capturing baseline test behavior...")
        
        # Run test suite
        test_suites = self._run_test_suite()
        
        # Test examples
        examples_working = self._test_examples()
        
        # Calculate totals
        total_tests = sum(suite.total_tests for suite in test_suites)
        total_passed = sum(suite.passed for suite in test_suites)
        total_failed = sum(suite.failed for suite in test_suites)
        
        baseline = TestBaseline(
            timestamp=datetime.now(),
            suites=test_suites,
            total_tests=total_tests,
            total_passed=total_passed,
            total_failed=total_failed,
            examples_working=examples_working
        )
        
        self.baseline_results = baseline
        return baseline
    
    def _run_test_suite(self) -> List[TestSuite]:
        """Run the complete test suite."""
        suites = []
        
        # Find test directories
        test_dirs = [
            self.project_root / 'tests',
            self.project_root / 'test',
        ]
        
        for test_dir in test_dirs:
            if test_dir.exists():
                suite = self._run_test_directory(test_dir)
                if suite:
                    suites.append(suite)
        
        return suites
    
    def _run_test_directory(self, test_dir: Path) -> Optional[TestSuite]:
        """Run tests in a specific directory."""
        try:
            # Run pytest on the directory
            cmd = self.test_command.split() + [str(test_dir), '--tb=short', '-v']
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=300  # 5 minute timeout
            )
            
            # Parse pytest output
            test_results = self._parse_pytest_output(result.stdout, result.stderr)
            
            # Get test files
            test_files = list(test_dir.rglob('test_*.py')) + list(test_dir.rglob('*_test.py'))
            
            # Calculate statistics
            passed = len([r for r in test_results if r.status == 'passed'])
            failed = len([r for r in test_results if r.status == 'failed'])
            skipped = len([r for r in test_results if r.status == 'skipped'])
            
            return TestSuite(
                name=test_dir.name,
                test_files=test_files,
                results=test_results,
                total_tests=len(test_results),
                passed=passed,
                failed=failed,
                skipped=skipped,
                duration=sum(r.duration for r in test_results)
            )
        
        except subprocess.TimeoutExpired:
            print(f"Test suite in {test_dir} timed out")
            return None
        except Exception as e:
            print(f"Error running tests in {test_dir}: {e}")
            return None
    
    def _parse_pytest_output(self, stdout: str, stderr: str) -> List[TestResult]:
        """Parse pytest output to extract test results."""
        results = []
        
        # This is a simplified parser
        # In production, use pytest's JSON output or pytest-json-report plugin
        lines = stdout.split('\n')
        
        for line in lines:
            if '::' in line and any(status in line for status in ['PASSED', 'FAILED', 'SKIPPED']):
                parts = line.split()
                if len(parts) >= 2:
                    test_name = parts[0]
                    status_part = parts[1]
                    
                    if 'PASSED' in status_part:
                        status = 'passed'
                    elif 'FAILED' in status_part:
                        status = 'failed'
                    elif 'SKIPPED' in status_part:
                        status = 'skipped'
                    else:
                        status = 'unknown'
                    
                    # Extract duration if available
                    duration = 0.0
                    for part in parts:
                        if 's' in part and part.replace('.', '').replace('s', '').isdigit():
                            try:
                                duration = float(part.replace('s', ''))
                            except:
                                pass
                    
                    results.append(TestResult(
                        test_name=test_name,
                        status=status,
                        duration=duration,
                        error_message=None,
                        traceback=None
                    ))
        
        return results
    
    def _test_examples(self) -> Dict[str, bool]:
        """Test that all examples work correctly."""
        examples_working = {}
        
        if not self.examples_dir.exists():
            return examples_working
        
        for example_file in self.examples_dir.glob('*.py'):
            try:
                # Run the example
                result = subprocess.run(
                    [sys.executable, str(example_file)],
                    capture_output=True,
                    text=True,
                    cwd=self.project_root,
                    timeout=60  # 1 minute timeout per example
                )
                
                examples_working[example_file.name] = result.returncode == 0
                
                if result.returncode != 0:
                    print(f"Example {example_file.name} failed:")
                    print(f"STDOUT: {result.stdout}")
                    print(f"STDERR: {result.stderr}")
            
            except subprocess.TimeoutExpired:
                print(f"Example {example_file.name} timed out")
                examples_working[example_file.name] = False
            except Exception as e:
                print(f"Error running example {example_file.name}: {e}")
                examples_working[example_file.name] = False
        
        return examples_working
    
    def run_test_suite(self) -> TestBaseline:
        """Execute comprehensive test suite."""
        print("Running comprehensive test suite...")
        
        # Run all tests
        test_suites = self._run_test_suite()
        
        # Test examples
        examples_working = self._test_examples()
        
        # Calculate totals
        total_tests = sum(suite.total_tests for suite in test_suites)
        total_passed = sum(suite.passed for suite in test_suites)
        total_failed = sum(suite.failed for suite in test_suites)
        
        return TestBaseline(
            timestamp=datetime.now(),
            suites=test_suites,
            total_tests=total_tests,
            total_passed=total_passed,
            total_failed=total_failed,
            examples_working=examples_working
        )
    
    def validate_no_regressions(self, current_results: TestBaseline) -> bool:
        """Ensure no functionality was broken compared to baseline."""
        if not self.baseline_results:
            print("No baseline results available for comparison")
            return True
        
        baseline = self.baseline_results
        
        # Check if total passed tests decreased
        if current_results.total_passed < baseline.total_passed:
            print(f"Regression detected: passed tests decreased from {baseline.total_passed} to {current_results.total_passed}")
            return False
        
        # Check if any previously working examples now fail
        for example_name, was_working in baseline.examples_working.items():
            if was_working and not current_results.examples_working.get(example_name, False):
                print(f"Regression detected: example {example_name} was working but now fails")
                return False
        
        # Check for new test failures
        baseline_failed_tests = set()
        for suite in baseline.suites:
            for result in suite.results:
                if result.status == 'failed':
                    baseline_failed_tests.add(result.test_name)
        
        current_failed_tests = set()
        for suite in current_results.suites:
            for result in suite.results:
                if result.status == 'failed':
                    current_failed_tests.add(result.test_name)
        
        new_failures = current_failed_tests - baseline_failed_tests
        if new_failures:
            print(f"Regression detected: new test failures: {new_failures}")
            return False
        
        return True
    
    def run_specific_tests(self, test_patterns: List[str]) -> List[TestResult]:
        """Run specific tests matching the given patterns."""
        results = []
        
        for pattern in test_patterns:
            try:
                cmd = self.test_command.split() + ['-k', pattern, '--tb=short', '-v']
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=self.project_root,
                    timeout=120
                )
                
                test_results = self._parse_pytest_output(result.stdout, result.stderr)
                results.extend(test_results)
            
            except Exception as e:
                print(f"Error running tests for pattern {pattern}: {e}")
        
        return results
    
    def analyze(self) -> Dict[str, Any]:
        """Analyze test suite status."""
        print("Analyzing test suite...")
        
        # Run current test suite
        current_results = self.run_test_suite()
        
        # Test examples
        examples_working = self._test_examples()
        working_examples = sum(1 for working in examples_working.values() if working)
        total_examples = len(examples_working)
        
        return {
            'total_tests': current_results.total_tests,
            'passed_tests': current_results.total_passed,
            'failed_tests': current_results.total_failed,
            'test_suites': len(current_results.suites),
            'examples_total': total_examples,
            'examples_working': working_examples,
            'examples_failing': total_examples - working_examples,
            'has_baseline': self.baseline_results is not None
        }
    
    def execute(self, dry_run: bool = False) -> CleanupReport:
        """Execute test validation."""
        report = CleanupReport(
            phase=self.get_phase(),
            start_time=datetime.now()
        )
        
        try:
            # Capture baseline if not already done
            if not self.baseline_results:
                baseline = self.capture_baseline()
                report.add_metric('baseline_captured', True)
                report.add_metric('baseline_total_tests', baseline.total_tests)
                report.add_metric('baseline_passed_tests', baseline.total_passed)
            
            # Run current test suite
            current_results = self.run_test_suite()
            
            # Validate no regressions
            no_regressions = self.validate_no_regressions(current_results)
            
            if not no_regressions:
                report.add_error("Test regressions detected")
            
            # Add detailed metrics
            report.add_metric('current_total_tests', current_results.total_tests)
            report.add_metric('current_passed_tests', current_results.total_passed)
            report.add_metric('current_failed_tests', current_results.total_failed)
            report.add_metric('no_regressions', no_regressions)
            
            # Report on examples
            working_examples = sum(1 for working in current_results.examples_working.values() if working)
            total_examples = len(current_results.examples_working)
            report.add_metric('examples_working', working_examples)
            report.add_metric('examples_total', total_examples)
            
            # Add warnings for failing tests
            for suite in current_results.suites:
                for result in suite.results:
                    if result.status == 'failed':
                        report.add_warning(f"Test failed: {result.test_name}")
            
            # Add warnings for failing examples
            for example_name, working in current_results.examples_working.items():
                if not working:
                    report.add_warning(f"Example not working: {example_name}")
        
        except Exception as e:
            report.add_error(f"Error during test validation: {str(e)}")
        
        report.finalize()
        return report
    
    def validate(self) -> bool:
        """Validate that test validation can be performed."""
        try:
            # Check if pytest is available
            result = subprocess.run(['python', '-m', 'pytest', '--version'], capture_output=True)
            if result.returncode != 0:
                print("pytest not available")
                return False
            
            # Check if test directory exists
            test_dirs = [
                self.project_root / 'tests',
                self.project_root / 'test',
            ]
            
            has_tests = any(test_dir.exists() for test_dir in test_dirs)
            if not has_tests:
                print("No test directories found")
                return False
            
            return True
        except:
            return False
"""Standards enforcement module for code formatting and conventions."""

import ast
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.base import CleanupPhase, CleanupReport, CodeCleanupBase, FileChange


@dataclass
class NamingViolation:
    """Represents a naming convention violation."""
    
    file_path: Path
    name: str
    line_number: int
    violation_type: str  # 'function', 'class', 'variable', 'constant'
    current_name: str
    suggested_name: str


@dataclass
class FormattingIssue:
    """Represents a formatting issue."""
    
    file_path: Path
    line_number: int
    issue_type: str
    description: str
    suggested_fix: Optional[str] = None


class StandardsEnforcer(CodeCleanupBase):
    """Enforces coding standards and formatting."""
    
    def __init__(self, project_root: Path, config: Optional[Dict[str, Any]] = None):
        super().__init__(project_root, config)
        self.ruff_config = self._load_ruff_config()
        self._ast_cache: Dict[Path, ast.AST] = {}
    
    def get_phase(self) -> CleanupPhase:
        """Get the cleanup phase."""
        return CleanupPhase.STANDARDS_ENFORCEMENT
    
    def _load_ruff_config(self) -> Dict[str, Any]:
        """Load ruff configuration."""
        default_config = {
            'line-length': 120,
            'select': ['E', 'F', 'W', 'I', 'UP', 'C90', 'N', 'D', 'S', 'B'],
            'ignore': ['D203', 'D212', 'D407', 'D416', 'E501'],
            'target-version': 'py39'
        }
        
        # In production, read from pyproject.toml
        return self.config.get('ruff', default_config)
    
    def _parse_ast(self, file_path: Path) -> Optional[ast.AST]:
        """Parse a Python file into an AST."""
        if file_path in self._ast_cache:
            return self._ast_cache[file_path]
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            tree = ast.parse(content, filename=str(file_path))
            self._ast_cache[file_path] = tree
            return tree
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None
    
    def apply_formatting(self) -> List[FileChange]:
        """Apply consistent code formatting using ruff."""
        changes = []
        
        try:
            # Run ruff format
            result = subprocess.run(
                ['ruff', 'format', str(self.project_root)],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                # Parse output to identify changed files
                # This is simplified - ruff format doesn't always provide detailed output
                for file_path in self.get_python_files():
                    changes.append(FileChange(
                        file_path=file_path,
                        change_type='modified',
                        description='Applied ruff formatting',
                        line_changes={'formatted': True}
                    ))
            else:
                print(f"Ruff format error: {result.stderr}")
        
        except FileNotFoundError:
            print("Ruff not found. Please install ruff: pip install ruff")
        except Exception as e:
            print(f"Error running ruff format: {e}")
        
        return changes
    
    def organize_imports(self) -> List[FileChange]:
        """Standardize import organization using ruff's isort."""
        changes = []
        
        try:
            # Run ruff check with import sorting
            result = subprocess.run(
                ['ruff', 'check', '--select', 'I', '--fix', str(self.project_root)],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0 or result.returncode == 1:  # 1 means fixes were applied
                # Parse output to identify changed files
                for file_path in self.get_python_files():
                    changes.append(FileChange(
                        file_path=file_path,
                        change_type='modified',
                        description='Organized imports',
                        line_changes={'imports_organized': True}
                    ))
        
        except FileNotFoundError:
            print("Ruff not found. Please install ruff: pip install ruff")
        except Exception as e:
            print(f"Error running ruff import organization: {e}")
        
        return changes
    
    def find_naming_violations(self) -> List[NamingViolation]:
        """Find naming convention violations."""
        violations = []
        
        for file_path in self.get_python_files():
            tree = self._parse_ast(file_path)
            if not tree:
                continue
            
            for node in ast.walk(tree):
                violation = self._check_naming_convention(node, file_path)
                if violation:
                    violations.append(violation)
        
        return violations
    
    def _check_naming_convention(self, node: ast.AST, file_path: Path) -> Optional[NamingViolation]:
        """Check naming convention for a node."""
        if isinstance(node, ast.FunctionDef):
            if not self._is_snake_case(node.name) and not node.name.startswith('__'):
                return NamingViolation(
                    file_path=file_path,
                    name=node.name,
                    line_number=node.lineno,
                    violation_type='function',
                    current_name=node.name,
                    suggested_name=self._to_snake_case(node.name)
                )
        
        elif isinstance(node, ast.ClassDef):
            if not self._is_pascal_case(node.name):
                return NamingViolation(
                    file_path=file_path,
                    name=node.name,
                    line_number=node.lineno,
                    violation_type='class',
                    current_name=node.name,
                    suggested_name=self._to_pascal_case(node.name)
                )
        
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    # Check for constants (all uppercase)
                    if target.id.isupper() and not self._is_upper_snake_case(target.id):
                        return NamingViolation(
                            file_path=file_path,
                            name=target.id,
                            line_number=node.lineno,
                            violation_type='constant',
                            current_name=target.id,
                            suggested_name=self._to_upper_snake_case(target.id)
                        )
                    # Check for variables
                    elif not target.id.isupper() and not self._is_snake_case(target.id):
                        return NamingViolation(
                            file_path=file_path,
                            name=target.id,
                            line_number=node.lineno,
                            violation_type='variable',
                            current_name=target.id,
                            suggested_name=self._to_snake_case(target.id)
                        )
        
        return None
    
    def _is_snake_case(self, name: str) -> bool:
        """Check if a name follows snake_case convention."""
        return re.match(r'^[a-z_][a-z0-9_]*$', name) is not None
    
    def _is_pascal_case(self, name: str) -> bool:
        """Check if a name follows PascalCase convention."""
        return re.match(r'^[A-Z][a-zA-Z0-9]*$', name) is not None
    
    def _is_upper_snake_case(self, name: str) -> bool:
        """Check if a name follows UPPER_SNAKE_CASE convention."""
        return re.match(r'^[A-Z_][A-Z0-9_]*$', name) is not None
    
    def _to_snake_case(self, name: str) -> str:
        """Convert a name to snake_case."""
        # Insert underscores before uppercase letters
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
    def _to_pascal_case(self, name: str) -> str:
        """Convert a name to PascalCase."""
        return ''.join(word.capitalize() for word in name.split('_'))
    
    def _to_upper_snake_case(self, name: str) -> str:
        """Convert a name to UPPER_SNAKE_CASE."""
        return self._to_snake_case(name).upper()
    
    def check_line_length(self) -> List[FormattingIssue]:
        """Check for lines that exceed the maximum length."""
        issues = []
        max_length = self.ruff_config.get('line-length', 120)
        
        for file_path in self.get_python_files():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for i, line in enumerate(lines, 1):
                    if len(line.rstrip()) > max_length:
                        issues.append(FormattingIssue(
                            file_path=file_path,
                            line_number=i,
                            issue_type='line_length',
                            description=f'Line exceeds {max_length} characters ({len(line.rstrip())} chars)',
                            suggested_fix='Break line into multiple lines'
                        ))
            
            except Exception as e:
                print(f"Error checking line length in {file_path}: {e}")
        
        return issues
    
    def check_import_order(self) -> List[FormattingIssue]:
        """Check import order and grouping."""
        issues = []
        
        for file_path in self.get_python_files():
            tree = self._parse_ast(file_path)
            if not tree:
                continue
            
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append((node.lineno, node))
            
            # Sort by line number
            imports.sort(key=lambda x: x[0])
            
            # Check import grouping
            prev_group = None
            for line_no, node in imports:
                current_group = self._get_import_group(node)
                
                if prev_group and current_group < prev_group:
                    issues.append(FormattingIssue(
                        file_path=file_path,
                        line_number=line_no,
                        issue_type='import_order',
                        description='Imports not properly grouped',
                        suggested_fix='Reorganize imports: stdlib, third-party, local'
                    ))
                
                prev_group = current_group
        
        return issues
    
    def _get_import_group(self, node: ast.AST) -> int:
        """Get the import group for ordering (0=stdlib, 1=third-party, 2=local)."""
        if isinstance(node, ast.Import):
            module = node.names[0].name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
        else:
            return 1
        
        # Standard library modules (simplified list)
        stdlib_modules = {
            'os', 'sys', 'json', 'datetime', 'pathlib', 'typing', 'abc',
            'collections', 'dataclasses', 'functools', 'itertools', 're',
            'subprocess', 'ast', 'difflib'
        }
        
        if module.split('.')[0] in stdlib_modules:
            return 0
        elif module.startswith('psysafe'):
            return 2
        else:
            return 1
    
    def run_linting_checks(self) -> List[FormattingIssue]:
        """Run comprehensive linting checks using ruff."""
        issues = []
        
        try:
            result = subprocess.run(
                ['ruff', 'check', str(self.project_root)],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            # Parse ruff output
            for line in result.stdout.split('\n'):
                if line.strip():
                    issue = self._parse_ruff_output(line)
                    if issue:
                        issues.append(issue)
        
        except FileNotFoundError:
            print("Ruff not found. Please install ruff: pip install ruff")
        except Exception as e:
            print(f"Error running ruff check: {e}")
        
        return issues
    
    def _parse_ruff_output(self, line: str) -> Optional[FormattingIssue]:
        """Parse a line of ruff output into a FormattingIssue."""
        # Example ruff output: "path/to/file.py:10:5: E302 expected 2 blank lines, found 1"
        match = re.match(r'^([^:]+):(\d+):(\d+):\s*(\w+)\s*(.+)$', line)
        if match:
            file_path, line_no, col, code, description = match.groups()
            return FormattingIssue(
                file_path=Path(file_path),
                line_number=int(line_no),
                issue_type=code,
                description=description.strip(),
                suggested_fix=None
            )
        return None
    
    def analyze(self) -> Dict[str, Any]:
        """Analyze code for standards compliance."""
        print("Analyzing code standards compliance...")
        
        naming_violations = self.find_naming_violations()
        line_length_issues = self.check_line_length()
        import_issues = self.check_import_order()
        linting_issues = self.run_linting_checks()
        
        return {
            'naming_violations': len(naming_violations),
            'function_naming_issues': len([v for v in naming_violations if v.violation_type == 'function']),
            'class_naming_issues': len([v for v in naming_violations if v.violation_type == 'class']),
            'variable_naming_issues': len([v for v in naming_violations if v.violation_type == 'variable']),
            'line_length_issues': len(line_length_issues),
            'import_order_issues': len(import_issues),
            'total_linting_issues': len(linting_issues),
            'affected_files': len(set(
                [v.file_path for v in naming_violations] +
                [i.file_path for i in line_length_issues] +
                [i.file_path for i in import_issues] +
                [i.file_path for i in linting_issues]
            ))
        }
    
    def execute(self, dry_run: bool = False) -> CleanupReport:
        """Execute standards enforcement."""
        report = CleanupReport(
            phase=self.get_phase(),
            start_time=datetime.now()
        )
        
        try:
            # Apply formatting
            if not dry_run:
                formatting_changes = self.apply_formatting()
                for change in formatting_changes:
                    report.add_file_change(change)
            
            # Organize imports
            if not dry_run:
                import_changes = self.organize_imports()
                for change in import_changes:
                    report.add_file_change(change)
            
            # Find remaining issues
            naming_violations = self.find_naming_violations()
            line_length_issues = self.check_line_length()
            linting_issues = self.run_linting_checks()
            
            # Report issues that need manual attention
            for violation in naming_violations:
                report.add_warning(
                    f"Naming violation in {violation.file_path}:{violation.line_number} - "
                    f"{violation.current_name} should be {violation.suggested_name}"
                )
            
            for issue in line_length_issues:
                report.add_warning(
                    f"Line length issue in {issue.file_path}:{issue.line_number} - {issue.description}"
                )
            
            # Add metrics
            report.add_metric('formatting_applied', not dry_run)
            report.add_metric('imports_organized', not dry_run)
            report.add_metric('naming_violations_found', len(naming_violations))
            report.add_metric('line_length_issues_found', len(line_length_issues))
            report.add_metric('linting_issues_found', len(linting_issues))
            
        except Exception as e:
            report.add_error(f"Error during standards enforcement: {str(e)}")
        
        report.finalize()
        return report
    
    def validate(self) -> bool:
        """Validate that standards enforcement can be performed."""
        try:
            # Check if ruff is available
            result = subprocess.run(['ruff', '--version'], capture_output=True)
            if result.returncode != 0:
                print("Ruff not available")
                return False
            
            # Check if we can parse all Python files
            for file_path in self.get_python_files():
                if not self._parse_ast(file_path):
                    return False
            
            return True
        except:
            return False
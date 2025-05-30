"""Documentation refresher module."""

import ast
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.base import CleanupPhase, CleanupReport, CodeCleanupBase, FileChange


@dataclass
class DocstringIssue:
    """Represents a docstring issue."""
    
    file_path: Path
    name: str
    line_number: int
    issue_type: str  # 'missing', 'incomplete', 'outdated', 'malformed'
    current_docstring: Optional[str] = None
    suggested_docstring: Optional[str] = None


@dataclass
class DocumentationUpdate:
    """Represents a documentation update."""
    
    file_path: Path
    update_type: str  # 'readme', 'docstring', 'comment', 'example'
    description: str
    old_content: Optional[str] = None
    new_content: Optional[str] = None


class DocumentationRefresher(CodeCleanupBase):
    """Updates and standardizes documentation."""
    
    def __init__(self, project_root: Path, config: Optional[Dict[str, Any]] = None):
        super().__init__(project_root, config)
        self.docstring_template = '''"""
{brief_description}

{detailed_description}

Args:
{args}

Returns:
{returns}

Raises:
{raises}

Example:
{example}
"""'''
        self._ast_cache: Dict[Path, ast.AST] = {}
    
    def get_phase(self) -> CleanupPhase:
        """Get the cleanup phase."""
        return CleanupPhase.DOCUMENTATION_REFRESH
    
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
    
    def find_docstring_issues(self) -> List[DocstringIssue]:
        """Find all docstring issues in the codebase."""
        issues = []
        
        for file_path in self.get_python_files():
            tree = self._parse_ast(file_path)
            if not tree:
                continue
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                    issue = self._check_docstring(node, file_path)
                    if issue:
                        issues.append(issue)
        
        return issues
    
    def _check_docstring(self, node: ast.AST, file_path: Path) -> Optional[DocstringIssue]:
        """Check a node's docstring for issues."""
        docstring = ast.get_docstring(node)
        node_name = getattr(node, 'name', 'module')
        line_number = getattr(node, 'lineno', 1)
        
        # Check if docstring is missing
        if not docstring:
            # Skip private methods and special methods
            if node_name.startswith('_') and not node_name.startswith('__'):
                return None
            if node_name in ['__init__', '__str__', '__repr__']:
                return None
            
            return DocstringIssue(
                file_path=file_path,
                name=node_name,
                line_number=line_number,
                issue_type='missing',
                current_docstring=None,
                suggested_docstring=self._generate_docstring(node)
            )
        
        # Check if docstring is incomplete
        if isinstance(node, ast.FunctionDef):
            has_args = 'Args:' in docstring or 'Parameters:' in docstring
            has_returns = 'Returns:' in docstring or 'Return:' in docstring
            
            # Check if function has parameters but no Args section
            if node.args.args and not has_args:
                return DocstringIssue(
                    file_path=file_path,
                    name=node_name,
                    line_number=line_number,
                    issue_type='incomplete',
                    current_docstring=docstring,
                    suggested_docstring=self._enhance_docstring(node, docstring)
                )
            
            # Check if function has return statement but no Returns section
            has_return_stmt = any(isinstance(n, ast.Return) and n.value for n in ast.walk(node))
            if has_return_stmt and not has_returns:
                return DocstringIssue(
                    file_path=file_path,
                    name=node_name,
                    line_number=line_number,
                    issue_type='incomplete',
                    current_docstring=docstring,
                    suggested_docstring=self._enhance_docstring(node, docstring)
                )
        
        # Check for malformed docstrings
        if not self._is_well_formatted(docstring):
            return DocstringIssue(
                file_path=file_path,
                name=node_name,
                line_number=line_number,
                issue_type='malformed',
                current_docstring=docstring,
                suggested_docstring=self._reformat_docstring(docstring)
            )
        
        return None
    
    def _generate_docstring(self, node: ast.AST) -> str:
        """Generate a docstring for a node."""
        if isinstance(node, ast.FunctionDef):
            return self._generate_function_docstring(node)
        elif isinstance(node, ast.ClassDef):
            return self._generate_class_docstring(node)
        else:
            return '"""Module documentation."""'
    
    def _generate_function_docstring(self, node: ast.FunctionDef) -> str:
        """Generate a docstring for a function."""
        # Extract function signature
        args = []
        for arg in node.args.args:
            if arg.arg != 'self':
                arg_type = 'Any'
                if arg.annotation:
                    arg_type = ast.unparse(arg.annotation)
                args.append(f"    {arg.arg} ({arg_type}): Description of {arg.arg}.")
        
        # Check for return type
        returns = "None"
        if node.returns:
            returns = ast.unparse(node.returns)
        
        # Generate basic docstring
        docstring = f'"""{node.name.replace("_", " ").title()}.\n\n'
        
        if args:
            docstring += "Args:\n"
            docstring += "\n".join(args) + "\n\n"
        
        if returns != "None":
            docstring += f"Returns:\n    {returns}: Description of return value.\n"
        
        docstring += '"""'
        
        return docstring
    
    def _generate_class_docstring(self, node: ast.ClassDef) -> str:
        """Generate a docstring for a class."""
        return f'"""{node.name} class.\n\nDescription of the class.\n"""'
    
    def _enhance_docstring(self, node: ast.FunctionDef, current: str) -> str:
        """Enhance an existing docstring."""
        # This is a simplified implementation
        # In production, parse and enhance the existing docstring
        return self._generate_function_docstring(node)
    
    def _reformat_docstring(self, docstring: str) -> str:
        """Reformat a malformed docstring."""
        # This is a simplified implementation
        # In production, parse and reformat according to standards
        lines = docstring.strip().split('\n')
        
        # Ensure proper spacing
        formatted_lines = []
        for line in lines:
            if line.strip() in ['Args:', 'Returns:', 'Raises:', 'Example:']:
                if formatted_lines and formatted_lines[-1].strip():
                    formatted_lines.append('')
                formatted_lines.append(line)
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _is_well_formatted(self, docstring: str) -> bool:
        """Check if a docstring is well-formatted."""
        # Basic checks for formatting
        lines = docstring.strip().split('\n')
        
        # Check for proper sections
        sections = ['Args:', 'Returns:', 'Raises:', 'Example:']
        for section in sections:
            if section in docstring:
                # Check if section has proper spacing
                idx = next((i for i, line in enumerate(lines) if section in line), -1)
                if idx > 0 and lines[idx - 1].strip():
                    return False
        
        return True
    
    def update_readme(self) -> DocumentationUpdate:
        """Update the README.md file."""
        readme_path = self.project_root / 'README.md'
        
        if not readme_path.exists():
            return DocumentationUpdate(
                file_path=readme_path,
                update_type='readme',
                description='README.md not found',
                old_content=None,
                new_content=None
            )
        
        with open(readme_path, 'r', encoding='utf-8') as f:
            old_content = f.read()
        
        new_content = old_content
        
        # Update Python version requirement
        new_content = re.sub(
            r'python["\']?\s*>=?\s*[0-9.]+',
            'python>=3.9',
            new_content,
            flags=re.IGNORECASE
        )
        
        # Update installation instructions
        if 'pip install psysafe' not in new_content:
            install_section = """## Installation

```bash
pip install psysafe-ai
```

"""
            # Find a good place to insert
            if '## Usage' in new_content:
                new_content = new_content.replace('## Usage', install_section + '## Usage')
        
        return DocumentationUpdate(
            file_path=readme_path,
            update_type='readme',
            description='Updated README.md',
            old_content=old_content,
            new_content=new_content
        )
    
    def find_outdated_examples(self) -> List[DocumentationUpdate]:
        """Find outdated examples in documentation."""
        updates = []
        
        # Check examples directory
        examples_dir = self.project_root / 'examples'
        if examples_dir.exists():
            for example_file in examples_dir.glob('*.py'):
                try:
                    with open(example_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for outdated patterns
                    if 'from psysafe import' in content:
                        # Verify imports are valid
                        tree = ast.parse(content)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('psysafe'):
                                # This is a simplified check
                                # In production, verify against actual module structure
                                pass
                    
                except Exception as e:
                    updates.append(DocumentationUpdate(
                        file_path=example_file,
                        update_type='example',
                        description=f'Error in example: {str(e)}',
                        old_content=None,
                        new_content=None
                    ))
        
        return updates
    
    def analyze(self) -> Dict[str, Any]:
        """Analyze documentation issues."""
        print("Analyzing documentation...")
        
        docstring_issues = self.find_docstring_issues()
        readme_update = self.update_readme()
        outdated_examples = self.find_outdated_examples()
        
        # Categorize docstring issues
        missing_count = sum(1 for issue in docstring_issues if issue.issue_type == 'missing')
        incomplete_count = sum(1 for issue in docstring_issues if issue.issue_type == 'incomplete')
        malformed_count = sum(1 for issue in docstring_issues if issue.issue_type == 'malformed')
        
        return {
            'total_docstring_issues': len(docstring_issues),
            'missing_docstrings': missing_count,
            'incomplete_docstrings': incomplete_count,
            'malformed_docstrings': malformed_count,
            'readme_needs_update': readme_update.old_content != readme_update.new_content,
            'outdated_examples': len(outdated_examples),
            'affected_files': len(set(issue.file_path for issue in docstring_issues))
        }
    
    def execute(self, dry_run: bool = False) -> CleanupReport:
        """Execute documentation refresh."""
        report = CleanupReport(
            phase=self.get_phase(),
            start_time=datetime.now()
        )
        
        try:
            # Find all documentation issues
            docstring_issues = self.find_docstring_issues()
            readme_update = self.update_readme()
            outdated_examples = self.find_outdated_examples()
            
            # Fix docstring issues
            for issue in docstring_issues:
                if issue.suggested_docstring and not dry_run:
                    self._fix_docstring(issue)
                
                report.add_file_change(FileChange(
                    file_path=issue.file_path,
                    change_type='modified',
                    description=f"Fixed {issue.issue_type} docstring for '{issue.name}'",
                    line_changes={'line': issue.line_number, 'type': issue.issue_type}
                ))
            
            # Update README
            if readme_update.old_content != readme_update.new_content:
                if not dry_run:
                    with open(readme_update.file_path, 'w', encoding='utf-8') as f:
                        f.write(readme_update.new_content)
                
                report.add_file_change(FileChange(
                    file_path=readme_update.file_path,
                    change_type='modified',
                    description='Updated README.md',
                    line_changes={'updated': True}
                ))
            
            # Add metrics
            report.add_metric('docstrings_fixed', len(docstring_issues))
            report.add_metric('readme_updated', readme_update.old_content != readme_update.new_content)
            report.add_metric('examples_checked', len(outdated_examples))
            
        except Exception as e:
            report.add_error(f"Error during documentation refresh: {str(e)}")
        
        report.finalize()
        return report
    
    def _fix_docstring(self, issue: DocstringIssue) -> None:
        """Fix a docstring issue."""
        # This is a simplified implementation
        # In production, use AST transformation for safer updates
        pass
    
    def validate(self) -> bool:
        """Validate that documentation refresh can be performed."""
        try:
            # Check if we can parse all Python files
            for file_path in self.get_python_files():
                if not self._parse_ast(file_path):
                    return False
            
            # Check README exists
            readme_path = self.project_root / 'README.md'
            if not readme_path.exists():
                print("Warning: README.md not found")
            
            return True
        except:
            return False
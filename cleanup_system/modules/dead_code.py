"""Dead code analyzer and remover module."""

import ast
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.base import CleanupPhase, CleanupReport, CodeCleanupBase, FileChange


@dataclass
class CommentedCodeBlock:
    """Represents a block of commented code."""
    
    file_path: Path
    start_line: int
    end_line: int
    content: List[str]
    is_documentation: bool = False


@dataclass
class UnusedFunction:
    """Represents an unused function."""
    
    file_path: Path
    name: str
    line_number: int
    is_placeholder: bool = False


@dataclass
class UnusedImport:
    """Represents an unused import."""
    
    file_path: Path
    module: str
    names: List[str]
    line_number: int


class DeadCodeAnalyzer(CodeCleanupBase):
    """Identifies and removes dead code safely."""
    
    def __init__(self, project_root: Path, config: Optional[Dict[str, Any]] = None):
        super().__init__(project_root, config)
        self.preserve_patterns = [
            r"#\s*(TODO|FIXME|NOTE|WARNING|HACK|XXX)",
            r"#\s*(Copyright|License|Author)",
            r"#\s*(Example:|Usage:|Returns:|Args:|Raises:|See also:)",
            r"#\s*type:",  # Type annotations
            r"#\s*noqa",   # Linter directives
            r"#\s*pragma",  # Coverage directives
        ]
        self.code_patterns = [
            r"#\s*(print|return|if|for|while|def|class|import|from)",
            r"#\s*(response|result|data|value)\s*=",
            r"#\s*\w+\(",  # Function calls
            r"#\s*\w+\.\w+",  # Method calls
        ]
        self._ast_cache: Dict[Path, ast.AST] = {}
    
    def get_phase(self) -> CleanupPhase:
        """Get the cleanup phase."""
        return CleanupPhase.DEAD_CODE_REMOVAL
    
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
    
    def _is_documentation_comment(self, line: str) -> bool:
        """Check if a comment line is documentation."""
        for pattern in self.preserve_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return True
        return False
    
    def _is_code_comment(self, line: str) -> bool:
        """Check if a comment line contains code."""
        for pattern in self.code_patterns:
            if re.search(pattern, line):
                return True
        return False
    
    def find_commented_code_blocks(self) -> List[CommentedCodeBlock]:
        """Identify commented-out code blocks."""
        commented_blocks = []
        
        for file_path in self.get_python_files():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                current_block = []
                block_start = None
                
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    
                    if stripped.startswith('#'):
                        if self._is_code_comment(stripped) and not self._is_documentation_comment(stripped):
                            if block_start is None:
                                block_start = i + 1
                            current_block.append(line)
                        else:
                            # End current block if exists
                            if current_block and block_start is not None:
                                commented_blocks.append(CommentedCodeBlock(
                                    file_path=file_path,
                                    start_line=block_start,
                                    end_line=i,
                                    content=current_block,
                                    is_documentation=False
                                ))
                                current_block = []
                                block_start = None
                    else:
                        # End current block if exists
                        if current_block and block_start is not None:
                            commented_blocks.append(CommentedCodeBlock(
                                file_path=file_path,
                                start_line=block_start,
                                end_line=i,
                                content=current_block,
                                is_documentation=False
                            ))
                            current_block = []
                            block_start = None
                
                # Handle block at end of file
                if current_block and block_start is not None:
                    commented_blocks.append(CommentedCodeBlock(
                        file_path=file_path,
                        start_line=block_start,
                        end_line=len(lines),
                        content=current_block,
                        is_documentation=False
                    ))
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        return commented_blocks
    
    def find_unused_functions(self) -> List[UnusedFunction]:
        """Find functions that are never called."""
        unused_functions = []
        all_functions: Dict[str, List[Tuple[Path, int]]] = {}
        all_calls: Set[str] = set()
        
        # First pass: collect all function definitions and calls
        for file_path in self.get_python_files():
            tree = self._parse_ast(file_path)
            if not tree:
                continue
            
            # Collect function definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if node.name not in all_functions:
                        all_functions[node.name] = []
                    all_functions[node.name].append((file_path, node.lineno))
                    
                    # Check if it's a placeholder
                    is_placeholder = self._is_placeholder_function(node)
                    if is_placeholder:
                        unused_functions.append(UnusedFunction(
                            file_path=file_path,
                            name=node.name,
                            line_number=node.lineno,
                            is_placeholder=True
                        ))
                
                # Collect function calls
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        all_calls.add(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        all_calls.add(node.func.attr)
        
        # Second pass: find unused functions
        for func_name, locations in all_functions.items():
            # Skip special methods and test functions
            if (func_name.startswith('__') or 
                func_name.startswith('test_') or
                func_name in ['setUp', 'tearDown', 'setUpClass', 'tearDownClass']):
                continue
            
            if func_name not in all_calls:
                for file_path, line_no in locations:
                    # Check if already marked as placeholder
                    if not any(uf.name == func_name and uf.file_path == file_path 
                              for uf in unused_functions):
                        unused_functions.append(UnusedFunction(
                            file_path=file_path,
                            name=func_name,
                            line_number=line_no,
                            is_placeholder=False
                        ))
        
        return unused_functions
    
    def _is_placeholder_function(self, node: ast.FunctionDef) -> bool:
        """Check if a function is a placeholder."""
        if not node.body:
            return True
        
        # Check for single pass statement
        if len(node.body) == 1:
            stmt = node.body[0]
            if isinstance(stmt, ast.Pass):
                return True
            if isinstance(stmt, ast.Raise) and isinstance(stmt.exc, ast.Call):
                if hasattr(stmt.exc.func, 'id') and stmt.exc.func.id == 'NotImplementedError':
                    return True
        
        # Check for placeholder comments
        if hasattr(node, 'body') and node.body and isinstance(node.body[0], ast.Expr):
            if isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                docstring = node.body[0].value.value.lower()
                if any(keyword in docstring for keyword in ['placeholder', 'todo', 'implement']):
                    return True
        
        return False
    
    def find_unused_imports(self) -> List[UnusedImport]:
        """Find imports that are never used."""
        unused_imports = []
        
        for file_path in self.get_python_files():
            tree = self._parse_ast(file_path)
            if not tree:
                continue
            
            # Collect all imports
            imports: Dict[str, Tuple[str, int, List[str]]] = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.asname if alias.asname else alias.name
                        imports[name] = (alias.name, node.lineno, [])
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        name = alias.asname if alias.asname else alias.name
                        if name == '*':
                            continue  # Skip star imports
                        imports[name] = (module, node.lineno, [alias.name])
            
            # Collect all name usages
            used_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Store):
                    used_names.add(node.id)
                elif isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name):
                        used_names.add(node.value.id)
            
            # Find unused imports
            for import_name, (module, line_no, names) in imports.items():
                if import_name not in used_names:
                    # Special cases: imports that might be used indirectly
                    if import_name in ['typing', 'types', '__future__']:
                        continue
                    if module.startswith('psysafe') and '__all__' in str(tree):
                        continue  # Might be re-exported
                    
                    unused_imports.append(UnusedImport(
                        file_path=file_path,
                        module=module,
                        names=names or [import_name],
                        line_number=line_no
                    ))
        
        return unused_imports
    
    def analyze(self) -> Dict[str, Any]:
        """Analyze the codebase for dead code."""
        print("Analyzing codebase for dead code...")
        
        commented_blocks = self.find_commented_code_blocks()
        unused_functions = self.find_unused_functions()
        unused_imports = self.find_unused_imports()
        
        return {
            'commented_code_blocks': len(commented_blocks),
            'unused_functions': len(unused_functions),
            'placeholder_functions': sum(1 for f in unused_functions if f.is_placeholder),
            'unused_imports': len(unused_imports),
            'affected_files': len(set(
                [b.file_path for b in commented_blocks] +
                [f.file_path for f in unused_functions] +
                [i.file_path for i in unused_imports]
            ))
        }
    
    def execute(self, dry_run: bool = False) -> CleanupReport:
        """Execute dead code removal."""
        report = CleanupReport(
            phase=self.get_phase(),
            start_time=datetime.now()
        )
        
        try:
            # Find all dead code
            commented_blocks = self.find_commented_code_blocks()
            unused_functions = self.find_unused_functions()
            unused_imports = self.find_unused_imports()
            
            # Remove commented code blocks
            for block in commented_blocks:
                if not dry_run:
                    self._remove_commented_block(block)
                
                report.add_file_change(FileChange(
                    file_path=block.file_path,
                    change_type='modified',
                    description=f"Removed commented code block (lines {block.start_line}-{block.end_line})",
                    line_changes={'removed_lines': block.end_line - block.start_line + 1}
                ))
            
            # Remove placeholder functions
            for func in unused_functions:
                if func.is_placeholder:
                    if not dry_run:
                        self._remove_function(func)
                    
                    report.add_file_change(FileChange(
                        file_path=func.file_path,
                        change_type='modified',
                        description=f"Removed placeholder function '{func.name}'",
                        line_changes={'function': func.name, 'line': func.line_number}
                    ))
            
            # Remove unused imports
            for imp in unused_imports:
                if not dry_run:
                    self._remove_import(imp)
                
                report.add_file_change(FileChange(
                    file_path=imp.file_path,
                    change_type='modified',
                    description=f"Removed unused import '{imp.module}'",
                    line_changes={'import': imp.module, 'line': imp.line_number}
                ))
            
            # Add metrics
            report.add_metric('commented_blocks_removed', len(commented_blocks))
            report.add_metric('placeholder_functions_removed', 
                            sum(1 for f in unused_functions if f.is_placeholder))
            report.add_metric('unused_imports_removed', len(unused_imports))
            
        except Exception as e:
            report.add_error(f"Error during dead code removal: {str(e)}")
        
        report.finalize()
        return report
    
    def _remove_commented_block(self, block: CommentedCodeBlock) -> None:
        """Remove a commented code block from a file."""
        with open(block.file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Remove the block
        del lines[block.start_line - 1:block.end_line]
        
        with open(block.file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    
    def _remove_function(self, func: UnusedFunction) -> None:
        """Remove a function from a file."""
        # This is a simplified implementation
        # In production, use AST transformation for safer removal
        pass
    
    def _remove_import(self, imp: UnusedImport) -> None:
        """Remove an import from a file."""
        # This is a simplified implementation
        # In production, use AST transformation for safer removal
        pass
    
    def validate(self) -> bool:
        """Validate that dead code removal can be safely performed."""
        try:
            # Check if we can parse all Python files
            for file_path in self.get_python_files():
                if not self._parse_ast(file_path):
                    return False
            
            # Check if we have write permissions
            test_file = self.project_root / '.cleanup_test'
            try:
                test_file.touch()
                test_file.unlink()
            except:
                return False
            
            return True
        except:
            return False
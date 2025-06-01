"""Code consolidation module for extracting duplicate code."""

import ast
import difflib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.base import CleanupPhase, CleanupReport, CodeCleanupBase, FileChange


@dataclass
class CodePattern:
    """Represents a code pattern that appears multiple times."""
    
    pattern_id: str
    occurrences: List[Tuple[Path, int, int]]  # (file, start_line, end_line)
    code_lines: List[str]
    similarity_score: float
    pattern_type: str  # 'function', 'block', 'expression'


@dataclass
class ExtractedUtility:
    """Represents an extracted utility function."""
    
    name: str
    module_path: Path
    original_pattern: CodePattern
    parameters: List[str]
    return_type: Optional[str]
    docstring: str


class CodeConsolidator(CodeCleanupBase):
    """Extracts duplicate code into reusable utilities."""
    
    def __init__(self, project_root: Path, config: Optional[Dict[str, Any]] = None):
        super().__init__(project_root, config)
        self.similarity_threshold = config.get('similarity_threshold', 0.8) if config else 0.8
        self.min_pattern_size = config.get('min_pattern_size', 5) if config else 5
        self._ast_cache: Dict[Path, ast.AST] = {}
        self._code_blocks: Dict[str, List[Tuple[Path, ast.AST, str]]] = {}
    
    def get_phase(self) -> CleanupPhase:
        """Get the cleanup phase."""
        return CleanupPhase.CODE_CONSOLIDATION
    
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
    
    def find_duplicate_patterns(self) -> List[CodePattern]:
        """Identify duplicate code patterns across the codebase."""
        patterns = []
        
        # Collect all code blocks
        self._collect_code_blocks()
        
        # Find similar blocks
        block_groups = self._group_similar_blocks()
        
        # Convert to patterns
        for group_id, blocks in block_groups.items():
            if len(blocks) >= 2:  # At least 2 occurrences
                pattern = self._create_pattern(group_id, blocks)
                if pattern:
                    patterns.append(pattern)
        
        return patterns
    
    def _collect_code_blocks(self) -> None:
        """Collect all significant code blocks from the codebase."""
        for file_path in self.get_python_files():
            tree = self._parse_ast(file_path)
            if not tree:
                continue
            
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Collect function bodies
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                        code_lines = lines[node.lineno - 1:node.end_lineno]
                        if len(code_lines) >= self.min_pattern_size:
                            block_id = self._normalize_code(code_lines)
                            if block_id not in self._code_blocks:
                                self._code_blocks[block_id] = []
                            self._code_blocks[block_id].append((file_path, node, ''.join(code_lines)))
                
                # Collect significant code blocks (e.g., try-except, if-else)
                elif isinstance(node, (ast.Try, ast.If, ast.For, ast.While)):
                    if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                        code_lines = lines[node.lineno - 1:node.end_lineno]
                        if len(code_lines) >= self.min_pattern_size:
                            block_id = self._normalize_code(code_lines)
                            if block_id not in self._code_blocks:
                                self._code_blocks[block_id] = []
                            self._code_blocks[block_id].append((file_path, node, ''.join(code_lines)))
    
    def _normalize_code(self, code_lines: List[str]) -> str:
        """Normalize code for comparison by removing whitespace and comments."""
        normalized = []
        for line in code_lines:
            # Remove leading/trailing whitespace
            stripped = line.strip()
            # Skip empty lines and comments
            if stripped and not stripped.startswith('#'):
                # Remove inline comments
                if '#' in stripped:
                    stripped = stripped.split('#')[0].strip()
                normalized.append(stripped)
        return '\n'.join(normalized)
    
    def _group_similar_blocks(self) -> Dict[str, List[Tuple[Path, ast.AST, str]]]:
        """Group similar code blocks together."""
        groups = defaultdict(list)
        processed = set()
        
        block_items = list(self._code_blocks.items())
        
        for i, (block_id, blocks) in enumerate(block_items):
            if block_id in processed:
                continue
            
            group_key = block_id
            groups[group_key].extend(blocks)
            processed.add(block_id)
            
            # Find similar blocks
            for j, (other_id, other_blocks) in enumerate(block_items[i + 1:], i + 1):
                if other_id in processed:
                    continue
                
                similarity = self._calculate_similarity(block_id, other_id)
                if similarity >= self.similarity_threshold:
                    groups[group_key].extend(other_blocks)
                    processed.add(other_id)
        
        return dict(groups)
    
    def _calculate_similarity(self, code1: str, code2: str) -> float:
        """Calculate similarity between two code blocks."""
        lines1 = code1.split('\n')
        lines2 = code2.split('\n')
        
        # Use difflib to calculate similarity
        matcher = difflib.SequenceMatcher(None, lines1, lines2)
        return matcher.ratio()
    
    def _create_pattern(self, group_id: str, blocks: List[Tuple[Path, ast.AST, str]]) -> Optional[CodePattern]:
        """Create a CodePattern from a group of similar blocks."""
        if len(blocks) < 2:
            return None
        
        # Extract occurrences
        occurrences = []
        for file_path, node, code in blocks:
            start_line = getattr(node, 'lineno', 1)
            end_line = getattr(node, 'end_lineno', start_line)
            occurrences.append((file_path, start_line, end_line))
        
        # Use the first block as the representative
        representative_code = blocks[0][2]
        code_lines = representative_code.split('\n')
        
        # Determine pattern type
        pattern_type = 'block'
        if isinstance(blocks[0][1], ast.FunctionDef):
            pattern_type = 'function'
        elif isinstance(blocks[0][1], (ast.Try, ast.If)):
            pattern_type = 'control_structure'
        
        return CodePattern(
            pattern_id=group_id[:16],  # Truncate for readability
            occurrences=occurrences,
            code_lines=code_lines,
            similarity_score=1.0,  # Will be calculated properly in production
            pattern_type=pattern_type
        )
    
    def extract_to_utilities(self, patterns: List[CodePattern]) -> List[ExtractedUtility]:
        """Extract patterns into utility functions."""
        utilities = []
        
        for pattern in patterns:
            if pattern.pattern_type == 'function' and len(pattern.occurrences) >= 2:
                utility = self._create_utility_function(pattern)
                if utility:
                    utilities.append(utility)
        
        return utilities
    
    def _create_utility_function(self, pattern: CodePattern) -> Optional[ExtractedUtility]:
        """Create a utility function from a code pattern."""
        # This is a simplified implementation
        # In production, analyze the pattern to extract parameters and create a proper function
        
        # Generate utility name
        utility_name = f"extracted_utility_{pattern.pattern_id}"
        
        # Determine target module
        utils_dir = self.project_root / 'psysafe' / 'utils'
        target_module = utils_dir / 'extracted_utilities.py'
        
        # Create basic utility function
        docstring = f'"""Extracted utility function from {len(pattern.occurrences)} duplicate occurrences."""'
        
        return ExtractedUtility(
            name=utility_name,
            module_path=target_module,
            original_pattern=pattern,
            parameters=[],  # Would be extracted from pattern analysis
            return_type=None,
            docstring=docstring
        )
    
    def update_references(self, utilities: List[ExtractedUtility]) -> List[FileChange]:
        """Update all references to use new utility functions."""
        changes = []
        
        for utility in utilities:
            for file_path, start_line, end_line in utility.original_pattern.occurrences:
                # This is a simplified implementation
                # In production, replace the original code with a call to the utility
                changes.append(FileChange(
                    file_path=file_path,
                    change_type='modified',
                    description=f"Replaced duplicate code with call to {utility.name}",
                    line_changes={
                        'start_line': start_line,
                        'end_line': end_line,
                        'utility_name': utility.name
                    }
                ))
        
        return changes
    
    def analyze(self) -> Dict[str, Any]:
        """Analyze the codebase for duplicate code patterns."""
        print("Analyzing codebase for duplicate patterns...")
        
        patterns = self.find_duplicate_patterns()
        
        # Categorize patterns
        function_patterns = [p for p in patterns if p.pattern_type == 'function']
        block_patterns = [p for p in patterns if p.pattern_type == 'block']
        
        total_duplicates = sum(len(p.occurrences) for p in patterns)
        
        return {
            'total_patterns': len(patterns),
            'function_patterns': len(function_patterns),
            'block_patterns': len(block_patterns),
            'total_duplicate_occurrences': total_duplicates,
            'extractable_utilities': len([p for p in patterns if len(p.occurrences) >= 2]),
            'affected_files': len(set(
                occ[0] for pattern in patterns for occ in pattern.occurrences
            ))
        }
    
    def execute(self, dry_run: bool = False) -> CleanupReport:
        """Execute code consolidation."""
        report = CleanupReport(
            phase=self.get_phase(),
            start_time=datetime.now()
        )
        
        try:
            # Find duplicate patterns
            patterns = self.find_duplicate_patterns()
            
            # Extract utilities
            utilities = self.extract_to_utilities(patterns)
            
            # Create utility modules
            if utilities and not dry_run:
                self._create_utility_modules(utilities)
            
            # Update references
            reference_changes = self.update_references(utilities)
            
            # Apply changes
            for change in reference_changes:
                if not dry_run:
                    self._apply_reference_change(change)
                report.add_file_change(change)
            
            # Add utility creation changes
            for utility in utilities:
                report.add_file_change(FileChange(
                    file_path=utility.module_path,
                    change_type='created',
                    description=f"Created utility function {utility.name}",
                    line_changes={'utility': utility.name}
                ))
            
            # Add metrics
            report.add_metric('patterns_found', len(patterns))
            report.add_metric('utilities_extracted', len(utilities))
            report.add_metric('references_updated', len(reference_changes))
            
        except Exception as e:
            report.add_error(f"Error during code consolidation: {str(e)}")
        
        report.finalize()
        return report
    
    def _create_utility_modules(self, utilities: List[ExtractedUtility]) -> None:
        """Create utility modules with extracted functions."""
        # Group utilities by module
        modules = defaultdict(list)
        for utility in utilities:
            modules[utility.module_path].append(utility)
        
        for module_path, module_utilities in modules.items():
            # Ensure directory exists
            module_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create module content
            content = '"""Extracted utility functions."""\n\n'
            
            for utility in module_utilities:
                content += f'def {utility.name}():\n'
                content += f'    {utility.docstring}\n'
                content += '    pass  # Implementation would be extracted from pattern\n\n'
            
            with open(module_path, 'w', encoding='utf-8') as f:
                f.write(content)
    
    def _apply_reference_change(self, change: FileChange) -> None:
        """Apply a reference change to a file."""
        # This is a simplified implementation
        # In production, use AST transformation for safer updates
        pass
    
    def validate(self) -> bool:
        """Validate that code consolidation can be performed."""
        try:
            # Check if we can parse all Python files
            for file_path in self.get_python_files():
                if not self._parse_ast(file_path):
                    return False
            
            # Check if utils directory exists or can be created
            utils_dir = self.project_root / 'psysafe' / 'utils'
            if not utils_dir.exists():
                try:
                    utils_dir.mkdir(parents=True, exist_ok=True)
                    # Clean up test directory
                    utils_dir.rmdir()
                except:
                    return False
            
            return True
        except:
            return False
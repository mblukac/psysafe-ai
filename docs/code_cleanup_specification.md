# PsySafe-AI Code Cleanup and Refactoring Specification

## Overview

This specification outlines a comprehensive cleanup and refactoring plan for the psysafe-ai repository to improve code quality, maintainability, and consistency while preserving all existing functionality.

## Project Analysis Summary

**Current State:**
- Python project with clear modular structure
- Main package: `psysafe/` with core, catalog, drivers, evaluation modules
- Test coverage: `tests/` directory with comprehensive test files
- Documentation: `docs/` with markdown files
- Examples: `examples/` with usage demonstrations
- Configuration: Modern `pyproject.toml` with proper dependencies

**Key Findings:**
- Extensive commented code blocks (278 comment matches found)
- Placeholder implementations in several modules
- Inconsistent documentation patterns
- Some redundant utility functions
- Configuration scattered across multiple files

## Cleanup Phases

### Phase 1: Dead Code Removal

#### 1.1 Commented Code Cleanup
**Target Files:**
- All Python files with extensive comment blocks
- Focus on `examples/`, `psysafe/evaluation/`, `psysafe/cli/`

**Pseudocode:**
```python
def cleanup_commented_code():
    """Remove commented-out code blocks while preserving documentation"""
    
    PRESERVE_PATTERNS = [
        "# TODO", "# FIXME", "# NOTE", "# WARNING",
        "# Copyright", "# License", "# Author",
        "# Example:", "# Usage:", "# Returns:",
        "# Args:", "# Raises:", "# See also:"
    ]
    
    REMOVE_PATTERNS = [
        "# print(", "# return ", "# if ", "# for ",
        "# def ", "# class ", "# import ",
        "# response = ", "# result = "
    ]
    
    for file_path in get_python_files():
        content = read_file(file_path)
        cleaned_lines = []
        
        for line in content.split('\n'):
            if should_preserve_comment(line, PRESERVE_PATTERNS):
                cleaned_lines.append(line)
            elif should_remove_comment(line, REMOVE_PATTERNS):
                continue  # Skip commented code
            else:
                cleaned_lines.append(line)
        
        write_file(file_path, '\n'.join(cleaned_lines))
        
    return cleanup_report
```

#### 1.2 Placeholder Function Removal
**Target Files:**
- `psysafe/utils/parsing.py` - `_parse_xml_like()` method
- `psysafe/evaluation/metrics.py` - placeholder metrics
- `psysafe/cli/` modules - incomplete CLI commands

**Pseudocode:**
```python
def remove_placeholder_functions():
    """Remove or complete placeholder implementations"""
    
    PLACEHOLDER_INDICATORS = [
        "# Placeholder", "NotImplementedError", 
        "pass  # TODO", "# This is a placeholder"
    ]
    
    for file_path in get_python_files():
        ast_tree = parse_ast(file_path)
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                if has_placeholder_body(node):
                    if is_unused_function(node, file_path):
                        remove_function(node, file_path)
                    else:
                        mark_for_implementation(node, file_path)
    
    return removal_report
```

#### 1.3 Unused Import Cleanup
**Pseudocode:**
```python
def cleanup_unused_imports():
    """Remove unused imports using AST analysis"""
    
    for file_path in get_python_files():
        imports = extract_imports(file_path)
        used_names = extract_used_names(file_path)
        
        unused_imports = []
        for imp in imports:
            if not is_import_used(imp, used_names):
                unused_imports.append(imp)
        
        remove_imports(file_path, unused_imports)
        
    return import_cleanup_report
```

### Phase 2: Documentation Refresh

#### 2.1 README and Core Documentation Update
**Target Files:**
- `README.md` - Update badges, links, examples
- `docs/` - Refresh all markdown files
- Inline docstrings across all modules

**Pseudocode:**
```python
def refresh_documentation():
    """Update documentation to match current functionality"""
    
    # Update README.md
    readme_updates = {
        'python_version': '>=3.9',
        'badges': update_badge_urls(),
        'examples': extract_working_examples(),
        'installation': verify_pip_install_command()
    }
    
    update_readme(readme_updates)
    
    # Update API documentation
    for module_path in get_all_modules():
        module_doc = generate_module_documentation(module_path)
        update_docstrings(module_path, module_doc)
        
    # Update inline comments
    for file_path in get_python_files():
        update_inline_documentation(file_path)
        
    return documentation_report
```

#### 2.2 Docstring Standardization
**Pseudocode:**
```python
def standardize_docstrings():
    """Ensure consistent docstring format across all modules"""
    
    DOCSTRING_TEMPLATE = '''
    """Brief description.
    
    Detailed description if needed.
    
    Args:
        param1 (type): Description.
        param2 (type): Description.
        
    Returns:
        type: Description.
        
    Raises:
        ExceptionType: Description.
        
    Example:
        >>> example_usage()
        expected_output
    """
    '''
    
    for file_path in get_python_files():
        ast_tree = parse_ast(file_path)
        
        for node in ast.walk(ast_tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                current_docstring = ast.get_docstring(node)
                standardized_docstring = standardize_format(
                    current_docstring, DOCSTRING_TEMPLATE
                )
                update_docstring(node, standardized_docstring, file_path)
                
    return docstring_report
```

### Phase 3: Code Consolidation

#### 3.1 Duplicate Code Extraction
**Target Areas:**
- LLM response parsing logic (scattered across guardrails)
- Configuration handling patterns
- Error handling patterns

**Pseudocode:**
```python
def extract_duplicate_code():
    """Identify and extract duplicate code into reusable utilities"""
    
    # Extract common LLM parsing logic
    common_parsing_patterns = identify_duplicate_patterns([
        'psysafe/catalog/*/guardrail.py',
        'psysafe/evaluation/runner.py'
    ])
    
    for pattern in common_parsing_patterns:
        if pattern.similarity_score > 0.8:
            utility_function = create_utility_function(pattern)
            add_to_utils_module(utility_function)
            replace_duplicates_with_calls(pattern, utility_function)
    
    # Extract configuration patterns
    config_patterns = identify_config_duplicates()
    consolidate_configuration_handling(config_patterns)
    
    return consolidation_report
```

#### 3.2 Utility Module Reorganization
**Pseudocode:**
```python
def reorganize_utilities():
    """Reorganize utility functions into logical modules"""
    
    UTILITY_MODULES = {
        'psysafe/utils/llm_parsing.py': [
            'parse_llm_response', 'format_conversation_for_classification'
        ],
        'psysafe/utils/config_helpers.py': [
            'load_environment', 'validate_config'
        ],
        'psysafe/utils/validation.py': [
            'validate_request_format', 'sanitize_input'
        ]
    }
    
    # Move functions to appropriate modules
    for target_module, functions in UTILITY_MODULES.items():
        create_utility_module(target_module)
        for func_name in functions:
            source_location = find_function_location(func_name)
            move_function(source_location, target_module, func_name)
            update_imports_across_codebase(func_name, target_module)
    
    return reorganization_report
```

### Phase 4: Standards Enforcement

#### 4.1 Code Formatting with Ruff
**Pseudocode:**
```python
def enforce_formatting_standards():
    """Apply consistent formatting using ruff"""
    
    # Update ruff configuration
    ruff_config = {
        'line-length': 120,
        'select': ['E', 'F', 'W', 'I', 'UP', 'C90', 'N', 'D', 'S', 'B'],
        'ignore': ['D203', 'D212', 'D407', 'D416', 'E501'],
        'target-version': 'py39'
    }
    
    update_pyproject_toml(ruff_config)
    
    # Run formatting
    run_command(['ruff', 'format', '.'])
    run_command(['ruff', 'check', '--fix', '.'])
    
    return formatting_report
```

#### 4.2 Import Organization
**Pseudocode:**
```python
def organize_imports():
    """Standardize import organization using ruff's isort"""
    
    IMPORT_ORDER = [
        'standard_library',
        'third_party', 
        'first_party',
        'local_folder'
    ]
    
    for file_path in get_python_files():
        imports = extract_imports(file_path)
        organized_imports = sort_imports(imports, IMPORT_ORDER)
        replace_imports(file_path, organized_imports)
    
    return import_organization_report
```

#### 4.3 Naming Convention Enforcement
**Pseudocode:**
```python
def enforce_naming_conventions():
    """Ensure consistent naming across the codebase"""
    
    NAMING_RULES = {
        'functions': 'snake_case',
        'variables': 'snake_case', 
        'classes': 'PascalCase',
        'constants': 'UPPER_SNAKE_CASE',
        'modules': 'snake_case'
    }
    
    violations = []
    
    for file_path in get_python_files():
        ast_tree = parse_ast(file_path)
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                if not is_snake_case(node.name):
                    violations.append(create_violation(node, 'function', file_path))
            elif isinstance(node, ast.ClassDef):
                if not is_pascal_case(node.name):
                    violations.append(create_violation(node, 'class', file_path))
    
    return naming_violations_report(violations)
```

### Phase 5: Dependency Cleanup

#### 5.1 Unused Dependency Removal
**Pseudocode:**
```python
def cleanup_dependencies():
    """Remove unused dependencies from pyproject.toml"""
    
    current_deps = extract_dependencies('pyproject.toml')
    used_imports = scan_all_imports()
    
    unused_deps = []
    for dep in current_deps:
        dep_name = extract_package_name(dep)
        if not is_dependency_used(dep_name, used_imports):
            unused_deps.append(dep)
    
    # Verify before removal
    for dep in unused_deps:
        if confirm_safe_to_remove(dep):
            remove_dependency('pyproject.toml', dep)
    
    return dependency_cleanup_report
```

#### 5.2 Configuration File Cleanup
**Pseudocode:**
```python
def cleanup_configuration_files():
    """Clean and optimize configuration files"""
    
    CONFIG_FILES = [
        'pyproject.toml',
        '.gitignore', 
        '.python-version'
    ]
    
    for config_file in CONFIG_FILES:
        if file_exists(config_file):
            content = read_file(config_file)
            cleaned_content = remove_redundant_config(content)
            optimized_content = optimize_config_structure(cleaned_content)
            write_file(config_file, optimized_content)
    
    return config_cleanup_report
```

### Phase 6: Testing Integration

#### 6.1 Test Suite Execution Strategy
**Pseudocode:**
```python
def execute_test_suite_after_cleanup():
    """Run comprehensive tests after each cleanup phase"""
    
    TEST_PHASES = [
        'unit_tests',
        'integration_tests', 
        'example_validation',
        'documentation_tests'
    ]
    
    for phase in TEST_PHASES:
        test_results = run_test_phase(phase)
        
        if test_results.failed_count > 0:
            rollback_last_changes()
            analyze_test_failures(test_results)
            fix_breaking_changes()
            rerun_tests(phase)
        
        record_test_results(phase, test_results)
    
    return comprehensive_test_report
```

#### 6.2 Regression Prevention
**Pseudocode:**
```python
def prevent_regressions():
    """Ensure no functionality is broken during cleanup"""
    
    # Capture baseline functionality
    baseline_behavior = capture_baseline_behavior([
        'examples/01_vulnerability.py',
        'examples/02_suicide_prevention.py', 
        'examples/03_complaints_handling.py',
        'examples/04_mental_health_support.py',
        'examples/05_pii_protection.py'
    ])
    
    # After each cleanup phase
    for cleanup_phase in CLEANUP_PHASES:
        execute_cleanup_phase(cleanup_phase)
        
        current_behavior = capture_current_behavior()
        
        if not behaviors_match(baseline_behavior, current_behavior):
            rollback_changes()
            investigate_regression()
            fix_and_retry()
    
    return regression_report
```

### Phase 7: Version Control Strategy

#### 7.1 Atomic Commit Structure
**Pseudocode:**
```python
def create_atomic_commits():
    """Create focused commits for each cleanup category"""
    
    COMMIT_CATEGORIES = {
        'dead_code_removal': 'Remove commented code and unused functions',
        'documentation_refresh': 'Update documentation and docstrings',
        'code_consolidation': 'Extract duplicate code into utilities',
        'formatting_standards': 'Apply consistent code formatting',
        'dependency_cleanup': 'Remove unused dependencies',
        'test_updates': 'Update and fix test suite'
    }
    
    for category, message in COMMIT_CATEGORIES.items():
        stage_changes_for_category(category)
        
        if has_staged_changes():
            commit_changes(message)
            tag_commit(f'cleanup-{category}')
            
    return commit_history_report
```

#### 7.2 Audit Trail Creation
**Pseudocode:**
```python
def create_audit_trail():
    """Document all changes for transparency"""
    
    AUDIT_LOG = {
        'timestamp': get_current_timestamp(),
        'cleanup_phases': [],
        'files_modified': [],
        'functions_removed': [],
        'dependencies_removed': [],
        'test_results': {}
    }
    
    for phase in CLEANUP_PHASES:
        phase_log = execute_and_log_phase(phase)
        AUDIT_LOG['cleanup_phases'].append(phase_log)
    
    write_audit_log('docs/cleanup_audit_log.json', AUDIT_LOG)
    generate_cleanup_summary_report(AUDIT_LOG)
    
    return audit_trail
```

## Implementation Modules

### Module 1: Dead Code Analyzer
```python
class DeadCodeAnalyzer:
    """Identifies and removes dead code safely"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.ast_cache = {}
        
    def find_commented_code_blocks(self) -> List[CodeBlock]:
        """Identify commented-out code blocks"""
        pass
        
    def find_unused_functions(self) -> List[FunctionDef]:
        """Find functions that are never called"""
        pass
        
    def find_unused_imports(self) -> List[ImportStatement]:
        """Find imports that are never used"""
        pass
        
    def safe_remove(self, items: List[CodeItem]) -> RemovalReport:
        """Safely remove items after validation"""
        pass
```

### Module 2: Documentation Refresher
```python
class DocumentationRefresher:
    """Updates and standardizes documentation"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.docstring_parser = DocstringParser()
        
    def update_readme(self) -> UpdateReport:
        """Update README.md with current information"""
        pass
        
    def standardize_docstrings(self) -> StandardizationReport:
        """Apply consistent docstring format"""
        pass
        
    def update_api_docs(self) -> DocumentationReport:
        """Generate updated API documentation"""
        pass
```

### Module 3: Code Consolidator
```python
class CodeConsolidator:
    """Extracts duplicate code into reusable utilities"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.similarity_threshold = 0.8
        
    def find_duplicate_patterns(self) -> List[DuplicatePattern]:
        """Identify duplicate code patterns"""
        pass
        
    def extract_to_utilities(self, patterns: List[DuplicatePattern]) -> ExtractionReport:
        """Extract patterns into utility functions"""
        pass
        
    def update_references(self, extractions: List[Extraction]) -> UpdateReport:
        """Update all references to use new utilities"""
        pass
```

### Module 4: Standards Enforcer
```python
class StandardsEnforcer:
    """Enforces coding standards and formatting"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.ruff_config = load_ruff_config()
        
    def apply_formatting(self) -> FormattingReport:
        """Apply consistent code formatting"""
        pass
        
    def organize_imports(self) -> ImportReport:
        """Standardize import organization"""
        pass
        
    def enforce_naming_conventions(self) -> NamingReport:
        """Check and fix naming convention violations"""
        pass
```

### Module 5: Test Validator
```python
class TestValidator:
    """Ensures tests pass after each cleanup phase"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.baseline_results = None
        
    def capture_baseline(self) -> TestBaseline:
        """Capture baseline test behavior"""
        pass
        
    def run_test_suite(self) -> TestResults:
        """Execute comprehensive test suite"""
        pass
        
    def validate_no_regressions(self, results: TestResults) -> ValidationReport:
        """Ensure no functionality was broken"""
        pass
```

## TDD Anchors

### Test 1: Dead Code Removal Validation
```python
def test_dead_code_removal():
    """Verify dead code is removed without breaking functionality"""
    
    # Arrange
    analyzer = DeadCodeAnalyzer(project_root)
    baseline_functionality = capture_functionality()
    
    # Act
    removal_report = analyzer.remove_dead_code()
    
    # Assert
    assert removal_report.items_removed > 0
    assert removal_report.errors == []
    assert functionality_preserved(baseline_functionality)
```

### Test 2: Documentation Consistency
```python
def test_documentation_consistency():
    """Verify all documentation is updated and consistent"""
    
    # Arrange
    refresher = DocumentationRefresher(project_root)
    
    # Act
    doc_report = refresher.refresh_all_documentation()
    
    # Assert
    assert doc_report.outdated_docs == []
    assert doc_report.missing_docstrings == []
    assert all_examples_work()
```

### Test 3: No Duplicate Code
```python
def test_no_duplicate_code():
    """Verify duplicate code has been extracted"""
    
    # Arrange
    consolidator = CodeConsolidator(project_root)
    
    # Act
    duplicate_patterns = consolidator.find_duplicate_patterns()
    
    # Assert
    assert len(duplicate_patterns) == 0
    assert all_utilities_are_used()
```

### Test 4: Standards Compliance
```python
def test_standards_compliance():
    """Verify code meets all formatting and naming standards"""
    
    # Arrange
    enforcer = StandardsEnforcer(project_root)
    
    # Act
    compliance_report = enforcer.check_compliance()
    
    # Assert
    assert compliance_report.violations == []
    assert compliance_report.formatting_issues == []
    assert compliance_report.naming_violations == []
```

### Test 5: Functionality Preservation
```python
def test_functionality_preservation():
    """Verify all original functionality is preserved"""
    
    # Arrange
    validator = TestValidator(project_root)
    baseline = validator.capture_baseline()
    
    # Act - Run all cleanup phases
    cleanup_pipeline = CleanupPipeline(project_root)
    cleanup_pipeline.execute_all_phases()
    
    # Assert
    current_results = validator.run_test_suite()
    assert validator.validate_no_regressions(current_results).passed
    assert all_examples_still_work()
```

## Success Criteria

1. **Code Quality Metrics:**
   - Zero commented-out code blocks
   - No unused imports or functions
   - 100% docstring coverage for public APIs
   - Consistent formatting (ruff compliance)

2. **Functionality Preservation:**
   - All existing tests pass
   - All examples execute successfully
   - No breaking changes to public APIs

3. **Maintainability Improvements:**
   - Reduced code duplication (< 5% similarity)
   - Modular utility functions
   - Clear separation of concerns

4. **Documentation Quality:**
   - Up-to-date README and docs
   - Consistent docstring format
   - Working code examples

5. **Version Control Hygiene:**
   - Atomic commits for each cleanup category
   - Clear commit messages
   - Complete audit trail

## Risk Mitigation

1. **Backup Strategy:** Create full repository backup before starting
2. **Incremental Approach:** Execute one phase at a time with validation
3. **Rollback Plan:** Maintain ability to revert any changes
4. **Test Coverage:** Run comprehensive tests after each phase
5. **Review Process:** Manual review of all automated changes


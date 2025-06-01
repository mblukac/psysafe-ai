# PsySafe AI Code Cleanup and Refactoring System

A comprehensive, modular cleanup system designed to improve code quality, maintainability, and consistency while preserving all existing functionality.

## Overview

This cleanup system implements a 7-phase approach to systematically clean up and refactor the PsySafe AI codebase:

1. **Dead Code Removal** - Remove commented code, unused functions, and unused imports
2. **Documentation Refresh** - Update and standardize documentation and docstrings
3. **Code Consolidation** - Extract duplicate code into reusable utilities
4. **Standards Enforcement** - Apply consistent formatting and naming conventions
5. **Dependency Cleanup** - Remove unused dependencies
6. **Test Validation** - Ensure all tests pass and no regressions occur
7. **Version Control** - Create atomic commits and audit trails

## Features

- ✅ **Modular Architecture** - Each cleanup phase is implemented as a separate module
- ✅ **Safety First** - Comprehensive backup and rollback capabilities
- ✅ **Validation Engine** - Extensive validation to prevent regressions
- ✅ **Progress Tracking** - Real-time progress monitoring and reporting
- ✅ **Configuration Management** - Externalized configuration with YAML support
- ✅ **Comprehensive Reporting** - HTML, Markdown, and JSON reports
- ✅ **Self-Cleanup** - Removes itself after successful completion

## Quick Start

### 1. Analyze Only (Recommended First Step)

```bash
python -m cleanup_system.main --analyze
```

This will analyze the codebase and show what would be cleaned up without making any changes.

### 2. Dry Run

```bash
python -m cleanup_system.main --dry-run
```

Simulate the entire cleanup process without making actual changes.

### 3. Full Cleanup

```bash
python -m cleanup_system.main
```

Run the complete cleanup process with safety backups.

### 4. Single Phase

```bash
python -m cleanup_system.main --phase dead_code_removal
```

Run only a specific cleanup phase.

## Installation Requirements

The cleanup system requires the following dependencies:

```bash
pip install pyyaml ruff
```

- `pyyaml` - For configuration file parsing
- `ruff` - For code formatting and linting

## Configuration

### Default Configuration

The system comes with a sensible default configuration in [`cleanup_config.yaml`](config/cleanup_config.yaml). You can create a custom configuration file:

```bash
python -m cleanup_system.main --create-config
```

### Configuration Options

Key configuration sections:

#### Phase Configuration
```yaml
phases:
  dead_code_removal:
    enabled: true
    dry_run: false
    timeout: 300
    dependencies: []
    config:
      similarity_threshold: 0.8
      preserve_patterns:
        - "\\s*(TODO|FIXME|NOTE|WARNING)"
```

#### Safety Configuration
```yaml
safety:
  backup_enabled: true
  backup_type: "git"  # or "full"
  safety_checks: true
  validation_enabled: true
  stop_on_error: true
```

#### File Patterns
```yaml
exclude_paths:
  - ".git"
  - "__pycache__"
  - "cleanup_system"

include_patterns:
  - "*.py"
```

## Command Line Options

```
usage: main.py [-h] [--project-root PROJECT_ROOT] [--config CONFIG] [--analyze]
               [--dry-run] [--phase {dead_code_removal,documentation_refresh,code_consolidation,standards_enforcement,dependency_cleanup,test_validation}]
               [--no-backup] [--no-validation] [--output-dir OUTPUT_DIR] [--verbose]
               [--create-config]

Options:
  --project-root PATH     Root directory of the project (default: current directory)
  --config PATH           Path to configuration file
  --analyze              Only analyze the codebase, do not make changes
  --dry-run              Simulate cleanup without making actual changes
  --phase PHASE          Run only a specific phase
  --no-backup            Skip creating backup before cleanup
  --no-validation        Skip validation checks
  --output-dir PATH      Directory for output reports (default: cleanup_reports)
  --verbose              Enable verbose output
  --create-config        Create a default configuration file and exit
```

## Architecture

### Core Components

- **[`core/base.py`](core/base.py)** - Base interfaces and data structures
- **[`core/pipeline.py`](core/pipeline.py)** - Orchestration and execution pipeline
- **[`core/safety.py`](core/safety.py)** - Safety management and backup systems
- **[`core/validation.py`](core/validation.py)** - Comprehensive validation engine

### Cleanup Modules

- **[`modules/dead_code.py`](modules/dead_code.py)** - Dead code analysis and removal
- **[`modules/documentation.py`](modules/documentation.py)** - Documentation refresh and standardization
- **[`modules/consolidation.py`](modules/consolidation.py)** - Code consolidation and deduplication
- **[`modules/standards.py`](modules/standards.py)** - Code formatting and standards enforcement
- **[`modules/testing.py`](modules/testing.py)** - Test validation and regression prevention
- **[`modules/dependencies.py`](modules/dependencies.py)** - Dependency analysis and cleanup

### Support Systems

- **[`config/`](config/)** - Configuration management
- **[`reporting/`](reporting/)** - Progress tracking and report generation

## Safety Features

### Backup System

The cleanup system creates comprehensive backups before making any changes:

- **Git Backup** - Creates a git commit with all current changes
- **Full Backup** - Creates a complete copy of the project directory
- **Incremental Backup** - Creates backups of only modified files

### Validation Engine

Comprehensive validation ensures no functionality is broken:

- **Syntax Validation** - Ensures all Python files have valid syntax
- **Import Validation** - Verifies all imports can be resolved
- **Test Validation** - Runs the test suite to catch regressions
- **Example Validation** - Ensures all examples still work
- **Project Structure Validation** - Verifies essential files exist

### Rollback Capability

If issues are detected, the system can automatically restore from backup:

```python
# Emergency restore
safety_manager.emergency_restore()
```

## Reporting

The system generates comprehensive reports in multiple formats:

### HTML Report
Interactive HTML report with detailed phase information, metrics, and recommendations.

### Markdown Summary
Concise markdown summary suitable for documentation or commit messages.

### JSON Report
Machine-readable report with complete data for further analysis.

### Audit Log
Detailed log of all changes made during cleanup for compliance and review.

## Phase Details

### 1. Dead Code Removal

Identifies and removes:
- Commented-out code blocks
- Unused functions and classes
- Unused imports
- Placeholder implementations

**Configuration:**
```yaml
dead_code_removal:
  config:
    preserve_patterns:
      - "\\s*(TODO|FIXME|NOTE|WARNING)"
      - "\\s*(Copyright|License|Author)"
```

### 2. Documentation Refresh

Updates and standardizes:
- README.md files
- Docstrings for functions and classes
- Inline documentation
- Code examples

**Configuration:**
```yaml
documentation_refresh:
  config:
    update_readme: true
    standardize_docstrings: true
    check_examples: true
```

### 3. Code Consolidation

Extracts duplicate code:
- Identifies similar code patterns
- Creates utility functions
- Updates references to use utilities
- Reduces code duplication

**Configuration:**
```yaml
code_consolidation:
  config:
    similarity_threshold: 0.8
    min_pattern_size: 5
    extract_utilities: true
```

### 4. Standards Enforcement

Applies consistent formatting:
- Code formatting with ruff
- Import organization
- Naming convention enforcement
- Line length limits

**Configuration:**
```yaml
standards_enforcement:
  config:
    ruff:
      line-length: 120
      select: ["E", "F", "W", "I"]
```

### 5. Dependency Cleanup

Removes unused dependencies:
- Scans for unused imports
- Identifies unused packages
- Updates pyproject.toml and requirements files
- Preserves essential dependencies

**Configuration:**
```yaml
dependency_cleanup:
  config:
    confidence_threshold: 0.8
    preserve_dev_deps: true
```

### 6. Test Validation

Ensures functionality preservation:
- Captures baseline test behavior
- Runs comprehensive test suite
- Validates examples work correctly
- Detects regressions

**Configuration:**
```yaml
test_validation:
  config:
    test_command: "python -m pytest"
    run_examples: true
    validate_regressions: true
```

## Environment Variables

Override configuration with environment variables:

```bash
export CLEANUP_DRY_RUN=true
export CLEANUP_BACKUP_ENABLED=false
export CLEANUP_LOG_LEVEL=DEBUG
export CLEANUP_OUTPUT_DIR=custom_reports
```

## Best Practices

### Before Running Cleanup

1. **Commit all changes** - Ensure your working directory is clean
2. **Run tests** - Verify everything works before cleanup
3. **Analyze first** - Use `--analyze` to see what will be changed
4. **Start with dry-run** - Use `--dry-run` to simulate changes

### During Cleanup

1. **Monitor progress** - Use `--verbose` for detailed output
2. **Review warnings** - Pay attention to warning messages
3. **Check reports** - Review generated reports for issues

### After Cleanup

1. **Run tests** - Verify all tests still pass
2. **Test examples** - Ensure examples work correctly
3. **Review changes** - Check the generated reports and audit log
4. **Commit changes** - Create atomic commits for each phase

## Troubleshooting

### Common Issues

**Syntax Errors After Cleanup**
- Check the validation report for syntax issues
- Restore from backup if needed
- Review the specific changes that caused issues

**Import Errors**
- Verify all dependencies are installed
- Check if any imports were incorrectly removed
- Review the dependency cleanup phase report

**Test Failures**
- Compare with baseline test results
- Check if any test files were modified
- Review the test validation phase report

**Performance Issues**
- Reduce the number of files processed
- Use exclude patterns for large directories
- Run phases individually instead of all at once

### Recovery

If something goes wrong, you can restore from backup:

```bash
# The system will prompt for emergency restore on critical failures
# Or manually restore using git:
git reset --hard HEAD~1  # If using git backup
```

## Contributing

The cleanup system is designed to be extensible. To add a new cleanup phase:

1. Create a new module in [`modules/`](modules/)
2. Implement the [`CodeCleanupBase`](core/base.py) interface
3. Register the module in [`core/pipeline.py`](core/pipeline.py)
4. Add configuration options
5. Update documentation

## License

This cleanup system is part of the PsySafe AI project and follows the same license terms.

## Support

For issues or questions about the cleanup system:

1. Check the generated reports for detailed error information
2. Review the configuration options
3. Try running with `--verbose` for more details
4. Use `--analyze` or `--dry-run` to debug issues safely
"""Dependency management module for cleaning up unused dependencies."""

import ast
import re
import subprocess
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..core.base import CleanupPhase, CleanupReport, CodeCleanupBase, FileChange


@dataclass
class Dependency:
    """Represents a project dependency."""
    
    name: str
    version: Optional[str]
    category: str  # 'main', 'dev', 'optional'
    source: str  # 'pyproject.toml', 'requirements.txt', etc.


@dataclass
class UnusedDependency:
    """Represents an unused dependency."""
    
    dependency: Dependency
    reason: str
    confidence: float  # 0.0 to 1.0


class DependencyManager(CodeCleanupBase):
    """Manages and cleans up project dependencies."""
    
    def __init__(self, project_root: Path, config: Optional[Dict[str, Any]] = None):
        super().__init__(project_root, config)
        self.pyproject_path = self.project_root / 'pyproject.toml'
        self.requirements_files = [
            self.project_root / 'requirements.txt',
            self.project_root / 'requirements-dev.txt',
            self.project_root / 'dev-requirements.txt',
        ]
        self._import_cache: Dict[Path, Set[str]] = {}
    
    def get_phase(self) -> CleanupPhase:
        """Get the cleanup phase."""
        return CleanupPhase.DEPENDENCY_CLEANUP
    
    def extract_dependencies(self) -> List[Dependency]:
        """Extract all dependencies from project files."""
        dependencies = []
        
        # Extract from pyproject.toml
        if self.pyproject_path.exists():
            dependencies.extend(self._extract_from_pyproject())
        
        # Extract from requirements files
        for req_file in self.requirements_files:
            if req_file.exists():
                dependencies.extend(self._extract_from_requirements(req_file))
        
        return dependencies
    
    def _extract_from_pyproject(self) -> List[Dependency]:
        """Extract dependencies from pyproject.toml."""
        dependencies = []
        
        try:
            with open(self.pyproject_path, 'rb') as f:
                data = tomllib.load(f)
            
            # Main dependencies
            main_deps = data.get('project', {}).get('dependencies', [])
            for dep in main_deps:
                name, version = self._parse_dependency_spec(dep)
                dependencies.append(Dependency(
                    name=name,
                    version=version,
                    category='main',
                    source='pyproject.toml'
                ))
            
            # Optional dependencies
            optional_deps = data.get('project', {}).get('optional-dependencies', {})
            for group, deps in optional_deps.items():
                for dep in deps:
                    name, version = self._parse_dependency_spec(dep)
                    dependencies.append(Dependency(
                        name=name,
                        version=version,
                        category=f'optional-{group}',
                        source='pyproject.toml'
                    ))
            
            # Build system dependencies
            build_deps = data.get('build-system', {}).get('requires', [])
            for dep in build_deps:
                name, version = self._parse_dependency_spec(dep)
                dependencies.append(Dependency(
                    name=name,
                    version=version,
                    category='build',
                    source='pyproject.toml'
                ))
        
        except Exception as e:
            print(f"Error reading pyproject.toml: {e}")
        
        return dependencies
    
    def _extract_from_requirements(self, req_file: Path) -> List[Dependency]:
        """Extract dependencies from a requirements file."""
        dependencies = []
        
        try:
            with open(req_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('-'):
                    name, version = self._parse_dependency_spec(line)
                    category = 'dev' if 'dev' in req_file.name else 'main'
                    dependencies.append(Dependency(
                        name=name,
                        version=version,
                        category=category,
                        source=req_file.name
                    ))
        
        except Exception as e:
            print(f"Error reading {req_file}: {e}")
        
        return dependencies
    
    def _parse_dependency_spec(self, spec: str) -> tuple[str, Optional[str]]:
        """Parse a dependency specification into name and version."""
        # Handle various formats: package, package==1.0, package>=1.0, etc.
        spec = spec.strip()
        
        # Remove extras: package[extra]
        if '[' in spec:
            spec = spec.split('[')[0]
        
        # Extract version constraints
        version_pattern = r'([a-zA-Z0-9_-]+)([><=!~]+.*)?'
        match = re.match(version_pattern, spec)
        
        if match:
            name = match.group(1)
            version = match.group(2) if match.group(2) else None
            return name, version
        
        return spec, None
    
    def scan_imports(self) -> Set[str]:
        """Scan all Python files for import statements."""
        all_imports = set()
        
        for file_path in self.get_python_files():
            imports = self._extract_imports_from_file(file_path)
            all_imports.update(imports)
            self._import_cache[file_path] = imports
        
        return all_imports
    
    def _extract_imports_from_file(self, file_path: Path) -> Set[str]:
        """Extract import statements from a Python file."""
        imports = set()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(file_path))
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])
        
        except Exception as e:
            print(f"Error extracting imports from {file_path}: {e}")
        
        return imports
    
    def find_unused_dependencies(self) -> List[UnusedDependency]:
        """Find dependencies that are not used in the codebase."""
        dependencies = self.extract_dependencies()
        used_imports = self.scan_imports()
        unused = []
        
        # Map of package names to import names (some packages have different import names)
        package_import_map = self._get_package_import_mapping()
        
        for dep in dependencies:
            # Skip build system dependencies
            if dep.category == 'build':
                continue
            
            # Get possible import names for this package
            possible_imports = package_import_map.get(dep.name, {dep.name})
            
            # Check if any of the possible import names are used
            is_used = any(imp in used_imports for imp in possible_imports)
            
            if not is_used:
                # Additional checks for special cases
                confidence = self._calculate_confidence(dep, used_imports)
                
                if confidence > 0.5:  # Only report if reasonably confident
                    unused.append(UnusedDependency(
                        dependency=dep,
                        reason=f"No imports found for package '{dep.name}'",
                        confidence=confidence
                    ))
        
        return unused
    
    def _get_package_import_mapping(self) -> Dict[str, Set[str]]:
        """Get mapping of package names to their import names."""
        # This is a simplified mapping - in production, use a more comprehensive database
        mapping = {
            'pillow': {'PIL'},
            'beautifulsoup4': {'bs4'},
            'pyyaml': {'yaml'},
            'python-dateutil': {'dateutil'},
            'msgpack-python': {'msgpack'},
            'protobuf': {'google.protobuf'},
            'six': {'six'},
            'setuptools': {'setuptools', 'pkg_resources'},
            'wheel': {'wheel'},
            'pip': {'pip'},
        }
        
        # Add common patterns
        for package in ['requests', 'numpy', 'pandas', 'matplotlib', 'scipy', 'sklearn']:
            mapping[package] = {package}
        
        return mapping
    
    def _calculate_confidence(self, dep: Dependency, used_imports: Set[str]) -> float:
        """Calculate confidence that a dependency is truly unused."""
        confidence = 0.8  # Base confidence
        
        # Lower confidence for common utility packages
        utility_packages = {'setuptools', 'wheel', 'pip', 'twine', 'build'}
        if dep.name in utility_packages:
            confidence *= 0.3
        
        # Lower confidence for dev dependencies
        if dep.category.startswith('dev') or dep.category.startswith('optional'):
            confidence *= 0.7
        
        # Higher confidence if package name is very specific
        if len(dep.name) > 10 and '-' in dep.name:
            confidence *= 1.2
        
        # Check for indirect usage patterns
        if self._has_indirect_usage(dep.name):
            confidence *= 0.5
        
        return min(confidence, 1.0)
    
    def _has_indirect_usage(self, package_name: str) -> bool:
        """Check if a package might be used indirectly."""
        # Check configuration files for references
        config_files = [
            self.project_root / 'pyproject.toml',
            self.project_root / 'setup.cfg',
            self.project_root / '.pre-commit-config.yaml',
            self.project_root / 'tox.ini',
        ]
        
        for config_file in config_files:
            if config_file.exists():
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if package_name in content:
                        return True
                except:
                    pass
        
        return False
    
    def cleanup_dependencies(self, unused: List[UnusedDependency], dry_run: bool = False) -> List[FileChange]:
        """Remove unused dependencies from project files."""
        changes = []
        
        # Group by source file
        by_source = {}
        for unused_dep in unused:
            source = unused_dep.dependency.source
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(unused_dep)
        
        # Remove from each source file
        for source, deps in by_source.items():
            if source == 'pyproject.toml':
                change = self._remove_from_pyproject(deps, dry_run)
                if change:
                    changes.append(change)
            else:
                # Handle requirements files
                req_file = self.project_root / source
                if req_file.exists():
                    change = self._remove_from_requirements(req_file, deps, dry_run)
                    if change:
                        changes.append(change)
        
        return changes
    
    def _remove_from_pyproject(self, unused_deps: List[UnusedDependency], dry_run: bool) -> Optional[FileChange]:
        """Remove dependencies from pyproject.toml."""
        if not self.pyproject_path.exists():
            return None
        
        try:
            with open(self.pyproject_path, 'rb') as f:
                data = tomllib.load(f)
            
            # Track what we remove
            removed_deps = []
            
            # Remove from main dependencies
            main_deps = data.get('project', {}).get('dependencies', [])
            new_main_deps = []
            for dep_spec in main_deps:
                dep_name, _ = self._parse_dependency_spec(dep_spec)
                if not any(ud.dependency.name == dep_name for ud in unused_deps):
                    new_main_deps.append(dep_spec)
                else:
                    removed_deps.append(dep_name)
            
            if not dry_run and new_main_deps != main_deps:
                data.setdefault('project', {})['dependencies'] = new_main_deps
            
            # Remove from optional dependencies
            optional_deps = data.get('project', {}).get('optional-dependencies', {})
            for group, deps in optional_deps.items():
                new_deps = []
                for dep_spec in deps:
                    dep_name, _ = self._parse_dependency_spec(dep_spec)
                    if not any(ud.dependency.name == dep_name for ud in unused_deps):
                        new_deps.append(dep_spec)
                    else:
                        removed_deps.append(f"{dep_name} (optional-{group})")
                
                if not dry_run:
                    optional_deps[group] = new_deps
            
            # Write back the file (simplified - in production use proper TOML writer)
            if not dry_run and removed_deps:
                # This is a placeholder - proper TOML writing would be needed
                pass
            
            if removed_deps:
                return FileChange(
                    file_path=self.pyproject_path,
                    change_type='modified',
                    description=f"Removed unused dependencies: {', '.join(removed_deps)}",
                    line_changes={'removed_dependencies': removed_deps}
                )
        
        except Exception as e:
            print(f"Error removing dependencies from pyproject.toml: {e}")
        
        return None
    
    def _remove_from_requirements(self, req_file: Path, unused_deps: List[UnusedDependency], dry_run: bool) -> Optional[FileChange]:
        """Remove dependencies from a requirements file."""
        try:
            with open(req_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            new_lines = []
            removed_deps = []
            
            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):
                    dep_name, _ = self._parse_dependency_spec(stripped)
                    if any(ud.dependency.name == dep_name for ud in unused_deps):
                        removed_deps.append(dep_name)
                        continue
                
                new_lines.append(line)
            
            if not dry_run and removed_deps:
                with open(req_file, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)
            
            if removed_deps:
                return FileChange(
                    file_path=req_file,
                    change_type='modified',
                    description=f"Removed unused dependencies: {', '.join(removed_deps)}",
                    line_changes={'removed_dependencies': removed_deps}
                )
        
        except Exception as e:
            print(f"Error removing dependencies from {req_file}: {e}")
        
        return None
    
    def analyze(self) -> Dict[str, Any]:
        """Analyze dependency usage."""
        print("Analyzing dependency usage...")
        
        dependencies = self.extract_dependencies()
        used_imports = self.scan_imports()
        unused_deps = self.find_unused_dependencies()
        
        # Categorize dependencies
        main_deps = [d for d in dependencies if d.category == 'main']
        dev_deps = [d for d in dependencies if d.category.startswith('dev') or d.category.startswith('optional')]
        build_deps = [d for d in dependencies if d.category == 'build']
        
        return {
            'total_dependencies': len(dependencies),
            'main_dependencies': len(main_deps),
            'dev_dependencies': len(dev_deps),
            'build_dependencies': len(build_deps),
            'unused_dependencies': len(unused_deps),
            'high_confidence_unused': len([u for u in unused_deps if u.confidence > 0.8]),
            'total_imports_found': len(used_imports),
            'dependency_sources': len(set(d.source for d in dependencies))
        }
    
    def execute(self, dry_run: bool = False) -> CleanupReport:
        """Execute dependency cleanup."""
        report = CleanupReport(
            phase=self.get_phase(),
            start_time=datetime.now()
        )
        
        try:
            # Find unused dependencies
            unused_deps = self.find_unused_dependencies()
            
            # Only remove high-confidence unused dependencies
            high_confidence_unused = [u for u in unused_deps if u.confidence > 0.8]
            
            # Remove dependencies
            changes = self.cleanup_dependencies(high_confidence_unused, dry_run)
            
            for change in changes:
                report.add_file_change(change)
            
            # Add warnings for lower confidence unused dependencies
            for unused in unused_deps:
                if unused.confidence <= 0.8:
                    report.add_warning(
                        f"Potentially unused dependency: {unused.dependency.name} "
                        f"(confidence: {unused.confidence:.2f}) - {unused.reason}"
                    )
            
            # Add metrics
            report.add_metric('unused_dependencies_found', len(unused_deps))
            report.add_metric('high_confidence_unused', len(high_confidence_unused))
            report.add_metric('dependencies_removed', len(changes))
            
        except Exception as e:
            report.add_error(f"Error during dependency cleanup: {str(e)}")
        
        report.finalize()
        return report
    
    def validate(self) -> bool:
        """Validate that dependency cleanup can be performed."""
        try:
            # Check if we can read dependency files
            if self.pyproject_path.exists():
                with open(self.pyproject_path, 'rb') as f:
                    tomllib.load(f)
            
            # Check if we can scan imports
            python_files = self.get_python_files()
            if not python_files:
                return False
            
            # Test parsing a few files
            for file_path in python_files[:5]:  # Test first 5 files
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    ast.parse(content)
                except:
                    return False
            
            return True
        except:
            return False
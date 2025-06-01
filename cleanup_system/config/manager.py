"""Configuration manager for the cleanup system."""

import os
import yaml
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schemas import CleanupConfig, PhaseConfig, SafetyConfig


class ConfigurationManager:
    """Manages configuration for the cleanup system."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.config_dir = project_root / 'cleanup_system' / 'config'
        self.config_file = self.config_dir / 'cleanup_config.yaml'
        self._config: Optional[CleanupConfig] = None
    
    def load_config(self, config_path: Optional[Path] = None) -> CleanupConfig:
        """Load configuration from file or create default."""
        config_path = config_path or self.config_file
        
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                
                self._config = self._parse_config(config_data)
                print(f"Configuration loaded from {config_path}")
                
            except Exception as e:
                print(f"Error loading config from {config_path}: {e}")
                print("Using default configuration")
                self._config = self._create_default_config()
        else:
            print("No configuration file found, using defaults")
            self._config = self._create_default_config()
        
        return self._config
    
    def save_config(self, config: CleanupConfig, config_path: Optional[Path] = None) -> None:
        """Save configuration to file."""
        config_path = config_path or self.config_file
        
        # Ensure config directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary
        config_dict = asdict(config)
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            print(f"Configuration saved to {config_path}")
            
        except Exception as e:
            print(f"Error saving config to {config_path}: {e}")
    
    def get_config(self) -> CleanupConfig:
        """Get the current configuration."""
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def _parse_config(self, config_data: Dict[str, Any]) -> CleanupConfig:
        """Parse configuration data into CleanupConfig object."""
        # Parse phase configurations
        phases = []
        phases_data = config_data.get('phases', {})
        
        for phase_name, phase_data in phases_data.items():
            phase_config = PhaseConfig(
                name=phase_name,
                enabled=phase_data.get('enabled', True),
                dry_run=phase_data.get('dry_run', False),
                config=phase_data.get('config', {}),
                dependencies=phase_data.get('dependencies', []),
                timeout=phase_data.get('timeout')
            )
            phases.append(phase_config)
        
        # Parse safety configuration
        safety_data = config_data.get('safety', {})
        safety_config = SafetyConfig(
            backup_enabled=safety_data.get('backup_enabled', True),
            backup_type=safety_data.get('backup_type', 'git'),
            backup_dir=safety_data.get('backup_dir'),
            safety_checks=safety_data.get('safety_checks', True),
            validation_enabled=safety_data.get('validation_enabled', True),
            stop_on_error=safety_data.get('stop_on_error', True)
        )
        
        return CleanupConfig(
            project_name=config_data.get('project_name', 'psysafe-ai'),
            phases=phases,
            safety=safety_config,
            exclude_paths=config_data.get('exclude_paths', []),
            include_patterns=config_data.get('include_patterns', ['*.py']),
            parallel_execution=config_data.get('parallel_execution', False),
            max_workers=config_data.get('max_workers', 4),
            log_level=config_data.get('log_level', 'INFO'),
            output_dir=config_data.get('output_dir', 'cleanup_reports')
        )
    
    def _create_default_config(self) -> CleanupConfig:
        """Create default configuration."""
        default_phases = [
            PhaseConfig(
                name='dead_code_removal',
                enabled=True,
                dry_run=False,
                config={
                    'similarity_threshold': 0.8,
                    'min_pattern_size': 5,
                    'preserve_patterns': [
                        r"#\s*(TODO|FIXME|NOTE|WARNING|HACK|XXX)",
                        r"#\s*(Copyright|License|Author)",
                        r"#\s*(Example:|Usage:|Returns:|Args:|Raises:|See also:)",
                    ]
                },
                dependencies=[],
                timeout=300
            ),
            PhaseConfig(
                name='documentation_refresh',
                enabled=True,
                dry_run=False,
                config={
                    'update_readme': True,
                    'standardize_docstrings': True,
                    'check_examples': True
                },
                dependencies=['dead_code_removal'],
                timeout=180
            ),
            PhaseConfig(
                name='code_consolidation',
                enabled=True,
                dry_run=False,
                config={
                    'similarity_threshold': 0.8,
                    'min_pattern_size': 5,
                    'extract_utilities': True
                },
                dependencies=['dead_code_removal'],
                timeout=300
            ),
            PhaseConfig(
                name='standards_enforcement',
                enabled=True,
                dry_run=False,
                config={
                    'ruff': {
                        'line-length': 120,
                        'select': ['E', 'F', 'W', 'I', 'UP', 'C90', 'N', 'D', 'S', 'B'],
                        'ignore': ['D203', 'D212', 'D407', 'D416', 'E501'],
                        'target-version': 'py39'
                    },
                    'apply_formatting': True,
                    'organize_imports': True,
                    'check_naming': True
                },
                dependencies=['code_consolidation'],
                timeout=120
            ),
            PhaseConfig(
                name='dependency_cleanup',
                enabled=True,
                dry_run=False,
                config={
                    'confidence_threshold': 0.8,
                    'check_indirect_usage': True,
                    'preserve_dev_deps': True
                },
                dependencies=['dead_code_removal'],
                timeout=180
            ),
            PhaseConfig(
                name='test_validation',
                enabled=True,
                dry_run=False,
                config={
                    'test_command': 'python -m pytest',
                    'run_examples': True,
                    'capture_baseline': True,
                    'validate_regressions': True
                },
                dependencies=[
                    'dead_code_removal',
                    'code_consolidation', 
                    'standards_enforcement',
                    'dependency_cleanup'
                ],
                timeout=600
            )
        ]
        
        default_safety = SafetyConfig(
            backup_enabled=True,
            backup_type='git',
            backup_dir=None,
            safety_checks=True,
            validation_enabled=True,
            stop_on_error=True
        )
        
        return CleanupConfig(
            project_name='psysafe-ai',
            phases=default_phases,
            safety=default_safety,
            exclude_paths=[
                '.git', '__pycache__', '.pytest_cache', '.mypy_cache',
                '*.egg-info', '.venv', 'venv', 'env', '.env',
                'node_modules', 'dist', 'build', '.coverage',
                'cleanup_system'
            ],
            include_patterns=['*.py'],
            parallel_execution=False,
            max_workers=4,
            log_level='INFO',
            output_dir='cleanup_reports'
        )
    
    def create_default_config_file(self) -> None:
        """Create a default configuration file."""
        default_config = self._create_default_config()
        self.save_config(default_config)
    
    def update_phase_config(self, phase_name: str, updates: Dict[str, Any]) -> None:
        """Update configuration for a specific phase."""
        config = self.get_config()
        
        for phase in config.phases:
            if phase.name == phase_name:
                for key, value in updates.items():
                    if hasattr(phase, key):
                        setattr(phase, key, value)
                    elif key == 'config':
                        phase.config.update(value)
                break
        
        self.save_config(config)
    
    def get_phase_config(self, phase_name: str) -> Optional[PhaseConfig]:
        """Get configuration for a specific phase."""
        config = self.get_config()
        
        for phase in config.phases:
            if phase.name == phase_name:
                return phase
        
        return None
    
    def validate_config(self, config: CleanupConfig) -> List[str]:
        """Validate configuration and return any errors."""
        errors = []
        
        # Check phase dependencies
        phase_names = {phase.name for phase in config.phases}
        
        for phase in config.phases:
            for dep in phase.dependencies:
                if dep not in phase_names:
                    errors.append(f"Phase '{phase.name}' depends on unknown phase '{dep}'")
        
        # Check circular dependencies
        if self._has_circular_dependencies(config.phases):
            errors.append("Circular dependencies detected in phase configuration")
        
        # Validate paths
        if config.safety.backup_dir:
            backup_path = Path(config.safety.backup_dir)
            if not backup_path.parent.exists():
                errors.append(f"Backup directory parent does not exist: {backup_path.parent}")
        
        # Validate log level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if config.log_level not in valid_log_levels:
            errors.append(f"Invalid log level: {config.log_level}")
        
        return errors
    
    def _has_circular_dependencies(self, phases: List[PhaseConfig]) -> bool:
        """Check for circular dependencies in phase configuration."""
        # Create dependency graph
        graph = {}
        for phase in phases:
            graph[phase.name] = phase.dependencies
        
        # Check for cycles using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for phase_name in graph:
            if phase_name not in visited:
                if has_cycle(phase_name):
                    return True
        
        return False
    
    def get_environment_overrides(self) -> Dict[str, Any]:
        """Get configuration overrides from environment variables."""
        overrides = {}
        
        # Environment variable mapping
        env_mapping = {
            'CLEANUP_DRY_RUN': ('global_dry_run', lambda x: x.lower() == 'true'),
            'CLEANUP_BACKUP_ENABLED': ('safety.backup_enabled', lambda x: x.lower() == 'true'),
            'CLEANUP_BACKUP_TYPE': ('safety.backup_type', str),
            'CLEANUP_BACKUP_DIR': ('safety.backup_dir', str),
            'CLEANUP_STOP_ON_ERROR': ('safety.stop_on_error', lambda x: x.lower() == 'true'),
            'CLEANUP_LOG_LEVEL': ('log_level', str),
            'CLEANUP_MAX_WORKERS': ('max_workers', int),
            'CLEANUP_OUTPUT_DIR': ('output_dir', str),
        }
        
        for env_var, (config_path, converter) in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    converted_value = converter(value)
                    self._set_nested_value(overrides, config_path, converted_value)
                except (ValueError, TypeError) as e:
                    print(f"Warning: Invalid value for {env_var}: {value} ({e})")
        
        return overrides
    
    def _set_nested_value(self, dictionary: Dict[str, Any], path: str, value: Any) -> None:
        """Set a nested value in a dictionary using dot notation."""
        keys = path.split('.')
        current = dictionary
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def apply_overrides(self, config: CleanupConfig, overrides: Dict[str, Any]) -> CleanupConfig:
        """Apply configuration overrides to a config object."""
        # This is a simplified implementation
        # In production, implement proper nested override application
        
        if 'global_dry_run' in overrides:
            for phase in config.phases:
                phase.dry_run = overrides['global_dry_run']
        
        if 'log_level' in overrides:
            config.log_level = overrides['log_level']
        
        if 'max_workers' in overrides:
            config.max_workers = overrides['max_workers']
        
        if 'output_dir' in overrides:
            config.output_dir = overrides['output_dir']
        
        # Apply safety overrides
        safety_overrides = overrides.get('safety', {})
        for key, value in safety_overrides.items():
            if hasattr(config.safety, key):
                setattr(config.safety, key, value)
        
        return config
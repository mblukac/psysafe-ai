"""Configuration schemas for the cleanup system."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PhaseConfig:
    """Configuration for a cleanup phase."""
    
    name: str
    enabled: bool = True
    dry_run: bool = False
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    timeout: Optional[int] = None  # seconds


@dataclass
class SafetyConfig:
    """Safety configuration for cleanup operations."""
    
    backup_enabled: bool = True
    backup_type: str = 'git'  # 'git', 'full', 'incremental'
    backup_dir: Optional[str] = None
    safety_checks: bool = True
    validation_enabled: bool = True
    stop_on_error: bool = True


@dataclass
class CleanupConfig:
    """Main configuration for the cleanup system."""
    
    project_name: str
    phases: List[PhaseConfig]
    safety: SafetyConfig
    exclude_paths: List[str] = field(default_factory=list)
    include_patterns: List[str] = field(default_factory=lambda: ['*.py'])
    parallel_execution: bool = False
    max_workers: int = 4
    log_level: str = 'INFO'
    output_dir: str = 'cleanup_reports'
"""Safety management and backup components."""

import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import CleanupReport


@dataclass
class BackupInfo:
    """Information about a backup."""
    
    backup_id: str
    timestamp: datetime
    backup_path: Path
    original_path: Path
    backup_type: str  # 'full', 'incremental', 'git'
    size_bytes: int
    description: str


class BackupManager:
    """Manages backups for safe cleanup operations."""
    
    def __init__(self, project_root: Path, backup_dir: Optional[Path] = None):
        self.project_root = project_root
        self.backup_dir = backup_dir or (project_root.parent / f"{project_root.name}_backups")
        self.backups: List[BackupInfo] = []
    
    def create_full_backup(self, description: str = "Pre-cleanup backup") -> BackupInfo:
        """Create a full backup of the project."""
        timestamp = datetime.now()
        backup_id = f"backup_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        backup_path = self.backup_dir / backup_id
        
        print(f"Creating full backup: {backup_id}")
        
        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy the entire project
        shutil.copytree(
            self.project_root,
            backup_path,
            ignore=shutil.ignore_patterns(
                '.git', '__pycache__', '*.pyc', '.pytest_cache',
                '.mypy_cache', '*.egg-info', '.venv', 'venv',
                'node_modules', '.coverage'
            )
        )
        
        # Calculate backup size
        size_bytes = self._calculate_directory_size(backup_path)
        
        backup_info = BackupInfo(
            backup_id=backup_id,
            timestamp=timestamp,
            backup_path=backup_path,
            original_path=self.project_root,
            backup_type='full',
            size_bytes=size_bytes,
            description=description
        )
        
        self.backups.append(backup_info)
        print(f"Backup created: {backup_path} ({size_bytes / 1024 / 1024:.1f} MB)")
        
        return backup_info
    
    def create_git_backup(self, description: str = "Pre-cleanup commit") -> Optional[BackupInfo]:
        """Create a git commit as backup."""
        if not (self.project_root / '.git').exists():
            print("No git repository found, skipping git backup")
            return None
        
        try:
            # Check if there are changes to commit
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if not result.stdout.strip():
                print("No changes to commit for git backup")
                return None
            
            # Add all changes
            subprocess.run(
                ['git', 'add', '.'],
                check=True,
                cwd=self.project_root
            )
            
            # Create commit
            commit_message = f"[CLEANUP BACKUP] {description}"
            result = subprocess.run(
                ['git', 'commit', '-m', commit_message],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                # Get commit hash
                hash_result = subprocess.run(
                    ['git', 'rev-parse', 'HEAD'],
                    capture_output=True,
                    text=True,
                    cwd=self.project_root
                )
                
                commit_hash = hash_result.stdout.strip()
                
                backup_info = BackupInfo(
                    backup_id=f"git_{commit_hash[:8]}",
                    timestamp=datetime.now(),
                    backup_path=Path(commit_hash),
                    original_path=self.project_root,
                    backup_type='git',
                    size_bytes=0,  # Git doesn't store size this way
                    description=f"{description} (commit: {commit_hash[:8]})"
                )
                
                self.backups.append(backup_info)
                print(f"Git backup created: {commit_hash[:8]}")
                
                return backup_info
        
        except subprocess.CalledProcessError as e:
            print(f"Git backup failed: {e}")
        
        return None
    
    def restore_backup(self, backup_info: BackupInfo) -> bool:
        """Restore from a backup."""
        print(f"Restoring backup: {backup_info.backup_id}")
        
        try:
            if backup_info.backup_type == 'git':
                # Git restore
                subprocess.run(
                    ['git', 'reset', '--hard', str(backup_info.backup_path)],
                    check=True,
                    cwd=self.project_root
                )
                print(f"Restored from git commit: {backup_info.backup_path}")
                return True
            
            elif backup_info.backup_type == 'full':
                # Full restore
                if backup_info.backup_path.exists():
                    # Remove current project (except .git)
                    for item in self.project_root.iterdir():
                        if item.name != '.git':
                            if item.is_dir():
                                shutil.rmtree(item)
                            else:
                                item.unlink()
                    
                    # Copy backup back
                    for item in backup_info.backup_path.iterdir():
                        if item.is_dir():
                            shutil.copytree(item, self.project_root / item.name)
                        else:
                            shutil.copy2(item, self.project_root / item.name)
                    
                    print(f"Restored from full backup: {backup_info.backup_path}")
                    return True
                else:
                    print(f"Backup path not found: {backup_info.backup_path}")
                    return False
        
        except Exception as e:
            print(f"Restore failed: {e}")
            return False
        
        return False
    
    def cleanup_old_backups(self, keep_count: int = 5) -> None:
        """Clean up old backups, keeping only the most recent ones."""
        if len(self.backups) <= keep_count:
            return
        
        # Sort by timestamp
        sorted_backups = sorted(self.backups, key=lambda b: b.timestamp, reverse=True)
        
        # Remove old backups
        for backup in sorted_backups[keep_count:]:
            if backup.backup_type == 'full' and backup.backup_path.exists():
                shutil.rmtree(backup.backup_path)
                print(f"Removed old backup: {backup.backup_id}")
        
        # Keep only recent backups in memory
        self.backups = sorted_backups[:keep_count]
    
    def _calculate_directory_size(self, path: Path) -> int:
        """Calculate the total size of a directory."""
        total_size = 0
        for file_path in path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    def get_backup_summary(self) -> Dict[str, Any]:
        """Get a summary of all backups."""
        total_size = sum(backup.size_bytes for backup in self.backups)
        
        return {
            'total_backups': len(self.backups),
            'total_size_mb': total_size / 1024 / 1024,
            'backup_types': {
                'full': len([b for b in self.backups if b.backup_type == 'full']),
                'git': len([b for b in self.backups if b.backup_type == 'git']),
            },
            'latest_backup': self.backups[-1].backup_id if self.backups else None,
            'backup_directory': str(self.backup_dir)
        }


class SafetyManager:
    """Manages safety operations for cleanup."""
    
    def __init__(self, project_root: Path, config: Optional[Dict[str, Any]] = None):
        self.project_root = project_root
        self.config = config or {}
        self.backup_manager = BackupManager(
            project_root,
            config.get('backup_dir') if config else None
        )
        self.safety_checks_enabled = config.get('safety_checks', True) if config else True
    
    def pre_cleanup_safety_check(self) -> bool:
        """Perform safety checks before cleanup."""
        print("Performing pre-cleanup safety checks...")
        
        checks = [
            self._check_git_status,
            self._check_uncommitted_changes,
            self._check_disk_space,
            self._check_file_permissions,
            self._check_running_processes,
        ]
        
        for check in checks:
            if not check():
                return False
        
        print("All safety checks passed")
        return True
    
    def _check_git_status(self) -> bool:
        """Check git repository status."""
        if not (self.project_root / '.git').exists():
            print("Warning: No git repository found")
            return True
        
        try:
            # Check if we're in a clean state
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.stdout.strip():
                print("Warning: Uncommitted changes detected")
                # This is a warning, not a failure
            
            return True
        
        except subprocess.CalledProcessError:
            print("Warning: Could not check git status")
            return True
    
    def _check_uncommitted_changes(self) -> bool:
        """Check for uncommitted changes."""
        # This is handled in _check_git_status
        return True
    
    def _check_disk_space(self) -> bool:
        """Check available disk space."""
        try:
            stat = shutil.disk_usage(self.project_root)
            free_gb = stat.free / (1024 ** 3)
            
            if free_gb < 1.0:  # Less than 1GB free
                print(f"Error: Insufficient disk space ({free_gb:.1f} GB free)")
                return False
            
            if free_gb < 5.0:  # Less than 5GB free
                print(f"Warning: Low disk space ({free_gb:.1f} GB free)")
            
            return True
        
        except Exception as e:
            print(f"Warning: Could not check disk space: {e}")
            return True
    
    def _check_file_permissions(self) -> bool:
        """Check file permissions."""
        try:
            # Test write permissions by creating a temporary file
            test_file = self.project_root / '.cleanup_permission_test'
            test_file.touch()
            test_file.unlink()
            return True
        
        except Exception as e:
            print(f"Error: Insufficient file permissions: {e}")
            return False
    
    def _check_running_processes(self) -> bool:
        """Check for running processes that might interfere."""
        # This is a simplified check
        # In production, check for IDEs, test runners, etc.
        return True
    
    def create_safety_backup(self, backup_type: str = "auto") -> Optional[BackupInfo]:
        """Create a safety backup before cleanup."""
        if backup_type == "git":
            return self.backup_manager.create_git_backup()
        else:
            return self.backup_manager.create_full_backup()
    
    def emergency_restore(self) -> bool:
        """Perform emergency restore from the latest backup."""
        if not self.backup_manager.backups:
            print("No backups available for emergency restore")
            return False
        
        latest_backup = self.backup_manager.backups[-1]
        print(f"Performing emergency restore from: {latest_backup.backup_id}")
        
        return self.backup_manager.restore_backup(latest_backup)
    
    def validate_cleanup_safety(self, reports: List[CleanupReport]) -> bool:
        """Validate that cleanup operations were safe."""
        print("Validating cleanup safety...")
        
        # Check for critical errors
        critical_errors = []
        for report in reports:
            for error in report.errors:
                if any(keyword in error.lower() for keyword in ['critical', 'fatal', 'corruption']):
                    critical_errors.append(error)
        
        if critical_errors:
            print(f"Critical errors detected: {critical_errors}")
            return False
        
        # Check file modification patterns
        total_files_modified = sum(len(report.files_modified) for report in reports)
        if total_files_modified > 1000:  # Arbitrary threshold
            print(f"Warning: Large number of files modified ({total_files_modified})")
        
        # Check for suspicious patterns
        for report in reports:
            for change in report.files_modified:
                if change.change_type == 'deleted' and 'important' in str(change.file_path).lower():
                    print(f"Warning: Important file deleted: {change.file_path}")
        
        print("Cleanup safety validation completed")
        return True
    
    def get_safety_summary(self) -> Dict[str, Any]:
        """Get a summary of safety status."""
        return {
            'safety_checks_enabled': self.safety_checks_enabled,
            'backup_summary': self.backup_manager.get_backup_summary(),
            'git_repository': (self.project_root / '.git').exists(),
            'project_root': str(self.project_root)
        }
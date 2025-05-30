"""Progress tracking for cleanup operations."""

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..core.base import CleanupPhase, CleanupReport


@dataclass
class PhaseProgress:
    """Progress information for a cleanup phase."""
    
    phase: CleanupPhase
    status: str  # 'pending', 'running', 'completed', 'failed', 'skipped'
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    progress_percentage: float = 0.0
    current_step: str = ""
    total_steps: int = 0
    completed_steps: int = 0
    estimated_remaining: Optional[timedelta] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OverallProgress:
    """Overall progress information for the cleanup pipeline."""
    
    total_phases: int
    completed_phases: int
    failed_phases: int
    skipped_phases: int
    current_phase: Optional[CleanupPhase] = None
    start_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    overall_percentage: float = 0.0


class ProgressTracker:
    """Tracks progress of cleanup operations."""
    
    def __init__(self):
        self.phase_progress: Dict[CleanupPhase, PhaseProgress] = {}
        self.overall_progress = OverallProgress(
            total_phases=0,
            completed_phases=0,
            failed_phases=0,
            skipped_phases=0
        )
        self.start_time: Optional[datetime] = None
        self.callbacks: List[callable] = []
    
    def initialize_phases(self, phases: List[CleanupPhase]) -> None:
        """Initialize tracking for all phases."""
        self.overall_progress.total_phases = len(phases)
        self.start_time = datetime.now()
        self.overall_progress.start_time = self.start_time
        
        for phase in phases:
            self.phase_progress[phase] = PhaseProgress(
                phase=phase,
                status='pending'
            )
        
        self._notify_callbacks()
    
    def start_phase(self, phase: CleanupPhase, total_steps: int = 0) -> None:
        """Mark a phase as started."""
        if phase not in self.phase_progress:
            self.phase_progress[phase] = PhaseProgress(phase=phase, status='pending')
        
        progress = self.phase_progress[phase]
        progress.status = 'running'
        progress.start_time = datetime.now()
        progress.total_steps = total_steps
        progress.completed_steps = 0
        progress.progress_percentage = 0.0
        
        self.overall_progress.current_phase = phase
        self._update_overall_progress()
        self._notify_callbacks()
    
    def update_phase_progress(self, phase: CleanupPhase, 
                            completed_steps: Optional[int] = None,
                            current_step: Optional[str] = None,
                            metrics: Optional[Dict[str, Any]] = None) -> None:
        """Update progress for a specific phase."""
        if phase not in self.phase_progress:
            return
        
        progress = self.phase_progress[phase]
        
        if completed_steps is not None:
            progress.completed_steps = completed_steps
            if progress.total_steps > 0:
                progress.progress_percentage = (completed_steps / progress.total_steps) * 100
        
        if current_step is not None:
            progress.current_step = current_step
        
        if metrics is not None:
            progress.metrics.update(metrics)
        
        # Estimate remaining time
        if progress.start_time and progress.progress_percentage > 0:
            elapsed = datetime.now() - progress.start_time
            if progress.progress_percentage < 100:
                estimated_total = elapsed * (100 / progress.progress_percentage)
                progress.estimated_remaining = estimated_total - elapsed
        
        self._update_overall_progress()
        self._notify_callbacks()
    
    def complete_phase(self, phase: CleanupPhase, report: CleanupReport) -> None:
        """Mark a phase as completed."""
        if phase not in self.phase_progress:
            return
        
        progress = self.phase_progress[phase]
        progress.end_time = datetime.now()
        progress.progress_percentage = 100.0
        progress.completed_steps = progress.total_steps
        progress.estimated_remaining = timedelta(0)
        
        if report.success:
            progress.status = 'completed'
            self.overall_progress.completed_phases += 1
        else:
            progress.status = 'failed'
            self.overall_progress.failed_phases += 1
        
        # Add report metrics
        progress.metrics.update({
            'files_modified': len(report.files_modified),
            'errors': len(report.errors),
            'warnings': len(report.warnings),
            'success': report.success
        })
        
        self._update_overall_progress()
        self._notify_callbacks()
    
    def skip_phase(self, phase: CleanupPhase, reason: str) -> None:
        """Mark a phase as skipped."""
        if phase not in self.phase_progress:
            return
        
        progress = self.phase_progress[phase]
        progress.status = 'skipped'
        progress.end_time = datetime.now()
        progress.current_step = f"Skipped: {reason}"
        progress.progress_percentage = 100.0
        
        self.overall_progress.skipped_phases += 1
        self._update_overall_progress()
        self._notify_callbacks()
    
    def _update_overall_progress(self) -> None:
        """Update overall progress calculations."""
        total = self.overall_progress.total_phases
        if total == 0:
            return
        
        completed = self.overall_progress.completed_phases
        failed = self.overall_progress.failed_phases
        skipped = self.overall_progress.skipped_phases
        
        # Calculate overall percentage
        finished_phases = completed + failed + skipped
        self.overall_progress.overall_percentage = (finished_phases / total) * 100
        
        # Estimate completion time
        if self.start_time and finished_phases > 0:
            elapsed = datetime.now() - self.start_time
            if finished_phases < total:
                estimated_total = elapsed * (total / finished_phases)
                self.overall_progress.estimated_completion = self.start_time + estimated_total
    
    def add_progress_callback(self, callback: callable) -> None:
        """Add a callback to be notified of progress updates."""
        self.callbacks.append(callback)
    
    def remove_progress_callback(self, callback: callable) -> None:
        """Remove a progress callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def _notify_callbacks(self) -> None:
        """Notify all registered callbacks of progress updates."""
        for callback in self.callbacks:
            try:
                callback(self)
            except Exception as e:
                print(f"Error in progress callback: {e}")
    
    def get_phase_status(self, phase: CleanupPhase) -> Optional[PhaseProgress]:
        """Get status for a specific phase."""
        return self.phase_progress.get(phase)
    
    def get_overall_status(self) -> OverallProgress:
        """Get overall progress status."""
        return self.overall_progress
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all progress."""
        phase_summaries = {}
        for phase, progress in self.phase_progress.items():
            duration = None
            if progress.start_time:
                end_time = progress.end_time or datetime.now()
                duration = (end_time - progress.start_time).total_seconds()
            
            phase_summaries[phase.value] = {
                'status': progress.status,
                'progress_percentage': progress.progress_percentage,
                'current_step': progress.current_step,
                'completed_steps': progress.completed_steps,
                'total_steps': progress.total_steps,
                'duration_seconds': duration,
                'estimated_remaining_seconds': progress.estimated_remaining.total_seconds() if progress.estimated_remaining else None,
                'metrics': progress.metrics
            }
        
        overall_duration = None
        if self.start_time:
            overall_duration = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'overall': {
                'total_phases': self.overall_progress.total_phases,
                'completed_phases': self.overall_progress.completed_phases,
                'failed_phases': self.overall_progress.failed_phases,
                'skipped_phases': self.overall_progress.skipped_phases,
                'current_phase': self.overall_progress.current_phase.value if self.overall_progress.current_phase else None,
                'overall_percentage': self.overall_progress.overall_percentage,
                'duration_seconds': overall_duration,
                'estimated_completion': self.overall_progress.estimated_completion.isoformat() if self.overall_progress.estimated_completion else None
            },
            'phases': phase_summaries
        }
    
    def print_progress(self) -> None:
        """Print current progress to console."""
        print("\n" + "="*60)
        print("CLEANUP PROGRESS")
        print("="*60)
        
        # Overall progress
        overall = self.overall_progress
        print(f"Overall: {overall.overall_percentage:.1f}% complete")
        print(f"Phases: {overall.completed_phases} completed, {overall.failed_phases} failed, {overall.skipped_phases} skipped")
        
        if overall.current_phase:
            print(f"Current: {overall.current_phase.value}")
        
        if overall.estimated_completion:
            remaining = overall.estimated_completion - datetime.now()
            if remaining.total_seconds() > 0:
                print(f"Estimated completion: {remaining}")
        
        print()
        
        # Phase details
        for phase, progress in self.phase_progress.items():
            status_icon = {
                'pending': '⏳',
                'running': '🔄',
                'completed': '✅',
                'failed': '❌',
                'skipped': '⏭️'
            }.get(progress.status, '❓')
            
            print(f"{status_icon} {phase.value}: {progress.status}")
            
            if progress.status == 'running':
                print(f"   Progress: {progress.progress_percentage:.1f}%")
                if progress.current_step:
                    print(f"   Step: {progress.current_step}")
                if progress.estimated_remaining:
                    print(f"   Remaining: {progress.estimated_remaining}")
            
            elif progress.status in ['completed', 'failed']:
                if progress.start_time and progress.end_time:
                    duration = progress.end_time - progress.start_time
                    print(f"   Duration: {duration}")
                
                if progress.metrics:
                    files_modified = progress.metrics.get('files_modified', 0)
                    errors = progress.metrics.get('errors', 0)
                    warnings = progress.metrics.get('warnings', 0)
                    print(f"   Files: {files_modified}, Errors: {errors}, Warnings: {warnings}")
        
        print("="*60)


def create_console_progress_callback() -> callable:
    """Create a callback that prints progress to console."""
    last_update = [0]  # Use list to allow modification in closure
    
    def callback(tracker: ProgressTracker):
        # Throttle updates to avoid spam
        now = time.time()
        if now - last_update[0] < 1.0:  # Update at most once per second
            return
        last_update[0] = now
        
        # Clear screen and print progress
        print("\033[2J\033[H", end="")  # Clear screen and move cursor to top
        tracker.print_progress()
    
    return callback
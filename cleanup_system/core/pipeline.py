"""Cleanup pipeline orchestration components."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from .base import CleanupPhase, CleanupReport, CodeCleanupBase
from ..modules.dead_code import DeadCodeAnalyzer
from ..modules.documentation import DocumentationRefresher
from ..modules.consolidation import CodeConsolidator
from ..modules.standards import StandardsEnforcer
from ..modules.testing import TestValidator
from ..modules.dependencies import DependencyManager


class ExecutionStrategy(Enum):
    """Execution strategy for the cleanup pipeline."""
    
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"


@dataclass
class PhaseConfig:
    """Configuration for a cleanup phase."""
    
    phase: CleanupPhase
    enabled: bool = True
    dry_run: bool = False
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[CleanupPhase] = field(default_factory=list)
    timeout: Optional[int] = None  # seconds


@dataclass
class ExecutionPlan:
    """Execution plan for the cleanup pipeline."""
    
    phases: List[PhaseConfig]
    strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL
    stop_on_error: bool = True
    backup_before_execution: bool = True
    validate_before_execution: bool = True
    
    def get_enabled_phases(self) -> List[PhaseConfig]:
        """Get only enabled phases."""
        return [phase for phase in self.phases if phase.enabled]
    
    def get_phase_config(self, phase: CleanupPhase) -> Optional[PhaseConfig]:
        """Get configuration for a specific phase."""
        for phase_config in self.phases:
            if phase_config.phase == phase:
                return phase_config
        return None


class PhaseExecutor:
    """Executes individual cleanup phases."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self._module_registry: Dict[CleanupPhase, Type[CodeCleanupBase]] = {
            CleanupPhase.DEAD_CODE_REMOVAL: DeadCodeAnalyzer,
            CleanupPhase.DOCUMENTATION_REFRESH: DocumentationRefresher,
            CleanupPhase.CODE_CONSOLIDATION: CodeConsolidator,
            CleanupPhase.STANDARDS_ENFORCEMENT: StandardsEnforcer,
            CleanupPhase.TEST_VALIDATION: TestValidator,
            CleanupPhase.DEPENDENCY_CLEANUP: DependencyManager,
        }
    
    def execute_phase(self, phase_config: PhaseConfig) -> CleanupReport:
        """Execute a single cleanup phase."""
        print(f"Executing phase: {phase_config.phase.value}")
        
        # Get the module class for this phase
        module_class = self._module_registry.get(phase_config.phase)
        if not module_class:
            report = CleanupReport(
                phase=phase_config.phase,
                start_time=datetime.now()
            )
            report.add_error(f"No module registered for phase {phase_config.phase.value}")
            report.finalize()
            return report
        
        try:
            # Create module instance
            module = module_class(self.project_root, phase_config.config)
            
            # Validate before execution
            if not module.validate():
                report = CleanupReport(
                    phase=phase_config.phase,
                    start_time=datetime.now()
                )
                report.add_error(f"Validation failed for phase {phase_config.phase.value}")
                report.finalize()
                return report
            
            # Execute the phase
            report = module.execute(dry_run=phase_config.dry_run)
            
            print(f"Phase {phase_config.phase.value} completed: {report.success}")
            return report
        
        except Exception as e:
            report = CleanupReport(
                phase=phase_config.phase,
                start_time=datetime.now()
            )
            report.add_error(f"Exception during phase execution: {str(e)}")
            report.finalize()
            return report
    
    def analyze_phase(self, phase_config: PhaseConfig) -> Dict[str, Any]:
        """Analyze a phase without executing it."""
        module_class = self._module_registry.get(phase_config.phase)
        if not module_class:
            return {"error": f"No module registered for phase {phase_config.phase.value}"}
        
        try:
            module = module_class(self.project_root, phase_config.config)
            return module.analyze()
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}


class CleanupPipeline:
    """Main cleanup pipeline orchestrator."""
    
    def __init__(self, project_root: Path, config: Optional[Dict[str, Any]] = None):
        self.project_root = project_root
        self.config = config or {}
        self.executor = PhaseExecutor(project_root)
        self.execution_plan: Optional[ExecutionPlan] = None
        self.reports: List[CleanupReport] = []
    
    def create_default_execution_plan(self) -> ExecutionPlan:
        """Create a default execution plan with all phases."""
        phases = [
            PhaseConfig(
                phase=CleanupPhase.DEAD_CODE_REMOVAL,
                config=self.config.get('dead_code', {})
            ),
            PhaseConfig(
                phase=CleanupPhase.DOCUMENTATION_REFRESH,
                config=self.config.get('documentation', {}),
                dependencies=[CleanupPhase.DEAD_CODE_REMOVAL]
            ),
            PhaseConfig(
                phase=CleanupPhase.CODE_CONSOLIDATION,
                config=self.config.get('consolidation', {}),
                dependencies=[CleanupPhase.DEAD_CODE_REMOVAL]
            ),
            PhaseConfig(
                phase=CleanupPhase.STANDARDS_ENFORCEMENT,
                config=self.config.get('standards', {}),
                dependencies=[CleanupPhase.CODE_CONSOLIDATION]
            ),
            PhaseConfig(
                phase=CleanupPhase.DEPENDENCY_CLEANUP,
                config=self.config.get('dependencies', {}),
                dependencies=[CleanupPhase.DEAD_CODE_REMOVAL]
            ),
            PhaseConfig(
                phase=CleanupPhase.TEST_VALIDATION,
                config=self.config.get('testing', {}),
                dependencies=[
                    CleanupPhase.DEAD_CODE_REMOVAL,
                    CleanupPhase.CODE_CONSOLIDATION,
                    CleanupPhase.STANDARDS_ENFORCEMENT,
                    CleanupPhase.DEPENDENCY_CLEANUP
                ]
            ),
        ]
        
        return ExecutionPlan(
            phases=phases,
            strategy=ExecutionStrategy.SEQUENTIAL,
            stop_on_error=self.config.get('stop_on_error', True),
            backup_before_execution=self.config.get('backup_before_execution', True),
            validate_before_execution=self.config.get('validate_before_execution', True)
        )
    
    def set_execution_plan(self, plan: ExecutionPlan) -> None:
        """Set the execution plan."""
        self.execution_plan = plan
    
    def analyze_all_phases(self) -> Dict[str, Any]:
        """Analyze all phases without executing them."""
        if not self.execution_plan:
            self.execution_plan = self.create_default_execution_plan()
        
        analysis = {}
        
        for phase_config in self.execution_plan.get_enabled_phases():
            phase_analysis = self.executor.analyze_phase(phase_config)
            analysis[phase_config.phase.value] = phase_analysis
        
        return analysis
    
    def execute_all_phases(self, dry_run: bool = False) -> List[CleanupReport]:
        """Execute all phases in the pipeline."""
        if not self.execution_plan:
            self.execution_plan = self.create_default_execution_plan()
        
        self.reports = []
        enabled_phases = self.execution_plan.get_enabled_phases()
        
        print(f"Starting cleanup pipeline with {len(enabled_phases)} phases")
        
        # Override dry_run if specified
        if dry_run:
            for phase_config in enabled_phases:
                phase_config.dry_run = True
        
        # Execute phases based on strategy
        if self.execution_plan.strategy == ExecutionStrategy.SEQUENTIAL:
            self._execute_sequential(enabled_phases)
        else:
            # For now, only sequential execution is implemented
            self._execute_sequential(enabled_phases)
        
        return self.reports
    
    def _execute_sequential(self, phases: List[PhaseConfig]) -> None:
        """Execute phases sequentially."""
        for phase_config in phases:
            # Check dependencies
            if not self._check_dependencies(phase_config):
                report = CleanupReport(
                    phase=phase_config.phase,
                    start_time=datetime.now()
                )
                report.add_error(f"Dependencies not satisfied for phase {phase_config.phase.value}")
                report.finalize()
                self.reports.append(report)
                
                if self.execution_plan.stop_on_error:
                    break
                continue
            
            # Execute the phase
            report = self.executor.execute_phase(phase_config)
            self.reports.append(report)
            
            # Stop on error if configured
            if not report.success and self.execution_plan.stop_on_error:
                print(f"Stopping pipeline due to error in phase {phase_config.phase.value}")
                break
    
    def _check_dependencies(self, phase_config: PhaseConfig) -> bool:
        """Check if all dependencies for a phase have been successfully executed."""
        for dep_phase in phase_config.dependencies:
            # Find the report for this dependency
            dep_report = None
            for report in self.reports:
                if report.phase == dep_phase:
                    dep_report = report
                    break
            
            # If dependency wasn't executed or failed, return False
            if not dep_report or not dep_report.success:
                return False
        
        return True
    
    def execute_single_phase(self, phase: CleanupPhase, dry_run: bool = False) -> CleanupReport:
        """Execute a single phase."""
        if not self.execution_plan:
            self.execution_plan = self.create_default_execution_plan()
        
        phase_config = self.execution_plan.get_phase_config(phase)
        if not phase_config:
            report = CleanupReport(
                phase=phase,
                start_time=datetime.now()
            )
            report.add_error(f"Phase {phase.value} not found in execution plan")
            report.finalize()
            # Add report to pipeline reports for summary
            if not hasattr(self, 'reports'):
                self.reports = []
            self.reports.append(report)
            return report
        
        if dry_run:
            phase_config.dry_run = True
        
        report = self.executor.execute_phase(phase_config)
        # Add report to pipeline reports for summary
        if not hasattr(self, 'reports'):
            self.reports = []
        self.reports.append(report)
        return report
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get a summary of the pipeline execution."""
        if not self.reports:
            return {"status": "not_executed"}
        
        total_phases = len(self.reports)
        successful_phases = sum(1 for report in self.reports if report.success)
        failed_phases = total_phases - successful_phases
        
        total_files_modified = sum(len(report.files_modified) for report in self.reports)
        total_errors = sum(len(report.errors) for report in self.reports)
        total_warnings = sum(len(report.warnings) for report in self.reports)
        
        return {
            "status": "completed",
            "total_phases": total_phases,
            "successful_phases": successful_phases,
            "failed_phases": failed_phases,
            "total_files_modified": total_files_modified,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "execution_time": self._calculate_total_execution_time(),
            "phases": [
                {
                    "phase": report.phase.value,
                    "success": report.success,
                    "files_modified": len(report.files_modified),
                    "errors": len(report.errors),
                    "warnings": len(report.warnings),
                    "duration": (report.end_time - report.start_time).total_seconds() if report.end_time else 0
                }
                for report in self.reports
            ]
        }
    
    def _calculate_total_execution_time(self) -> float:
        """Calculate total execution time for all phases."""
        if not self.reports:
            return 0.0
        
        total_time = 0.0
        for report in self.reports:
            if report.end_time:
                duration = (report.end_time - report.start_time).total_seconds()
                total_time += duration
        
        return total_time
    
    def save_reports(self, output_path: Path) -> None:
        """Save all reports to a JSON file."""
        reports_data = []
        
        for report in self.reports:
            report_data = {
                "phase": report.phase.value,
                "start_time": report.start_time.isoformat(),
                "end_time": report.end_time.isoformat() if report.end_time else None,
                "success": report.success,
                "files_modified": [
                    {
                        "file_path": str(change.file_path),
                        "change_type": change.change_type,
                        "description": change.description,
                        "line_changes": change.line_changes
                    }
                    for change in report.files_modified
                ],
                "errors": report.errors,
                "warnings": report.warnings,
                "metrics": report.metrics
            }
            reports_data.append(report_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(reports_data, f, indent=2)
        
        print(f"Reports saved to {output_path}")
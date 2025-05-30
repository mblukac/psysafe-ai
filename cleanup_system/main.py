"""Main entry point for the PsySafe AI cleanup system."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .config.manager import ConfigurationManager
from .core.pipeline import CleanupPipeline
from .core.safety import SafetyManager
from .core.validation import ValidationEngine
from .reporting.progress import ProgressTracker, create_console_progress_callback
from .reporting.reports import ReportGenerator


def main():
    """Main entry point for the cleanup system."""
    parser = argparse.ArgumentParser(
        description="PsySafe AI Code Cleanup and Refactoring System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m cleanup_system.main --analyze                    # Analyze only, no changes
  python -m cleanup_system.main --dry-run                    # Simulate cleanup
  python -m cleanup_system.main --phase dead_code_removal    # Run single phase
  python -m cleanup_system.main --config custom_config.yaml # Use custom config
  python -m cleanup_system.main --no-backup                  # Skip backup creation
        """
    )
    
    parser.add_argument(
        '--project-root',
        type=Path,
        default=Path.cwd(),
        help='Root directory of the project (default: current directory)'
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Only analyze the codebase, do not make changes'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulate cleanup without making actual changes'
    )
    
    parser.add_argument(
        '--phase',
        choices=[
            'dead_code_removal',
            'documentation_refresh', 
            'code_consolidation',
            'standards_enforcement',
            'dependency_cleanup',
            'test_validation'
        ],
        help='Run only a specific phase'
    )
    
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup before cleanup'
    )
    
    parser.add_argument(
        '--no-validation',
        action='store_true',
        help='Skip validation checks'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('cleanup_reports'),
        help='Directory for output reports (default: cleanup_reports)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--create-config',
        action='store_true',
        help='Create a default configuration file and exit'
    )
    
    args = parser.parse_args()
    
    # Validate project root
    if not args.project_root.exists():
        print(f"Error: Project root does not exist: {args.project_root}")
        sys.exit(1)
    
    if not args.project_root.is_dir():
        print(f"Error: Project root is not a directory: {args.project_root}")
        sys.exit(1)
    
    # Initialize configuration manager
    config_manager = ConfigurationManager(args.project_root)
    
    # Handle config creation
    if args.create_config:
        config_manager.create_default_config_file()
        print(f"Default configuration created at: {config_manager.config_file}")
        sys.exit(0)
    
    # Load configuration
    config = config_manager.load_config(args.config)
    
    # Apply command line overrides
    if args.dry_run:
        for phase in config.phases:
            phase.dry_run = True
    
    if args.no_backup:
        config.safety.backup_enabled = False
    
    if args.no_validation:
        config.safety.validation_enabled = False
    
    if args.output_dir:
        config.output_dir = str(args.output_dir)
    
    # Initialize components
    pipeline = CleanupPipeline(args.project_root, config.__dict__)
    safety_manager = SafetyManager(args.project_root, config.safety.__dict__)
    validation_engine = ValidationEngine(args.project_root)
    progress_tracker = ProgressTracker()
    report_generator = ReportGenerator(Path(config.output_dir))
    
    # Add console progress callback if verbose
    if args.verbose:
        progress_tracker.add_progress_callback(create_console_progress_callback())
    
    try:
        # Handle analyze-only mode
        if args.analyze:
            return run_analysis_only(pipeline, validation_engine, report_generator)
        
        # Run full cleanup
        return run_cleanup(
            pipeline=pipeline,
            safety_manager=safety_manager,
            validation_engine=validation_engine,
            progress_tracker=progress_tracker,
            report_generator=report_generator,
            single_phase=args.phase,
            dry_run=args.dry_run,
            verbose=args.verbose
        )
    
    except KeyboardInterrupt:
        print("\nCleanup interrupted by user")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def run_analysis_only(pipeline: CleanupPipeline, 
                     validation_engine: ValidationEngine,
                     report_generator: ReportGenerator) -> int:
    """Run analysis only without making changes."""
    print("Running analysis mode...")
    
    # Analyze all phases
    analysis_results = pipeline.analyze_all_phases()
    
    # Run validation
    validation_results = validation_engine.run_comprehensive_validation()
    
    # Generate analysis report
    timestamp = pipeline.project_root.name
    analysis_report = {
        'mode': 'analysis_only',
        'analysis_results': analysis_results,
        'validation_summary': validation_engine.get_validation_summary()
    }
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    for phase_name, phase_analysis in analysis_results.items():
        print(f"\n{phase_name.replace('_', ' ').title()}:")
        for key, value in phase_analysis.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value}")
    
    validation_summary = validation_engine.get_validation_summary()
    print(f"\nValidation: {validation_summary['passed_checks']}/{validation_summary['total_checks']} checks passed")
    
    if validation_summary['critical_failures']:
        print("Critical validation failures:")
        for failure in validation_summary['critical_failures']:
            print(f"  - {failure}")
    
    print("="*60)
    print("Analysis complete. No changes were made.")
    
    return 0


def run_cleanup(pipeline: CleanupPipeline,
               safety_manager: SafetyManager,
               validation_engine: ValidationEngine,
               progress_tracker: ProgressTracker,
               report_generator: ReportGenerator,
               single_phase: Optional[str] = None,
               dry_run: bool = False,
               verbose: bool = False) -> int:
    """Run the full cleanup process."""
    
    print("Starting PsySafe AI Cleanup System")
    print(f"Project: {pipeline.project_root}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    
    if single_phase:
        print(f"Running single phase: {single_phase}")
    
    # Pre-cleanup safety checks
    if not safety_manager.pre_cleanup_safety_check():
        print("Safety checks failed. Aborting cleanup.")
        return 1
    
    # Create backup if enabled
    backup_info = None
    if safety_manager.config.get('backup_enabled', True) and not dry_run:
        print("Creating safety backup...")
        backup_info = safety_manager.create_safety_backup()
        if not backup_info:
            print("Failed to create backup. Aborting cleanup.")
            return 1
    
    # Run initial validation if enabled
    validation_results = []
    if safety_manager.config.get('validation_enabled', True):
        print("Running pre-cleanup validation...")
        validation_results = validation_engine.run_comprehensive_validation()
        
        validation_summary = validation_engine.get_validation_summary()
        if validation_summary['critical_failures']:
            print("Critical validation failures detected:")
            for failure in validation_summary['critical_failures']:
                print(f"  - {failure}")
            print("Aborting cleanup due to critical failures.")
            return 1
    
    # Execute cleanup
    try:
        # Import CleanupPhase at the beginning so it's available in both branches
        from .core.base import CleanupPhase
        
        if single_phase:
            # Run single phase
            phase_enum = CleanupPhase(single_phase)
            reports = [pipeline.execute_single_phase(phase_enum, dry_run)]
        else:
            # Run all phases
            execution_plan = pipeline.create_default_execution_plan()
            pipeline.set_execution_plan(execution_plan)
            
            # Initialize progress tracking
            enabled_phases = [p.phase for p in execution_plan.get_enabled_phases()]
            progress_tracker.initialize_phases(enabled_phases)
            
            reports = pipeline.execute_all_phases(dry_run)
    
    except Exception as e:
        print(f"Cleanup execution failed: {e}")
        
        # Emergency restore if backup exists
        if backup_info and not dry_run:
            print("Attempting emergency restore...")
            if safety_manager.emergency_restore():
                print("Emergency restore successful")
            else:
                print("Emergency restore failed")
        
        return 1
    
    # Validate cleanup safety
    if not safety_manager.validate_cleanup_safety(reports):
        print("Cleanup safety validation failed")
        
        # Consider restoring backup
        if backup_info and not dry_run:
            response = input("Restore from backup? (y/N): ")
            if response.lower() == 'y':
                if safety_manager.emergency_restore():
                    print("Restored from backup")
                    return 1
    
    # Run post-cleanup validation
    if safety_manager.config.get('validation_enabled', True):
        print("Running post-cleanup validation...")
        post_validation_results = validation_engine.run_comprehensive_validation()
        validation_results.extend(post_validation_results)
    
    # Generate comprehensive report
    print("Generating cleanup report...")
    report_path = report_generator.generate_comprehensive_report(
        reports=reports,
        progress_tracker=progress_tracker,
        validation_results=validation_results
    )
    
    # Generate audit log
    audit_path = report_generator.generate_audit_log(reports)
    
    # Print summary
    pipeline_summary = pipeline.get_pipeline_summary()
    print("\n" + "="*60)
    print("CLEANUP SUMMARY")
    print("="*60)
    print(f"Status: {pipeline_summary['status']}")
    
    # Only print detailed stats if execution actually occurred
    if pipeline_summary['status'] != 'not_executed':
        print(f"Phases: {pipeline_summary['successful_phases']}/{pipeline_summary['total_phases']} successful")
        print(f"Files Modified: {pipeline_summary['total_files_modified']}")
        print(f"Errors: {pipeline_summary['total_errors']}")
        print(f"Warnings: {pipeline_summary['total_warnings']}")
        print(f"Duration: {pipeline_summary['execution_time']:.1f}s")
    
    print(f"Report: {report_path}")
    print(f"Audit Log: {audit_path}")
    
    if backup_info:
        print(f"Backup: {backup_info.backup_path}")
    
    print("="*60)
    
    # Return appropriate exit code
    if pipeline_summary['status'] == 'not_executed':
        print("Cleanup was not executed")
        return 1
    elif pipeline_summary.get('failed_phases', 0) > 0:
        print("Cleanup completed with errors")
        return 1
    else:
        print("Cleanup completed successfully")
        return 0


def cleanup_self():
    """Remove the cleanup system after successful completion."""
    import shutil
    
    cleanup_dir = Path(__file__).parent
    
    try:
        # Remove the cleanup system directory
        shutil.rmtree(cleanup_dir)
        print(f"Cleanup system removed: {cleanup_dir}")
    except Exception as e:
        print(f"Warning: Could not remove cleanup system: {e}")
        print(f"Please manually remove: {cleanup_dir}")


if __name__ == '__main__':
    exit_code = main()
    
    # Self-cleanup on successful completion (if not in dry-run mode)
    if exit_code == 0 and '--dry-run' not in sys.argv and '--analyze' not in sys.argv:
        response = input("\nCleanup completed successfully. Remove cleanup system? (y/N): ")
        if response.lower() == 'y':
            cleanup_self()
    
    sys.exit(exit_code)
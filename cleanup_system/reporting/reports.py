"""Report generation for cleanup operations."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.base import CleanupReport
from ..core.validation import ValidationResult
from .progress import ProgressTracker


class ReportGenerator:
    """Generates comprehensive reports for cleanup operations."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_comprehensive_report(self, 
                                    reports: List[CleanupReport],
                                    progress_tracker: Optional[ProgressTracker] = None,
                                    validation_results: Optional[List[ValidationResult]] = None) -> Path:
        """Generate a comprehensive cleanup report."""
        timestamp = datetime.now()
        report_data = {
            'metadata': {
                'generated_at': timestamp.isoformat(),
                'generator': 'PsySafe AI Cleanup System',
                'version': '1.0.0'
            },
            'summary': self._generate_summary(reports),
            'phases': self._generate_phase_reports(reports),
            'validation': self._generate_validation_report(validation_results) if validation_results else None,
            'progress': progress_tracker.get_summary() if progress_tracker else None,
            'recommendations': self._generate_recommendations(reports)
        }
        
        # Save JSON report
        json_path = self.output_dir / f"cleanup_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Generate HTML report
        html_path = self._generate_html_report(report_data, timestamp)
        
        # Generate markdown summary
        md_path = self._generate_markdown_summary(report_data, timestamp)
        
        print(f"Comprehensive report generated:")
        print(f"  JSON: {json_path}")
        print(f"  HTML: {html_path}")
        print(f"  Markdown: {md_path}")
        
        return json_path
    
    def _generate_summary(self, reports: List[CleanupReport]) -> Dict[str, Any]:
        """Generate summary statistics."""
        total_phases = len(reports)
        successful_phases = sum(1 for r in reports if r.success)
        failed_phases = total_phases - successful_phases
        
        total_files_modified = sum(len(r.files_modified) for r in reports)
        total_errors = sum(len(r.errors) for r in reports)
        total_warnings = sum(len(r.warnings) for r in reports)
        
        # Calculate total execution time
        total_duration = 0.0
        for report in reports:
            if report.end_time:
                duration = (report.end_time - report.start_time).total_seconds()
                total_duration += duration
        
        # Collect all metrics
        all_metrics = {}
        for report in reports:
            phase_metrics = {f"{report.phase.value}_{k}": v for k, v in report.metrics.items()}
            all_metrics.update(phase_metrics)
        
        return {
            'total_phases': total_phases,
            'successful_phases': successful_phases,
            'failed_phases': failed_phases,
            'success_rate': successful_phases / total_phases if total_phases > 0 else 0,
            'total_files_modified': total_files_modified,
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'total_duration_seconds': total_duration,
            'metrics': all_metrics
        }
    
    def _generate_phase_reports(self, reports: List[CleanupReport]) -> List[Dict[str, Any]]:
        """Generate detailed reports for each phase."""
        phase_reports = []
        
        for report in reports:
            duration = 0.0
            if report.end_time:
                duration = (report.end_time - report.start_time).total_seconds()
            
            phase_report = {
                'phase': report.phase.value,
                'success': report.success,
                'start_time': report.start_time.isoformat(),
                'end_time': report.end_time.isoformat() if report.end_time else None,
                'duration_seconds': duration,
                'files_modified': [
                    {
                        'file_path': str(change.file_path),
                        'change_type': change.change_type,
                        'description': change.description,
                        'line_changes': change.line_changes
                    }
                    for change in report.files_modified
                ],
                'errors': report.errors,
                'warnings': report.warnings,
                'metrics': report.metrics,
                'summary': report.get_summary()
            }
            
            phase_reports.append(phase_report)
        
        return phase_reports
    
    def _generate_validation_report(self, validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate validation report."""
        total_checks = len(validation_results)
        passed_checks = sum(1 for r in validation_results if r.passed)
        failed_checks = total_checks - passed_checks
        
        # Group by severity
        by_severity = {}
        for result in validation_results:
            if result.severity not in by_severity:
                by_severity[result.severity] = []
            by_severity[result.severity].append({
                'check_name': result.check_name,
                'passed': result.passed,
                'message': result.message,
                'details': result.details,
                'timestamp': result.timestamp.isoformat()
            })
        
        # Group by check type
        by_check_type = {}
        for result in validation_results:
            if result.check_name not in by_check_type:
                by_check_type[result.check_name] = {'passed': 0, 'failed': 0, 'results': []}
            
            if result.passed:
                by_check_type[result.check_name]['passed'] += 1
            else:
                by_check_type[result.check_name]['failed'] += 1
            
            by_check_type[result.check_name]['results'].append({
                'passed': result.passed,
                'message': result.message,
                'severity': result.severity,
                'details': result.details,
                'timestamp': result.timestamp.isoformat()
            })
        
        return {
            'summary': {
                'total_checks': total_checks,
                'passed_checks': passed_checks,
                'failed_checks': failed_checks,
                'success_rate': passed_checks / total_checks if total_checks > 0 else 0
            },
            'by_severity': by_severity,
            'by_check_type': by_check_type
        }
    
    def _generate_recommendations(self, reports: List[CleanupReport]) -> List[Dict[str, Any]]:
        """Generate recommendations based on cleanup results."""
        recommendations = []
        
        # Analyze results and generate recommendations
        total_errors = sum(len(r.errors) for r in reports)
        total_warnings = sum(len(r.warnings) for r in reports)
        
        if total_errors > 0:
            recommendations.append({
                'type': 'error',
                'priority': 'high',
                'title': 'Address Cleanup Errors',
                'description': f'There were {total_errors} errors during cleanup that need attention.',
                'action': 'Review error messages and fix underlying issues before running cleanup again.'
            })
        
        if total_warnings > 10:
            recommendations.append({
                'type': 'warning',
                'priority': 'medium',
                'title': 'Review Warnings',
                'description': f'There were {total_warnings} warnings during cleanup.',
                'action': 'Review warning messages to identify potential improvements.'
            })
        
        # Check for phases that took a long time
        for report in reports:
            if report.end_time:
                duration = (report.end_time - report.start_time).total_seconds()
                if duration > 300:  # 5 minutes
                    recommendations.append({
                        'type': 'performance',
                        'priority': 'low',
                        'title': f'Optimize {report.phase.value} Phase',
                        'description': f'The {report.phase.value} phase took {duration:.1f} seconds.',
                        'action': 'Consider optimizing this phase or running it separately.'
                    })
        
        # Check for phases with many file modifications
        for report in reports:
            if len(report.files_modified) > 100:
                recommendations.append({
                    'type': 'review',
                    'priority': 'medium',
                    'title': f'Review {report.phase.value} Changes',
                    'description': f'The {report.phase.value} phase modified {len(report.files_modified)} files.',
                    'action': 'Review the changes to ensure they are correct and necessary.'
                })
        
        return recommendations
    
    def _generate_html_report(self, report_data: Dict[str, Any], timestamp: datetime) -> Path:
        """Generate an HTML report."""
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PsySafe AI Cleanup Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }}
        .metric {{ background: #e9e9e9; padding: 15px; border-radius: 5px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #333; }}
        .metric-label {{ color: #666; }}
        .phase {{ border: 1px solid #ddd; margin-bottom: 15px; border-radius: 5px; }}
        .phase-header {{ background: #f9f9f9; padding: 10px; border-bottom: 1px solid #ddd; }}
        .phase-content {{ padding: 15px; }}
        .success {{ color: #28a745; }}
        .error {{ color: #dc3545; }}
        .warning {{ color: #ffc107; }}
        .file-changes {{ max-height: 200px; overflow-y: auto; background: #f8f9fa; padding: 10px; border-radius: 3px; }}
        .recommendations {{ background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; }}
        .recommendation {{ margin-bottom: 10px; padding: 10px; border-left: 4px solid #ffc107; background: #fffbf0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>PsySafe AI Cleanup Report</h1>
        <p>Generated on {timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <h2>Summary</h2>
    <div class="summary">
        <div class="metric">
            <div class="metric-value">{report_data['summary']['total_phases']}</div>
            <div class="metric-label">Total Phases</div>
        </div>
        <div class="metric">
            <div class="metric-value">{report_data['summary']['successful_phases']}</div>
            <div class="metric-label">Successful</div>
        </div>
        <div class="metric">
            <div class="metric-value">{report_data['summary']['failed_phases']}</div>
            <div class="metric-label">Failed</div>
        </div>
        <div class="metric">
            <div class="metric-value">{report_data['summary']['total_files_modified']}</div>
            <div class="metric-label">Files Modified</div>
        </div>
        <div class="metric">
            <div class="metric-value">{report_data['summary']['total_duration_seconds']:.1f}s</div>
            <div class="metric-label">Total Duration</div>
        </div>
    </div>
    
    <h2>Phase Details</h2>
"""
        
        for phase in report_data['phases']:
            status_class = 'success' if phase['success'] else 'error'
            html_content += f"""
    <div class="phase">
        <div class="phase-header">
            <h3 class="{status_class}">{phase['phase'].replace('_', ' ').title()}</h3>
            <p>Duration: {phase['duration_seconds']:.1f}s | Files Modified: {len(phase['files_modified'])} | 
               Errors: {len(phase['errors'])} | Warnings: {len(phase['warnings'])}</p>
        </div>
        <div class="phase-content">
"""
            
            if phase['errors']:
                html_content += f"""
            <h4 class="error">Errors ({len(phase['errors'])})</h4>
            <ul>
"""
                for error in phase['errors']:
                    html_content += f"                <li>{error}</li>\n"
                html_content += "            </ul>\n"
            
            if phase['warnings']:
                html_content += f"""
            <h4 class="warning">Warnings ({len(phase['warnings'])})</h4>
            <ul>
"""
                for warning in phase['warnings']:
                    html_content += f"                <li>{warning}</li>\n"
                html_content += "            </ul>\n"
            
            if phase['files_modified']:
                html_content += f"""
            <h4>File Changes ({len(phase['files_modified'])})</h4>
            <div class="file-changes">
"""
                for change in phase['files_modified'][:20]:  # Limit to first 20
                    html_content += f"                <p><strong>{change['change_type']}:</strong> {change['file_path']} - {change['description']}</p>\n"
                
                if len(phase['files_modified']) > 20:
                    html_content += f"                <p><em>... and {len(phase['files_modified']) - 20} more files</em></p>\n"
                
                html_content += "            </div>\n"
            
            html_content += """
        </div>
    </div>
"""
        
        if report_data['recommendations']:
            html_content += """
    <h2>Recommendations</h2>
    <div class="recommendations">
"""
            for rec in report_data['recommendations']:
                html_content += f"""
        <div class="recommendation">
            <h4>{rec['title']} ({rec['priority']} priority)</h4>
            <p>{rec['description']}</p>
            <p><strong>Action:</strong> {rec['action']}</p>
        </div>
"""
            html_content += "    </div>\n"
        
        html_content += """
</body>
</html>
"""
        
        html_path = self.output_dir / f"cleanup_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_path
    
    def _generate_markdown_summary(self, report_data: Dict[str, Any], timestamp: datetime) -> Path:
        """Generate a markdown summary report."""
        md_content = f"""# PsySafe AI Cleanup Report

**Generated:** {timestamp.strftime('%Y-%m-%d %H:%M:%S')}

## Summary

| Metric | Value |
|--------|-------|
| Total Phases | {report_data['summary']['total_phases']} |
| Successful Phases | {report_data['summary']['successful_phases']} |
| Failed Phases | {report_data['summary']['failed_phases']} |
| Success Rate | {report_data['summary']['success_rate']:.1%} |
| Files Modified | {report_data['summary']['total_files_modified']} |
| Total Errors | {report_data['summary']['total_errors']} |
| Total Warnings | {report_data['summary']['total_warnings']} |
| Duration | {report_data['summary']['total_duration_seconds']:.1f}s |

## Phase Results

"""
        
        for phase in report_data['phases']:
            status = "✅ SUCCESS" if phase['success'] else "❌ FAILED"
            md_content += f"""### {phase['phase'].replace('_', ' ').title()} {status}

- **Duration:** {phase['duration_seconds']:.1f}s
- **Files Modified:** {len(phase['files_modified'])}
- **Errors:** {len(phase['errors'])}
- **Warnings:** {len(phase['warnings'])}

"""
            
            if phase['errors']:
                md_content += "**Errors:**\n"
                for error in phase['errors']:
                    md_content += f"- {error}\n"
                md_content += "\n"
            
            if phase['warnings']:
                md_content += "**Warnings:**\n"
                for warning in phase['warnings']:
                    md_content += f"- {warning}\n"
                md_content += "\n"
        
        if report_data['recommendations']:
            md_content += "## Recommendations\n\n"
            for rec in report_data['recommendations']:
                priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(rec['priority'], "⚪")
                md_content += f"""### {priority_emoji} {rec['title']}

**Priority:** {rec['priority']}
**Description:** {rec['description']}
**Action:** {rec['action']}

"""
        
        md_path = self.output_dir / f"cleanup_summary_{timestamp.strftime('%Y%m%d_%H%M%S')}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        return md_path
    
    def generate_audit_log(self, reports: List[CleanupReport]) -> Path:
        """Generate an audit log of all changes."""
        timestamp = datetime.now()
        
        audit_data = {
            'audit_metadata': {
                'generated_at': timestamp.isoformat(),
                'total_phases': len(reports),
                'total_changes': sum(len(r.files_modified) for r in reports)
            },
            'changes_by_phase': {},
            'changes_by_file': {},
            'all_changes': []
        }
        
        # Organize changes by phase and file
        for report in reports:
            phase_name = report.phase.value
            audit_data['changes_by_phase'][phase_name] = []
            
            for change in report.files_modified:
                change_record = {
                    'phase': phase_name,
                    'timestamp': report.start_time.isoformat(),
                    'file_path': str(change.file_path),
                    'change_type': change.change_type,
                    'description': change.description,
                    'line_changes': change.line_changes
                }
                
                audit_data['changes_by_phase'][phase_name].append(change_record)
                audit_data['all_changes'].append(change_record)
                
                # Group by file
                file_path = str(change.file_path)
                if file_path not in audit_data['changes_by_file']:
                    audit_data['changes_by_file'][file_path] = []
                audit_data['changes_by_file'][file_path].append(change_record)
        
        audit_path = self.output_dir / f"cleanup_audit_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        with open(audit_path, 'w', encoding='utf-8') as f:
            json.dump(audit_data, f, indent=2, default=str)
        
        print(f"Audit log generated: {audit_path}")
        return audit_path
"""
VS-Bench - Visualization and Analytics
=======================================

Interactive dashboards and visualization tools for:
- Performance distributions
- OOD failure analysis
- Uncertainty calibration plots
- Model comparison radar charts

Author: VS-Bench Development Team
Version: 1.0.0
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class PlotData:
    """Data structure for plot generation."""
    title: str
    x_label: str
    y_label: str
    data: Dict[str, Any]
    plot_type: str


class BenchmarkVisualizer:
    """
    Generate visualizations for benchmark results.
    
    Creates interactive HTML/JavaScript visualizations
    for analysis and reporting.
    """
    
    def __init__(self, theme: str = "light"):
        self.theme = theme
        self.colors = {
            "primary": "#4CAF50",
            "secondary": "#2196F3",
            "accent": "#FF9800",
            "danger": "#f44336",
            "success": "#4CAF50"
        }
    
    def generate_performance_distribution(
        self,
        benchmark_scores: Dict[str, List[float]],
        metric: str = "spearman"
    ) -> PlotData:
        """
        Generate performance distribution plot.
        
        Shows violin or box plots of performance across methods.
        
        Args:
            benchmark_scores: Dict mapping benchmark to list of scores
            metric: Metric to visualize
            
        Returns:
            PlotData object
        """
        data = {
            "benchmarks": list(benchmark_scores.keys()),
            "scores": {
                bench: {
                    "values": scores,
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "min": np.min(scores),
                    "max": np.max(scores),
                    "median": np.median(scores),
                    "q25": np.percentile(scores, 25),
                    "q75": np.percentile(scores, 75)
                }
                for bench, scores in benchmark_scores.items()
            }
        }
        
        return PlotData(
            title=f"Performance Distribution - {metric.upper()}",
            x_label="Benchmark",
            y_label=metric.upper(),
            data=data,
            plot_type="violin"
        )
    
    def generate_ood_failure_visualization(
        self,
        in_dist_scores: Dict[str, float],
        ood_scores: Dict[str, float],
        ood_categories: List[str]
    ) -> PlotData:
        """
        Generate OOD failure mode visualization.
        
        Shows performance drop across different OOD scenarios.
        
        Args:
            in_dist_scores: In-distribution scores by benchmark
            ood_scores: OOD scores by benchmark
            ood_categories: OOD category labels
            
        Returns:
            PlotData object
        """
        data = {
            "categories": ood_categories,
            "performance": {
                "in_distribution": in_dist_scores,
                "ood": ood_scores,
                "drop": {
                    cat: in_dist_scores.get(cat, 0) - ood_scores.get(cat, 0)
                    for cat in ood_categories
                }
            }
        }
        
        return PlotData(
            title="OOD Generalization Performance",
            x_label="OOD Category",
            y_label="Score",
            data=data,
            plot_type="grouped_bar"
        )
    
    def generate_calibration_plot(
        self,
        calibration_data: Dict[str, Any]
    ) -> PlotData:
        """
        Generate uncertainty calibration plot.
        
        Shows reliability diagram with ECE.
        
        Args:
            calibration_data: Calibration metrics from evaluation
            
        Returns:
            PlotData object
        """
        data = {
            "ece": calibration_data.get("ece", 0),
            "brier": calibration_data.get("brier_score", 0),
            "bins": calibration_data.get("calibration_bins", []),
            "accuracies": calibration_data.get("calibration_accuracies", []),
            "confidences": calibration_data.get("calibration_confidences", []),
            "perfect_calibration": [
                [0, 0], [1, 1]
            ]  # Diagonal line
        }
        
        return PlotData(
            title=f"Reliability Diagram (ECE: {data['ece']:.3f})",
            x_label="Confidence",
            y_label="Accuracy",
            data=data,
            plot_type="scatter_with_line"
        )
    
    def generate_radar_chart(
        self,
        methods: Dict[str, Dict[str, float]],
        categories: List[str]
    ) -> PlotData:
        """
        Generate radar chart for multi-dimensional comparison.
        
        Compares methods across multiple metrics simultaneously.
        
        Args:
            methods: Dict mapping method name to metric scores
            categories: List of metric categories
            
        Returns:
            PlotData object
        """
        data = {
            "categories": categories,
            "methods": {
                name: {
                    cat: scores.get(cat, 0)
                    for cat in categories
                }
                for name, scores in methods.items()
            }
        }
        
        return PlotData(
            title="Multi-Metric Performance Comparison",
            x_label="",
            y_label="",
            data=data,
            plot_type="radar"
        )
    
    def generate_leaderboard_chart(
        self,
        entries: List[Dict[str, Any]],
        top_k: int = 10
    ) -> PlotData:
        """
        Generate leaderboard bar chart.
        
        Args:
            entries: Leaderboard entries
            top_k: Show top k entries
            
        Returns:
            PlotData object
        """
        top_entries = entries[:top_k]
        
        data = {
            "ranks": [e["rank"] for e in top_entries],
            "teams": [e["team"] for e in top_entries],
            "methods": [e.get("method", "") for e in top_entries],
            "scores": [e["global_score"] for e in top_entries],
            "colors": [
                self.colors["success"] if e["rank"] <= 3 else self.colors["secondary"]
                for e in top_entries
            ]
        }
        
        return PlotData(
            title=f"Top {top_k} Leaderboard",
            x_label="Rank",
            y_label="Global Score",
            data=data,
            plot_type="horizontal_bar"
        )
    
    def generate_enrichment_plot(
        self,
        enrichment_curves: Dict[str, List[Tuple[float, float]]]
    ) -> PlotData:
        """
        Generate enrichment factor plot.
        
        Shows enrichment curves for different methods.
        
        Args:
            enrichment_curves: Dict mapping method to (fraction, enrichment) pairs
            
        Returns:
            PlotData object
        """
        data = {
            "methods": {
                name: {
                    "fractions": [p[0] for p in points],
                    "enrichment": [p[1] for p in points]
                }
                for name, points in enrichment_curves.items()
            },
            "random_baseline": {
                "fractions": [0, 1],
                "enrichment": [1, 1]
            }
        }
        
        return PlotData(
            title="Enrichment Curves",
            x_label="Fraction of Database",
            y_label="Enrichment Factor",
            data=data,
            plot_type="line"
        )
    
    def generate_uncertainty_error_scatter(
        self,
        uncertainties: np.ndarray,
        errors: np.ndarray,
        method_name: str
    ) -> PlotData:
        """
        Generate uncertainty vs error scatter plot.
        
        Args:
            uncertainties: Predicted uncertainties
            errors: Actual prediction errors
            method_name: Method name for title
            
        Returns:
            PlotData object
        """
        correlation = np.corrcoef(uncertainties, np.abs(errors))[0, 1]
        
        data = {
            "uncertainties": uncertainties.tolist(),
            "errors": np.abs(errors).tolist(),
            "correlation": correlation,
            "method": method_name,
            "trend_line": self._calculate_trend_line(uncertainties, np.abs(errors))
        }
        
        return PlotData(
            title=f"Uncertainty-Error Correlation (r={correlation:.3f}) - {method_name}",
            x_label="Predicted Uncertainty",
            y_label="Absolute Error",
            data=data,
            plot_type="scatter_with_trend"
        )
    
    def _calculate_trend_line(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_points: int = 100
    ) -> List[List[float]]:
        """Calculate linear trend line."""
        from scipy import stats
        
        slope, intercept, _, _, _ = stats.linregress(x, y)
        
        x_range = np.linspace(x.min(), x.max(), n_points)
        y_range = slope * x_range + intercept
        
        return [[float(xi), float(yi)] for xi, yi in zip(x_range, y_range)]
    
    def to_plotly_json(self, plot_data: PlotData) -> str:
        """Convert plot data to Plotly-compatible JSON."""
        plotly_config = {
            "data": self._convert_to_plotly_format(plot_data),
            "layout": {
                "title": {"text": plot_data.title},
                "xaxis": {"title": plot_data.x_label},
                "yaxis": {"title": plot_data.y_label},
                "template": "plotly_white" if self.theme == "light" else "plotly_dark"
            }
        }
        
        return json.dumps(plotly_config)
    
    def _convert_to_plotly_format(self, plot_data: PlotData) -> List[Dict]:
        """Convert internal format to Plotly traces."""
        traces = []
        
        if plot_data.plot_type == "violin":
            for bench, stats in plot_data.data["scores"].items():
                traces.append({
                    "type": "violin",
                    "name": bench,
                    "y": stats["values"],
                    "box": {"visible": True},
                    "meanline": {"visible": True}
                })
        
        elif plot_data.plot_type == "radar":
            for method, scores in plot_data.data["methods"].items():
                values = [scores[cat] for cat in plot_data.data["categories"]]
                values.append(values[0])  # Close the loop
                
                traces.append({
                    "type": "scatterpolar",
                    "name": method,
                    "r": values,
                    "theta": plot_data.data["categories"] + [plot_data.data["categories"][0]],
                    "fill": "toself"
                })
        
        elif plot_data.plot_type == "horizontal_bar":
            traces.append({
                "type": "bar",
                "orientation": "h",
                "y": plot_data.data["teams"],
                "x": plot_data.data["scores"],
                "marker": {"color": plot_data.data["colors"]},
                "text": [f"{s:.3f}" for s in plot_data.data["scores"]],
                "textposition": "outside"
            })
        
        return traces


class DashboardGenerator:
    """
    Generate comprehensive benchmark dashboards.
    
    Creates interactive HTML dashboards with multiple visualizations.
    """
    
    def __init__(self, title: str = "VS-Bench Dashboard"):
        self.title = title
        self.visualizer = BenchmarkVisualizer()
    
    def generate_full_dashboard(
        self,
        leaderboard_data: List[Dict],
        evaluation_results: Dict[str, Any],
        output_path: str
    ) -> str:
        """
        Generate complete HTML dashboard.
        
        Args:
            leaderboard_data: Current leaderboard
            evaluation_results: Evaluation results
            output_path: Output HTML file path
            
        Returns:
            Path to generated dashboard
        """
        # Generate individual plots
        plots = []
        
        # Leaderboard
        leaderboard_plot = self.visualizer.generate_leaderboard_chart(leaderboard_data)
        plots.append(("leaderboard", leaderboard_plot))
        
        # Radar chart for top methods
        if len(leaderboard_data) >= 3:
            top_methods = {
                e["team"]: e.get("scores", {})
                for e in leaderboard_data[:3]
            }
            categories = list(leaderboard_data[0].get("scores", {}).keys())
            if categories:
                radar_plot = self.visualizer.generate_radar_chart(top_methods, categories)
                plots.append(("comparison", radar_plot))
        
        # Build HTML
        html = self._build_dashboard_html(plots)
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(html)
        
        return output_path
    
    def _build_dashboard_html(self, plots: List[Tuple[str, PlotData]]) -> str:
        """Build complete dashboard HTML."""
        plot_divs = ""
        for plot_id, plot_data in plots:
            plot_divs += f"""
            <div class="plot-container">
                <h2>{plot_data.title}</h2>
                <div id="{plot_id}" class="plot"></div>
            </div>
            """
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{self.title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .plot-container {{
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .plot {{
            width: 100%;
            height: 500px;
        }}
        h1, h2 {{
            margin-top: 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{self.title}</h1>
        <p>Interactive Benchmark Analytics Dashboard</p>
    </div>
    
    {plot_divs}
    
    <script>
        // Plot data will be embedded here
    </script>
</body>
</html>"""
        
        return html


# Export
__all__ = [
    'PlotData',
    'BenchmarkVisualizer',
    'DashboardGenerator'
]

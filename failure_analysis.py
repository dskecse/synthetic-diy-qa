import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from failure_modes import FailureModeDefinitions

class FailureAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.failure_modes = [failure_mode.name for failure_mode in FailureModeDefinitions.failure_modes()]
        self.correlation_matrix = df[self.failure_modes].corr()

    def _generate_failure_summary(self) -> dict[str, any]:
        """
        Generate summary statistics for failure modes

        Return dictionary with failure statistics.
        """
        summary = {
            "total_samples": len(self.df),
            "overall_failure_rate": self.df["overall_failure"].mean(),
            "overall_success_rate": 1 - self.df["overall_failure"].mean(),
            "failure_mode_rates": {},
            "failure_mode_counts": {},
            "most_common_failures": [],
            "least_common_failures": []
        }

        # Calculate rates and counts for each failure mode
        for mode in self.failure_modes:
            rate = self.df[mode].mean()
            count = self.df[mode].sum()
            summary["failure_mode_rates"][mode] = rate
            summary["failure_mode_counts"][mode] = count

        # Sort failure modes by frequency
        sorted_modes = sorted(self.failure_modes,
                              key=lambda mode: summary["failure_mode_rates"][mode],
                              reverse=True)

        summary["most_common_failures"] = sorted_modes[:3]
        summary["least_common_failures"] = sorted_modes[-3:]

        return summary

    def print_summary_report(self):
        summary = self._generate_failure_summary()

        print("=" * 60)
        print("FAILURE ANALYSIS SUMMARY REPORT:")
        print("=" * 60)
        print(f"Total samples: {summary['total_samples']}")
        print(f"Overall Success Rate: {summary['overall_success_rate']:.1%}")
        print(f"Overall Failure Rate: {summary['overall_failure_rate']:.1%}")

        print(f"\nFAILURE MODE BREAKDOWN:")
        print("-" * 40)
        for mode in self.failure_modes:
            rate = summary["failure_mode_rates"][mode]
            count = summary["failure_mode_counts"][mode]
            print(f"{mode.replace('_', ' ').title():25}: {rate:6.1%} ({count:2d}/{summary['total_samples']})")

        print(f"\nMOST PROBLEMATIC AREAS:")
        print("-" * 40)
        for i, mode in enumerate(summary["most_common_failures"], start=1):
            rate = summary["failure_mode_rates"][mode]
            print(f"{i}. {mode.replace('_', ' ').title()}: {rate:.1%}")

    def create_failure_heatmap(self, filename: str = "failure_heatmap.png"):
        """
        Create a heatmap of failure modes across samples
        """
        # Create failure mode matrix
        failure_matrix = self.df[self.failure_modes].values

        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(failure_matrix.T,
                    xticklabels=[f"Sample {i+1}" for i in range(len(self.df))],
                    yticklabels=[mode.replace('_', ' ').title() for mode in self.failure_modes],
                    cmap="RdYlBu_r",
                    cbar_kws={"label": "Failure (1) / Success (0)"},
                    annot=True,
                    fmt="d")

        plt.title("Failure Mode Heatmap Across All Samples", fontsize=16, fontweight="bold")
        plt.xlabel("Sample ID", fontsize=12)
        plt.ylabel("Failure Modes", fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        os.makedirs("assets", exist_ok=True)
        filepath = os.path.join("assets", filename)

        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"Heatmap saved to {filepath}")

    def create_failure_rate_chart(self, filename: str = "failure_rates.png"):
        """
        Create a bar chart showing failure rates by failure mode
        """
        failure_rates = [self.df[mode].mean() for mode in self.failure_modes]
        mode_labels = [mode.replace("_", " ").title() for mode in self.failure_modes]

        plt.figure(figsize=(12, 6))
        bars = plt.bar(mode_labels, failure_rates, color="lightcoral", alpha=0.7)

        # Add value labels on bars
        for bar, rate in zip(bars, failure_rates):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{rate:.1%}", ha="center", va="bottom", fontweight="bold")

        plt.title("Failure Rates by Mode", fontsize=16, fontweight="bold")
        plt.xlabel("Failure Modes", fontsize=12)
        plt.ylabel("Failure Rate", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.ylim(0, 1)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        os.makedirs("assets", exist_ok=True)
        filepath = os.path.join("assets", filename)

        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"Failure rates chart saved to {filepath}")

    def analyze_correlations(self, filename: str = "failure_correlations.png"):
        """
        Analyze correlations between failure modes
        """
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.correlation_matrix,
                    annot=True,
                    cmap="coolwarm",
                    center=0,
                    square=True,
                    xticklabels=[mode.replace("_", " ").title() for mode in self.failure_modes],
                    yticklabels=[mode.replace("_", " ").title() for mode in self.failure_modes])

        plt.title("Failure Mode Correlations", fontsize=16, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        os.makedirs("assets", exist_ok=True)
        filepath = os.path.join("assets", filename)

        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"Correlation heatmap saved to {filepath}")

        return self.correlation_matrix

    def _pattern_to_name(self, pattern: tuple[int, ...]) -> str:
        """
        Convert failure pattern tuple (0s and 1s representing failure modes) to readable name

        Returns human-readable pattern name.
        """
        if all(x == 0 for x in pattern):
            return "No Failures"

        failed_modes = [self.failure_modes[i] for i, x in enumerate(pattern) if x == 1]
        return " + ".join([mode.replace('_', ' ').title() for mode in failed_modes])

    def _identify_failure_patterns(self) -> dict[str, list[int]]:
        """
        Identify common failure patterns across samples

        Returns a dictionary mapping failure patterns to sample indices.
        """
        patterns = {}

        for idx, row in self.df.iterrows():
            # Create pattern signature
            pattern = tuple(row[mode] for mode in self.failure_modes)
            pattern_name = self._pattern_to_name(pattern)

            if pattern_name not in patterns:
                patterns[pattern_name] = []
            patterns[pattern_name].append(idx)

        # Sort by frequency
        sorted_patterns = dict(sorted(patterns.items(), key=lambda x: len(x[1]), reverse=True))

        return sorted_patterns

    def _generate_recommendations(self, summary: dict, patterns: dict) -> list[str]:
        """
        Generate recommendations based on analysis results

        Args:
            summary: Failure summary statistics
            patterns: Failure pattern analysis

        Returns a list of recommendation strings.
        """
        recommendations = []

        # Most common failure recommendations
        most_common = summary["most_common_failures"][0]
        recommendations.append(f"Focus on improving '{most_common.replace('_', ' ')}' - it's the most common failure mode ({summary['failure_mode_rates'][most_common]:.1%} failure rate)")

        # Pattern-based recommendations
        if "No Failures" in patterns and len(patterns["No Failures"]) > 0:
            success_rate = len(patterns["No Failures"]) / summary["total_samples"]
            recommendations.append(f"Good news: {success_rate:.1%} of samples have no failures - analyze these for best practices")

        # Correlation-based recommendations
        if summary["overall_failure_rate"] > 0.5:
            recommendations.append("High overall failure rate suggests need for better prompt engineering or model fine-tuning")

        return recommendations

    def generate_detailed_report(self, filename: str = "failure_analysis_report.json"):
        """
        Generate comprehensive failure analysis report
        """
        summary = self._generate_failure_summary()
        correlations = self.correlation_matrix # failure mode correlation matrix
        patterns = self._identify_failure_patterns()

        report = {
            "summary": summary,
            "correlations": correlations,
            "failure_patterns": {k: len(v) for k, v in patterns.items()},
            "detailed_patterns": patterns,
            "recommendations": self._generate_recommendations(summary, patterns)
        }

        os.makedirs("data", exist_ok=True)
        filepath = os.path.join("data", filename)

        # Save report
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False, default=str)

        print(f"Detailed analysis report saved to {filepath}")
        return report

def _load_failure_labeled_data(filename: str) -> pd.DataFrame:
    # Create a directory (do not raise if already present)
    os.makedirs("data", exist_ok=True)

    try:
        filepath = os.path.join("data", filename)
        return pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Please run failure labeling phase first or provide the correct path.")
        sys.exit(1)

def run_failure_analysis(filename: str = "failure_labeled_data.csv"):
    """
    Run complete failure analysis of the failure labeled data
    """
    df = _load_failure_labeled_data(filename=filename)
    print(f"Loaded {len(df)} failure labeled samples")

    # Create analyzer
    analyzer = FailureAnalyzer(df)

    # Generate failure analysis
    print("\nGenerating Failure Analysis...")
    analyzer.print_summary_report()

    # Create visualizations
    analyzer.create_failure_heatmap()
    analyzer.create_failure_rate_chart()
    analyzer.analyze_correlations()

    print("\nGENERATING DETAILED REPORT...")
    report = analyzer.generate_detailed_report()

    print("\nRECOMMENDATIONS:")
    for i, recommendation in enumerate(report["recommendations"], 1):
        print(f"{i}. {recommendation}")

    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    run_failure_analysis()

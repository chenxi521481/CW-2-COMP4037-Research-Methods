import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
import matplotlib.patheffects as path_effects

# ======================
# Professional Visualization Parameters Configuration
# ======================
PROFESSIONAL_STYLE = {
    'figure.figsize': (14, 14),
    'font.family': 'Arial',
    'axes.titlesize': 16,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'legend.title_fontsize': 12,
    'grid.color': '#E0E0E0',
    'grid.alpha': 0.7,
    'axes.edgecolor': '#3E3E3E',
    'axes.linewidth': 1.2,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white'
}

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update(PROFESSIONAL_STYLE)


# ======================
# Data Preparation (modified normalization method)
# ======================
def load_and_prepare_data(filepath):
    """Load and prepare radar chart data"""
    try:
        df_raw = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file '{filepath}' not found, please check the file path")

    # Define analysis metrics
    value_vars = [
        'mean_ghgs', 'mean_land', 'mean_watscar', 'mean_eut',
        'mean_ghgs_ch4', 'mean_ghgs_n2o', 'mean_bio', 'mean_watuse', 'mean_acid'
    ]

    # Validate data columns
    missing_cols = [col for col in value_vars if col not in df_raw.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in data file: {missing_cols}")

    # Calculate group averages
    df_avg = df_raw.groupby("diet_group")[value_vars].mean().reset_index()

    # Convert to long format
    df_melt = df_avg.melt(id_vars="diet_group", var_name="variable", value_name="value")

    # Calculate angles
    variables = df_melt["variable"].unique()
    n = len(variables)
    angle_map = {var: i * 2 * np.pi / n for i, var in enumerate(variables)}
    df_melt["angle"] = df_melt["variable"].map(angle_map)

    # Modified normalization method - avoid all zeros for minimum values
    # Use 95% of global minimum as baseline to preserve vegan data visibility
    def safe_normalize(x):
        x_min = x.min()
        x_max = x.max()
        # If minimum is 0, use 95% of global minimum as baseline
        if x_min == 0:
            adjusted_min = 0.95 * x.min()
        else:
            adjusted_min = 0.95 * x_min
        return (x - adjusted_min) / (x_max - adjusted_min)

    df_melt["value_norm"] = df_melt.groupby("variable")["value"].transform(safe_normalize)

    # Ensure polygon closure
    df_closed = []
    for group in df_melt["diet_group"].unique():
        df_group = df_melt[df_melt["diet_group"] == group].sort_values("angle")
        df_first = df_group.iloc[0].copy()
        df_closed.append(pd.concat([df_group, pd.DataFrame([df_first])]))

    df_final = pd.concat(df_closed)

    # Calculate descriptive statistics of raw values
    desc_stats = df_raw[value_vars].describe().T.reset_index()
    desc_stats.columns = ['variable'] + [f'raw_{col}' for col in desc_stats.columns[1:]]

    # Merge with raw data statistics
    df_final = pd.merge(df_final, desc_stats, on='variable', how='left')

    return df_final, df_raw.shape[0], df_avg


# ======================
# Data Export Function
# ======================
def export_radar_data(df_final, df_avg, export_path):
    """Export radar chart data"""

    # Export original average data
    df_avg.to_csv(f"{export_path}_diet_group_averages.csv", index=False)

    # Export transformed radar chart data
    radar_data = df_final.copy()

    # Add variable order information
    variables = df_final["variable"].unique()
    var_order = {var: i for i, var in enumerate(variables)}
    radar_data["variable_order"] = radar_data["variable"].map(var_order)

    # Reorder columns
    cols = ['diet_group', 'variable', 'variable_order', 'angle',
            'value', 'value_norm', 'raw_count', 'raw_mean', 'raw_std',
            'raw_min', 'raw_25%', 'raw_50%', 'raw_75%', 'raw_max']
    cols = [c for c in cols if c in radar_data.columns]

    radar_data[cols].to_csv(f"{export_path}_radar_transformed_data.csv", index=False)

    print(f"Data successfully exported to {export_path}_*.csv files")


# ======================
# Visualization Function (enhanced vegan group visibility)
# ======================
def create_professional_radar(df_final, total_samples, save_path=None):
    """Create professional radar chart"""
    # Create figure
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111, polar=True)

    # Get unique diet groups and colors
    groups = df_final["diet_group"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))

    # Set more prominent color for vegan group
    vegan_index = np.where(groups == 'vegan')[0][0] if 'vegan' in groups else -1
    if vegan_index >= 0:
        colors[vegan_index] = to_rgba('#FF0000', 1.0)  # Highlight in red

    # ======================
    # Radar Chart Basic Settings
    # ======================
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 1.15)

    # Set grid and ticks
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], color='#555555')
    ax.grid(color='#E0E0E0', linestyle='--', linewidth=0.8, alpha=0.8)

    # ======================
    # Axis Label Settings
    # ======================
    variables = df_final["variable"].unique()
    n = len(variables)
    ax.set_xticks(np.linspace(0, 2 * np.pi, n, endpoint=False))

    # Define metric units
    units = {
        'ghgs': '(kg CO2-eq/day)',
        'land': '(m²/year)',
        'watscar': '(L/day)',
        'eut': '(g PO4-eq/day)',
        'ghgs_ch4': '(kg CO2-eq/day)',
        'ghgs_n2o': '(kg CO2-eq/day)',
        'bio': '(species·year)',
        'watuse': '(L/day)',
        'acid': '(g SO2-eq/day)'
    }

    # Create labels with units
    xtick_labels = []
    for v in variables:
        key = v.replace('mean_', '')
        label = f"{key.replace('_', ' ').title()}\n{units.get(key, '')}"
        xtick_labels.append(label)

    # Set label styles
    xticks = ax.set_xticklabels(
        xtick_labels,
        fontsize=11,
        color='#2A2A2A',
        fontweight='semibold'
    )

    # Add label effects for better readability
    for label in xticks:
        label.set_path_effects([
            path_effects.withStroke(linewidth=3, foreground="white", alpha=0.7)
        ])

    # ======================
    # Plot Data (special handling for vegan group)
    # ======================
    line_styles = ['-', '--', '-.', ':']
    marker_styles = ['o', 's', 'D', '^', 'v']

    for i, group in enumerate(groups):
        sub_df = df_final[df_final["diet_group"] == group]
        values = sub_df["value_norm"].values[:-1]
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False))

        # Special handling for vegan group fill transparency
        fill_alpha = 0.25
        if group == 'vegan':
            fill_alpha = 0.4  # Increase fill transparency
            line_width = 3.0  # Increase line width
            marker_size = 10  # Increase marker size
        else:
            line_width = 2.5
            marker_size = 8

        # Draw filled area
        ax.fill(
            angles, values,
            color=to_rgba(colors[i], alpha=fill_alpha),
            linewidth=0
        )

        # Draw lines and markers
        ax.plot(
            angles, values,
            color=colors[i],
            linewidth=line_width,
            linestyle=line_styles[i % len(line_styles)],
            marker=marker_styles[i % len(marker_styles)],
            markersize=marker_size,
            markeredgecolor='white',
            markeredgewidth=1.5 if group == 'vegan' else 1,  # Thicker border for vegan
            label=group,
            zorder=3 if group == 'vegan' else 2  # Vegan group on top layer
        )

    # ======================
    # Add Professional Legend (highlight vegan)
    # ======================
    # Create sample size info
    sample_info = f"Total Samples: {total_samples:,}"

    # Create legend elements
    legend_elements = []
    for i, group in enumerate(groups):
        # Special mark for vegan group
        if group == 'vegan':
            label = f"{group} (lowest impact)"
            marker_edge_width = 1.5
        else:
            label = group
            marker_edge_width = 1

        legend_elements.append(
            Line2D([0], [0],
                   color=colors[i],
                   marker=marker_styles[i % len(marker_styles)],
                   markersize=9,
                   markeredgecolor='white',
                   markeredgewidth=marker_edge_width,
                   linewidth=2.5,
                   linestyle=line_styles[i % len(line_styles)],
                   label=label)
        )

    # Add main legend
    legend1 = ax.legend(
        handles=legend_elements,
        loc='upper right',
        bbox_to_anchor=(1.35, 1.0),
        title="Diet Groups",
        frameon=True,
        framealpha=0.95,
        borderpad=1
    )
    legend1.get_frame().set_edgecolor('#CCCCCC')

    # Add statistics info
    stats_text = "\n".join([
        "Data Characteristics:",
        f"- Total participants: {total_samples:,}",
        "- Values normalized (adjusted min)",
        "- Data collected: March 2022",
        "- Environmental impact metrics",
        "- Vegan group highlighted in red"
    ])

    fig.text(
        0.95, 0.15, stats_text,
        ha='right', va='bottom',
        fontsize=10,
        bbox=dict(
            facecolor='white',
            alpha=0.8,
            edgecolor='#DDDDDD',
            boxstyle='round,pad=0.5'
        )
    )

    # ======================
    # Add Center Title and Decorations
    # ======================
    # Add center circle
    center_circle = Circle((0, 0), 0.1, transform=ax.transData._b,
                           facecolor='white', edgecolor='#AAAAAA', alpha=0.8)
    ax.add_patch(center_circle)

    # Add center title
    ax.text(
        0, 0, "Environmental\nImpact\nAssessment",
        ha='center', va='center',
        fontsize=14,
        fontweight='bold',
        color='#555555'
    )

    # Add scale arrow
    arrowprops = dict(
        arrowstyle="->",
        color="#5F5F5F",
        linewidth=1.5,
        connectionstyle="angle3,angleA=0,angleB=90"
    )
    ax.annotate('Higher Impact', xy=(0.5, 1.05), xycoords='axes fraction',
                ha='center', fontsize=10, color='#5F5F5F')
    ax.annotate('', xy=(0.7, 1.02), xytext=(0.3, 1.02),
                xycoords='axes fraction', arrowprops=arrowprops)

    # ======================
    # Add Main Title and Source
    # ======================
    plt.suptitle(
        "Comparative Environmental Impact of Dietary Patterns",
        y=1.05,
        fontsize=18,
        fontweight='bold'
    )

    plt.title(
        "Normalized Radar Chart of Nine Environmental Indicators by Diet Group",
        pad=20,
        fontsize=14
    )

    # Add source information
    fig.text(
        0.95, 0.02,
        "Data Source: Results_21MAR2022_nokcaladjust.csv | "
        "Visualization: Python/Matplotlib | "
        "Author: Xi Chen",
        ha='right',
        fontsize=9,
        color='#777777'
    )

    # ======================
    # Final Adjustments and Saving
    # ======================
    plt.tight_layout(pad=4.0)
    plt.subplots_adjust(right=0.75)  # Make space for legend

    if save_path:
        formats = ['png', 'svg', 'pdf']
        for fmt in formats:
            plt.savefig(
                f"{save_path}_radar_chart.{fmt}",
                dpi=300,
                bbox_inches='tight',
                facecolor='white'
            )

    plt.show()


# ======================
# Execute Data Preparation and Visualization
# ======================
if __name__ == "__main__":
    # Load data
    df_final, total_samples, df_avg = load_and_prepare_data("Results_21MAR2022_nokcaladjust.csv")

    # Export data
    export_radar_data(df_final, df_avg, "diet_environmental_impact")

    # Create visualization
    create_professional_radar(df_final, total_samples, save_path="diet_environmental_impact")

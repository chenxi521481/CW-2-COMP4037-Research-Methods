import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
import matplotlib.patheffects as path_effects

# ======================
# 专业可视化参数配置
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
# 数据准备 (修改了标准化方法)
# ======================
def load_and_prepare_data(filepath):
    """加载并准备雷达图数据"""
    try:
        df_raw = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"数据文件 '{filepath}' 未找到，请检查文件路径")

    # 定义分析指标
    value_vars = [
        'mean_ghgs', 'mean_land', 'mean_watscar', 'mean_eut',
        'mean_ghgs_ch4', 'mean_ghgs_n2o', 'mean_bio', 'mean_watuse', 'mean_acid'
    ]

    # 验证数据列
    missing_cols = [col for col in value_vars if col not in df_raw.columns]
    if missing_cols:
        raise ValueError(f"数据文件中缺少以下列: {missing_cols}")

    # 计算分组平均值
    df_avg = df_raw.groupby("diet_group")[value_vars].mean().reset_index()

    # 转换为长格式
    df_melt = df_avg.melt(id_vars="diet_group", var_name="variable", value_name="value")

    # 计算角度
    variables = df_melt["variable"].unique()
    n = len(variables)
    angle_map = {var: i * 2 * np.pi / n for i, var in enumerate(variables)}
    df_melt["angle"] = df_melt["variable"].map(angle_map)

    # 修改标准化方法 - 避免最小值全为0的问题
    # 使用全局最小值的95%作为基准，保留vegan数据的可见性
    def safe_normalize(x):
        x_min = x.min()
        x_max = x.max()
        # 如果最小值是0，使用全局最小值的95%作为基准
        if x_min == 0:
            adjusted_min = 0.95 * x.min()
        else:
            adjusted_min = 0.95 * x_min
        return (x - adjusted_min) / (x_max - adjusted_min)

    df_melt["value_norm"] = df_melt.groupby("variable")["value"].transform(safe_normalize)

    # 确保多边形闭合
    df_closed = []
    for group in df_melt["diet_group"].unique():
        df_group = df_melt[df_melt["diet_group"] == group].sort_values("angle")
        df_first = df_group.iloc[0].copy()
        df_closed.append(pd.concat([df_group, pd.DataFrame([df_first])]))

    df_final = pd.concat(df_closed)

    # 计算原始值的描述统计
    desc_stats = df_raw[value_vars].describe().T.reset_index()
    desc_stats.columns = ['variable'] + [f'raw_{col}' for col in desc_stats.columns[1:]]

    # 合并原始数据统计
    df_final = pd.merge(df_final, desc_stats, on='variable', how='left')

    return df_final, df_raw.shape[0], df_avg


# ======================
# 数据导出函数
# ======================
def export_radar_data(df_final, df_avg, export_path):
    """导出雷达图使用的数据"""

    # 导出原始平均值数据
    df_avg.to_csv(f"{export_path}_diet_group_averages.csv", index=False)

    # 导出雷达图转换后的数据
    radar_data = df_final.copy()

    # 添加变量顺序信息
    variables = df_final["variable"].unique()
    var_order = {var: i for i, var in enumerate(variables)}
    radar_data["variable_order"] = radar_data["variable"].map(var_order)

    # 重新排列列
    cols = ['diet_group', 'variable', 'variable_order', 'angle',
            'value', 'value_norm', 'raw_count', 'raw_mean', 'raw_std',
            'raw_min', 'raw_25%', 'raw_50%', 'raw_75%', 'raw_max']
    cols = [c for c in cols if c in radar_data.columns]

    radar_data[cols].to_csv(f"{export_path}_radar_transformed_data.csv", index=False)

    print(f"数据已成功导出到 {export_path}_*.csv 文件")


# ======================
# 可视化函数 (增强vegan组的可见性)
# ======================
def create_professional_radar(df_final, total_samples, save_path=None):
    """创建专业级雷达图"""
    # 创建图形
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111, polar=True)

    # 获取唯一饮食组和颜色
    groups = df_final["diet_group"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))

    # 特别为vegan组设置更醒目的颜色
    vegan_index = np.where(groups == 'vegan')[0][0] if 'vegan' in groups else -1
    if vegan_index >= 0:
        colors[vegan_index] = to_rgba('#FF0000', 1.0)  # 红色突出显示

    # ======================
    # 雷达图基础设置
    # ======================
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 1.15)

    # 设置网格和刻度
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], color='#555555')
    ax.grid(color='#E0E0E0', linestyle='--', linewidth=0.8, alpha=0.8)

    # ======================
    # 轴标签设置
    # ======================
    variables = df_final["variable"].unique()
    n = len(variables)
    ax.set_xticks(np.linspace(0, 2 * np.pi, n, endpoint=False))

    # 定义指标单位
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

    # 创建带单位的标签
    xtick_labels = []
    for v in variables:
        key = v.replace('mean_', '')
        label = f"{key.replace('_', ' ').title()}\n{units.get(key, '')}"
        xtick_labels.append(label)

    # 设置标签样式
    xticks = ax.set_xticklabels(
        xtick_labels,
        fontsize=11,
        color='#2A2A2A',
        fontweight='semibold'
    )

    # 添加标签效果增强可读性
    for label in xticks:
        label.set_path_effects([
            path_effects.withStroke(linewidth=3, foreground="white", alpha=0.7)
        ])

    # ======================
    # 绘制数据 (特别处理vegan组)
    # ======================
    line_styles = ['-', '--', '-.', ':']
    marker_styles = ['o', 's', 'D', '^', 'v']

    for i, group in enumerate(groups):
        sub_df = df_final[df_final["diet_group"] == group]
        values = sub_df["value_norm"].values[:-1]
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

        # 特别处理vegan组的填充透明度
        fill_alpha = 0.25
        if group == 'vegan':
            fill_alpha = 0.4  # 增加填充透明度
            line_width = 3.0  # 增加线宽
            marker_size = 10  # 增加标记点大小
        else:
            line_width = 2.5
            marker_size = 8

        # 绘制填充区域
        ax.fill(
            angles, values,
            color=to_rgba(colors[i], alpha=fill_alpha),
            linewidth=0
        )

        # 绘制线条和标记点
        ax.plot(
            angles, values,
            color=colors[i],
            linewidth=line_width,
            linestyle=line_styles[i % len(line_styles)],
            marker=marker_styles[i % len(marker_styles)],
            markersize=marker_size,
            markeredgecolor='white',
            markeredgewidth=1.5 if group == 'vegan' else 1,  # vegan组边框更粗
            label=group,
            zorder=3 if group == 'vegan' else 2  # vegan组在最上层
        )

    # ======================
    # 添加专业图例 (突出显示vegan)
    # ======================
    # 创建样本量统计
    sample_info = f"Total Samples: {total_samples:,}"

    # 创建图例元素
    legend_elements = []
    for i, group in enumerate(groups):
        # 特别标记vegan组
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

    # 添加主图例
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

    # 添加统计信息
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
    # 添加中心标题和装饰
    # ======================
    # 添加中心圆
    center_circle = Circle((0, 0), 0.1, transform=ax.transData._b,
                           facecolor='white', edgecolor='#AAAAAA', alpha=0.8)
    ax.add_patch(center_circle)

    # 添加中心标题
    ax.text(
        0, 0, "Environmental\nImpact\nAssessment",
        ha='center', va='center',
        fontsize=14,
        fontweight='bold',
        color='#555555'
    )

    # 添加比例箭头
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
    # 添加主标题和来源
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

    # 添加来源信息
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
    # 最终调整和保存
    # ======================
    plt.tight_layout(pad=4.0)
    plt.subplots_adjust(right=0.75)  # 为图例留出空间

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
# 执行数据准备和可视化
# ======================
if __name__ == "__main__":
    # 加载数据
    df_final, total_samples, df_avg = load_and_prepare_data("Results_21MAR2022_nokcaladjust.csv")

    # 导出数据
    export_radar_data(df_final, df_avg, "diet_environmental_impact")

    # 创建可视化
    create_professional_radar(df_final, total_samples, save_path="diet_environmental_impact")
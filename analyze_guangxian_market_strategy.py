"""冠县风电场日前信息：价格相关性、分时仓位和实时尖峰诊断。

依赖：pandas、openpyxl、numpy、matplotlib
输出：3张PNG、2个CSV。所有金额单位为元/MWh，功率单位为MW。
"""

from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_ROOT = (
    ROOT
    / "CLP_Wind_Farm_Data_Guangxian_20260713"
    / "CLP_Wind_Farm_Data_Guangxian_20260713"
)

FIGURE_1 = ROOT / "01_典型曲线_火电竞价空间相关性.png"
FIGURE_2 = ROOT / "02_全样本分时价差与仓位倾向.png"
FIGURE_3 = ROOT / "03_实时价格突跳日_原因诊断_2026-04-19.png"
CSV_INTRADAY = ROOT / "全样本_15分钟价差统计.csv"
CSV_SPIKE = ROOT / "实时价格突跳日_诊断明细_2026-04-19.csv"

TYPICAL_DAYS = [
    ("2026-04-10", "典型盆形电价日"),
    ("2026-06-19", "相对平直电价日"),
]
SPIKE_DAY = "2026-04-19"


def configure_plotting() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.8


def load_data() -> pd.DataFrame:
    files = sorted(DATA_ROOT.glob("冠县风电场_日前信息_*.xlsx"))
    if not files:
        raise FileNotFoundError(f"未在{DATA_ROOT}找到冠县风电场Excel数据。")

    frames = []
    for path in files:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Workbook contains no default style"
            )
            frame = pd.read_excel(path, sheet_name="Sheet0")

        # 每天第96个时点写作24:00。保留原日期，并按行序生成0.25至24.00小时。
        frame["日期"] = frame["时间"].astype(str).str.slice(0, 10)
        frame["时点序号"] = frame.groupby("日期").cumcount() + 1
        frame["小时"] = frame["时点序号"] / 4.0
        frame["来源文件"] = path.name
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    required = [
        "统一结算点电价-日前",
        "统一结算点电价-实时",
        "火电竞价空间-出清前",
        "火电竞价空间-实际",
        "直调负荷-出清前",
        "直调负荷-实际",
        "新能源总加-出清前",
        "新能源总加-实际",
        "风电总加-出清前",
        "风电总加-实际",
        "光伏总加-出清前",
        "光伏总加-实际",
        "联络线受电负荷-出清前",
        "联络线受电负荷-实际",
    ]
    missing_columns = [column for column in required if column not in data.columns]
    if missing_columns:
        raise KeyError(f"缺少必要字段：{missing_columns}")
    if data[required].isna().any().any():
        nulls = data[required].isna().sum()
        raise ValueError(f"关键字段存在缺失值：\n{nulls[nulls > 0]}")

    points_per_day = data.groupby("日期").size()
    if not points_per_day.eq(96).all():
        raise ValueError(f"存在非96点日期：\n{points_per_day[points_per_day != 96]}")

    data["日前减实时价差"] = (
        data["统一结算点电价-日前"] - data["统一结算点电价-实时"]
    )
    data["实时减日前价差"] = -data["日前减实时价差"]
    data["负荷预测误差"] = data["直调负荷-实际"] - data["直调负荷-出清前"]
    data["新能源预测误差"] = (
        data["新能源总加-实际"] - data["新能源总加-出清前"]
    )
    data["风电预测误差"] = data["风电总加-实际"] - data["风电总加-出清前"]
    data["光伏预测误差"] = data["光伏总加-实际"] - data["光伏总加-出清前"]
    data["省际受电预测误差"] = (
        data["联络线受电负荷-实际"] - data["联络线受电负荷-出清前"]
    )
    data["火电竞价空间预测误差"] = (
        data["火电竞价空间-实际"] - data["火电竞价空间-出清前"]
    )

    # 数据中的火电竞价空间严格满足：直调负荷 - 新能源 - 省际受电。
    boundary_residual = data["火电竞价空间预测误差"] - (
        data["负荷预测误差"]
        - data["新能源预测误差"]
        - data["省际受电预测误差"]
    )
    if boundary_residual.abs().max() > 1e-6:
        raise ValueError("火电竞价空间与负荷、新能源、省际受电的边界恒等式不闭合。")
    return data


def pearson(data: pd.DataFrame, x: str, y: str) -> float:
    return float(data[x].corr(data[y]))


def spearman(data: pd.DataFrame, x: str, y: str) -> float:
    return float(data[x].rank().corr(data[y].rank()))


def plot_typical_days(data: pd.DataFrame) -> dict[str, dict[str, float]]:
    colors = {
        "da": "#1F4F8C",
        "rt": "#C7382E",
        "wind": "#1C855C",
        "solar": "#ED941A",
        "thermal": "#76518A",
    }
    correlations: dict[str, dict[str, float]] = {}
    fig, axes = plt.subplots(2, 1, figsize=(15, 9.6))
    fig.subplots_adjust(left=0.075, right=0.925, top=0.91, bottom=0.07, hspace=0.43)

    for ax, (date, kind) in zip(axes, TYPICAL_DAYS):
        day = data.loc[data["日期"] == date].sort_values("小时")
        ax_power = ax.twinx()
        x = day["小时"]

        line_da, = ax.plot(
            x, day["统一结算点电价-日前"], color=colors["da"], lw=2.2, label="日前价格"
        )
        line_rt, = ax.plot(
            x, day["统一结算点电价-实时"], color=colors["rt"], lw=2.2, label="实时价格"
        )
        line_wind, = ax_power.plot(
            x, day["风电总加-实际"], color=colors["wind"], lw=1.7, ls="--", label="风电实际"
        )
        line_solar, = ax_power.plot(
            x,
            day["光伏总加-实际"].clip(lower=0),
            color=colors["solar"],
            lw=1.7,
            ls="--",
            label="光伏实际",
        )
        line_thermal, = ax_power.plot(
            x,
            day["火电竞价空间-实际"],
            color=colors["thermal"],
            lw=1.9,
            ls="-.",
            label="火电竞价空间-实际",
        )

        r_da = pearson(day, "统一结算点电价-日前", "火电竞价空间-出清前")
        r_rt = pearson(day, "统一结算点电价-实时", "火电竞价空间-实际")
        correlations[date] = {"r_da": r_da, "r_rt": r_rt}

        ax.axhline(0, color="#666666", lw=1, ls=":")
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 2))
        ax.set_xlabel("时刻")
        ax.set_ylabel("统一结算点电价（元/MWh）")
        ax_power.set_ylabel("全省出力/竞价空间（MW）")
        ax_power.set_ylim(bottom=0)
        ax.grid(True, alpha=0.16)
        ax.set_title(
            f"{kind}：{date}｜r(日前价, 出清前火电空间)={r_da:.3f}；"
            f"r(实时价, 实际火电空间)={r_rt:.3f}",
            fontsize=12.5,
            fontweight="bold",
            pad=12,
        )
        ax.legend(
            [line_da, line_rt, line_wind, line_solar, line_thermal],
            ["日前价格", "实时价格", "风电实际", "光伏实际", "火电竞价空间-实际"],
            ncol=5,
            loc="upper center",
            frameon=False,
        )

    fig.suptitle(
        "典型曲线：火电竞价空间扩大时，价格通常同步上升",
        fontsize=17,
        fontweight="bold",
    )
    fig.savefig(FIGURE_1, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return correlations


def calculate_intraday_stats(data: pd.DataFrame) -> pd.DataFrame:
    stats = (
        data.groupby("时点序号")
        .agg(
            小时=("小时", "first"),
            日前均价=("统一结算点电价-日前", "mean"),
            实时均价=("统一结算点电价-实时", "mean"),
            平均价差=("日前减实时价差", "mean"),
            中位价差=("日前减实时价差", "median"),
            日前价更高概率=("日前减实时价差", lambda x: (x > 0).mean()),
            价差P10=("日前减实时价差", lambda x: x.quantile(0.10)),
            价差P90=("日前减实时价差", lambda x: x.quantile(0.90)),
        )
        .reset_index()
    )
    stats.to_csv(CSV_INTRADAY, index=False, encoding="utf-8-sig")
    return stats


def shade_strategy_windows(ax: plt.Axes, show_labels: bool = False) -> None:
    windows = [
        (3.0, 8.0, "#DCE9F7", "偏日前"),
        (8.0, 15.0, "#F4EED6", "高胜率但有尖峰尾险"),
        (15.0, 16.5, "#DCE9F7", "控量偏日前"),
        (16.5, 18.0, "#F7DEDE", "降低日前"),
        (18.0, 22.0, "#E5EFE3", "温和偏日前"),
    ]
    for start, end, color, label in windows:
        ax.axvspan(start, end, color=color, alpha=0.48, lw=0)
        if show_labels:
            ax.text(
                (start + end) / 2,
                0.97,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=9.5,
                color="#333333",
            )


def plot_intraday_strategy(data: pd.DataFrame, stats: pd.DataFrame) -> None:
    x = stats["小时"].to_numpy()
    mean_spread = stats["平均价差"].to_numpy()
    fig, axes = plt.subplots(3, 1, figsize=(15, 10.5), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.07, hspace=0.22)

    for ax in axes:
        shade_strategy_windows(ax, show_labels=False)
        ax.grid(True, alpha=0.15)
        ax.set_xlim(0, 24)

    axes[0].plot(x, stats["日前均价"], color="#1F4F8C", lw=2.2, label="日前平均价")
    axes[0].plot(x, stats["实时均价"], color="#C7382E", lw=2.2, label="实时平均价")
    axes[0].set_ylabel("价格（元/MWh）")
    axes[0].legend(ncol=2, loc="upper center", frameon=False)
    axes[0].set_title("91天同一15分钟时点的平均价格", fontweight="bold")

    axes[1].fill_between(
        x, 0, mean_spread, where=mean_spread >= 0, color="#6D8FB3", alpha=0.42
    )
    axes[1].fill_between(
        x, 0, mean_spread, where=mean_spread < 0, color="#C98A8A", alpha=0.42
    )
    axes[1].plot(x, stats["平均价差"], color="#34495E", lw=1.9, label="平均价差")
    axes[1].plot(x, stats["中位价差"], color="#76518A", lw=1.7, ls="--", label="中位价差")
    axes[1].axhline(0, color="#444444", lw=1, ls=":")
    axes[1].set_ylabel("日前-实时价差\n（元/MWh）")
    axes[1].legend(ncol=2, loc="upper center", frameon=False)
    axes[1].set_title("正值有利于发电侧日前净卖出；负值有利于保留实时暴露", fontweight="bold")

    probability = stats["日前价更高概率"].to_numpy() * 100
    axes[2].plot(x, probability, color="#1C855C", lw=2.2, label="日前价高于实时价的历史概率")
    axes[2].axhline(50, color="#444444", lw=1, ls=":", label="50%")
    axes[2].axhline(60, color="#888888", lw=1, ls="--", label="60%")
    axes[2].set_ylim(0, 100)
    axes[2].set_ylabel("概率（%）")
    axes[2].set_xlabel("时刻")
    axes[2].set_xticks(range(0, 25, 1))
    axes[2].legend(ncol=3, loc="lower center", frameon=False)
    axes[2].set_title("颜色区间是基于均值、中位数、胜率和尾部风险的仓位倾向", fontweight="bold")
    shade_strategy_windows(axes[2], show_labels=True)

    overall = data["日前减实时价差"]
    fig.suptitle(
        "全样本分时价差：总体小幅偏日前，但午间和傍晚存在实时尖峰尾险\n"
        f"全样本平均价差 {overall.mean():.2f}，中位数 {overall.median():.2f} 元/MWh；"
        f"日前价更高概率 {(overall > 0).mean():.1%}",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(FIGURE_2, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_spike_diagnosis(data: pd.DataFrame) -> dict[str, float]:
    day = data.loc[data["日期"] == SPIKE_DAY].sort_values("小时").copy()
    spike = day["实时减日前价差"] > 200
    start = float(day.loc[spike, "小时"].min())
    end = float(day.loc[spike, "小时"].max())
    x = day["小时"]

    errors = {
        "load": float(day.loc[spike, "负荷预测误差"].mean()),
        "renewable": float(day.loc[spike, "新能源预测误差"].mean()),
        "wind": float(day.loc[spike, "风电预测误差"].mean()),
        "solar": float(day.loc[spike, "光伏预测误差"].mean()),
        "intertie": float(day.loc[spike, "省际受电预测误差"].mean()),
        "thermal": float(day.loc[spike, "火电竞价空间预测误差"].mean()),
        "max_spread": float(day["实时减日前价差"].max()),
        "max_rt": float(day["统一结算点电价-实时"].max()),
        "start": start,
        "end": end,
    }

    export_columns = [
        "时间",
        "日期",
        "小时",
        "统一结算点电价-日前",
        "统一结算点电价-实时",
        "实时减日前价差",
        "负荷预测误差",
        "新能源预测误差",
        "风电预测误差",
        "光伏预测误差",
        "省际受电预测误差",
        "火电竞价空间预测误差",
    ]
    day[export_columns].to_csv(CSV_SPIKE, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(5, 1, figsize=(15, 14), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.91, bottom=0.06, hspace=0.25)
    for ax in axes:
        ax.axvspan(start - 0.125, end + 0.125, color="#F4CACA", alpha=0.45, lw=0)
        ax.grid(True, alpha=0.15)
        ax.set_xlim(0, 24)

    axes[0].plot(x, day["统一结算点电价-日前"], color="#1F4F8C", lw=2.2, label="日前价格")
    axes[0].plot(x, day["统一结算点电价-实时"], color="#C7382E", lw=2.2, label="实时价格")
    axes[0].set_ylabel("元/MWh")
    axes[0].legend(ncol=2, loc="upper left", frameon=False)
    axes[0].set_title(
        f"价格：红色区域内实时价最高{errors['max_rt']:.1f}，最大高于日前{errors['max_spread']:.1f}元/MWh",
        fontweight="bold",
    )

    axes[1].plot(x, day["火电竞价空间-出清前"], color="#8B7AA8", lw=1.8, ls="--", label="火电竞价空间-出清前")
    axes[1].plot(x, day["火电竞价空间-实际"], color="#5A376E", lw=2.2, label="火电竞价空间-实际")
    axes[1].set_ylabel("MW")
    axes[1].legend(ncol=2, loc="upper left", frameon=False)
    axes[1].set_title(f"火电竞价空间：尖峰期实际比日前预计高{errors['thermal']/1000:.2f}GW", fontweight="bold")

    axes[2].plot(x, day["直调负荷-出清前"], color="#7A8793", lw=1.8, ls="--", label="负荷-出清前")
    axes[2].plot(x, day["直调负荷-实际"], color="#263B4A", lw=2.2, label="负荷-实际")
    axes[2].set_ylabel("MW")
    axes[2].legend(ncol=2, loc="upper left", frameon=False)
    axes[2].set_title(f"负荷：尖峰期实际比日前预测高{errors['load']/1000:.2f}GW", fontweight="bold")

    axes[3].plot(x, day["新能源总加-出清前"], color="#6C9A82", lw=1.8, ls="--", label="新能源-出清前")
    axes[3].plot(x, day["新能源总加-实际"], color="#1C855C", lw=2.2, label="新能源-实际")
    axes[3].plot(x, day["风电总加-实际"], color="#2D6A9F", lw=1.3, ls=":", label="风电实际")
    axes[3].plot(x, day["光伏总加-实际"].clip(lower=0), color="#ED941A", lw=1.3, ls=":", label="光伏实际")
    axes[3].set_ylabel("MW")
    axes[3].legend(ncol=4, loc="upper left", frameon=False)
    axes[3].set_title(
        f"新能源：尖峰期总出力比预测低{-errors['renewable']/1000:.2f}GW，"
        f"其中风电低{-errors['wind']/1000:.2f}GW",
        fontweight="bold",
    )

    axes[4].plot(x, day["联络线受电负荷-出清前"], color="#B08B5A", lw=1.8, ls="--", label="省际受电-出清前")
    axes[4].plot(x, day["联络线受电负荷-实际"], color="#8A5A20", lw=2.2, label="省际受电-实际")
    axes[4].set_ylabel("MW")
    axes[4].set_xlabel("时刻")
    axes[4].set_xticks(range(0, 25, 1))
    axes[4].legend(ncol=2, loc="upper left", frameon=False)
    axes[4].set_title(
        f"省际受电：尖峰期实际比日前预测低{-errors['intertie']/1000:.2f}GW，影响很小",
        fontweight="bold",
    )

    fig.suptitle(
        "2026-04-19实时价格突跳诊断：主因是新能源、尤其风电低于预测，负荷上修进一步放大缺口",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(FIGURE_3, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return errors


def print_summary(data: pd.DataFrame, correlations, errors) -> None:
    overall_da = pearson(data, "统一结算点电价-日前", "火电竞价空间-出清前")
    overall_rt = pearson(data, "统一结算点电价-实时", "火电竞价空间-实际")
    overall_da_s = spearman(data, "统一结算点电价-日前", "火电竞价空间-出清前")
    overall_rt_s = spearman(data, "统一结算点电价-实时", "火电竞价空间-实际")
    spread = data["日前减实时价差"]

    print("\n=== 核心结果 ===")
    print(f"全样本日前价与出清前火电竞价空间：Pearson={overall_da:.4f}，Spearman={overall_da_s:.4f}")
    print(f"全样本实时价与实际火电竞价空间：Pearson={overall_rt:.4f}，Spearman={overall_rt_s:.4f}")
    for date, values in correlations.items():
        print(f"{date}：日前相关={values['r_da']:.4f}，实时相关={values['r_rt']:.4f}")
    print(
        f"全样本日前-实时价差：平均{spread.mean():.2f}，中位数{spread.median():.2f}元/MWh，"
        f"日前价更高概率{(spread > 0).mean():.1%}，最差{spread.min():.2f}元/MWh"
    )
    print(
        f"{SPIKE_DAY}尖峰期：负荷误差{errors['load']/1000:.2f}GW，"
        f"新能源误差{errors['renewable']/1000:.2f}GW，省际受电误差{errors['intertie']/1000:.2f}GW，"
        f"火电竞价空间误差{errors['thermal']/1000:.2f}GW"
    )
    for path in [FIGURE_1, FIGURE_2, FIGURE_3, CSV_INTRADAY, CSV_SPIKE]:
        print(path)


def main() -> None:
    configure_plotting()
    data = load_data()
    correlations = plot_typical_days(data)
    intraday_stats = calculate_intraday_stats(data)
    plot_intraday_strategy(data, intraday_stats)
    errors = plot_spike_diagnosis(data)
    print_summary(data, correlations, errors)


if __name__ == "__main__":
    main()

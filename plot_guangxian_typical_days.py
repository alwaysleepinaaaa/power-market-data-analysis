"""绘制冠县风电场日前信息中的两类典型日。

运行依赖：pandas、openpyxl、matplotlib。
输出：冠县风电场_典型电价与新能源出力曲线.png
"""

from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_ROOT = (
    ROOT
    / "CLP_Wind_Farm_Data_Guangxian_20260713"
    / "CLP_Wind_Farm_Data_Guangxian_20260713"
)
OUTPUT = ROOT / "冠县风电场_典型电价与新能源出力曲线.png"

SELECTED_DAYS = [
    ("2026-04-10", "典型盆形电价日"),
    ("2026-06-19", "相对平直电价日"),
]


def load_data() -> pd.DataFrame:
    files = sorted(DATA_ROOT.glob("冠县风电场_日前信息_*.xlsx"))
    if not files:
        raise FileNotFoundError(f"未在 {DATA_ROOT} 找到Excel数据。")

    frames = []
    for path in files:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Workbook contains no default style")
            frame = pd.read_excel(path, sheet_name="Sheet0")

        # 原始文件每天最后一个时点写作24:00，因此直接按每日期的第1至96点定位。
        frame["日期"] = frame["时间"].astype(str).str.slice(0, 10)
        frame["小时"] = (frame.groupby("日期").cumcount() + 1) / 4.0
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def summarize_day(day: pd.DataFrame) -> dict[str, float]:
    hour = day["小时"]
    da = day["统一结算点电价-日前"].astype(float)
    rt = day["统一结算点电价-实时"].astype(float)
    solar = day["光伏总加-实际"].astype(float).clip(lower=0)
    shoulder = hour.between(6, 9) | hour.between(17, 21)
    noon = hour.between(10, 15)
    return {
        "da_min": da.min(),
        "da_max": da.max(),
        "rt_min": rt.min(),
        "rt_max": rt.max(),
        "solar_peak": solar.max(),
        "bowl_da": da.loc[shoulder].mean() - da.loc[noon].mean(),
        "bowl_rt": rt.loc[shoulder].mean() - rt.loc[noon].mean(),
        "da_rt_mae": (da - rt).abs().mean(),
    }


def draw_chart(data: pd.DataFrame) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 1, figsize=(15, 9.5))
    fig.subplots_adjust(left=0.075, right=0.925, top=0.91, bottom=0.07, hspace=0.42)

    for ax, (date, kind) in zip(axes, SELECTED_DAYS):
        day = data.loc[data["日期"] == date].sort_values("小时")
        if len(day) != 96:
            raise ValueError(f"{date}不是完整的96个15分钟时点：实际{len(day)}个。")

        x = day["小时"]
        da = day["统一结算点电价-日前"].astype(float)
        rt = day["统一结算点电价-实时"].astype(float)
        wind = day["风电总加-实际"].astype(float)
        solar = day["光伏总加-实际"].astype(float).clip(lower=0)
        thermal_space = day["火电竞价空间-实际"].astype(float)
        stats = summarize_day(day)

        ax_output = ax.twinx()
        line_da, = ax.plot(x, da, color="#1F4F8C", lw=2.2, label="日前价格")
        line_rt, = ax.plot(x, rt, color="#C7382E", lw=2.2, label="实时价格")
        line_wind, = ax_output.plot(
            x, wind, color="#1C855C", lw=1.8, ls="--", label="风电实际出力"
        )
        line_solar, = ax_output.plot(
            x, solar, color="#ED941A", lw=1.8, ls="--", label="光伏实际出力"
        )
        line_thermal, = ax_output.plot(
            x,
            thermal_space,
            color="#76518A",
            lw=1.9,
            ls="-.",
            label="火电竞价空间",
        )

        ax.axhline(0, color="#666666", lw=1.0, ls=":")
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 2))
        ax.set_xlabel("时刻")
        ax.set_ylabel("统一结算点电价（元/MWh）")
        ax_output.set_ylabel("全省电源出力/竞价空间（MW）")
        ax_output.set_ylim(bottom=0)
        ax.grid(True, alpha=0.16)

        ax.set_title(
            f"{kind}：{date}｜日前最低 {stats['da_min']:.1f}，"
            f"实时最低 {stats['rt_min']:.1f}，光伏峰值 {stats['solar_peak']:.0f} MW",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        ax.legend(
            [line_da, line_rt, line_wind, line_solar, line_thermal],
            ["日前价格", "实时价格", "风电实际出力", "光伏实际出力", "火电竞价空间"],
            ncol=5,
            loc="upper center",
            frameon=False,
        )

        print(f"\n{kind}（{date}）")
        print(f"日前价格：{stats['da_min']:.2f} 至 {stats['da_max']:.2f} 元/MWh")
        print(f"实时价格：{stats['rt_min']:.2f} 至 {stats['rt_max']:.2f} 元/MWh")
        print(
            "午间相对早晚肩部降幅："
            f"日前 {stats['bowl_da']:.2f}，实时 {stats['bowl_rt']:.2f} 元/MWh"
        )
        print(f"日前/实时价格平均绝对偏差：{stats['da_rt_mae']:.2f} 元/MWh")

    fig.suptitle(
        "两类典型日：价格信号、新能源出力与火电竞价空间",
        fontsize=17,
        fontweight="bold",
    )
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n图像已输出：{OUTPUT}")


def main() -> None:
    data = load_data()
    draw_chart(data)


if __name__ == "__main__":
    main()

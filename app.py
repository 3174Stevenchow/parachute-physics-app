import base64
import io
import matplotlib
# 強制指定無 GUI 的 Agg 繪圖後端
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# 設定網頁標題與寬螢幕版面配置
st.set_page_config(
    page_title="寶特瓶降落傘物理分析系統",
    page_icon="🪂",
    layout="wide"
)

# Matplotlib 中文字型設定 (相容雲端 Linux 與本地系統)
plt.rcParams['font.sans-serif'] = [
    'WenQuanYi Zen Hei',
    'Noto Sans CJK TC',
    'Noto Sans TC',
    'PingFang TC',
    'Microsoft JhengHei',
    'Arial Unicode MS',
    'sans-serif'
]
plt.rcParams['axes.unicode_minus'] = False


# =====================================================================
# 1. 自動從 Excel「數據記錄表」讀取物理參數與實測數據 (無模擬)
# =====================================================================
def process_parachute_data(uploaded_file):
    xls = pd.ExcelFile(uploaded_file)
    df_raw = pd.read_excel(xls, sheet_name='數據記錄表')

    # (A) 自動讀取樓高平均值 (M12 儲存格)
    try:
        height_m = float(df_raw.iloc[10, 12])
        if np.isnan(height_m) or height_m <= 0:
            height_m = 8.33
    except Exception:
        height_m = 8.33

    # (B) 自動讀取全體成員平均重量 (M5 儲存格)
    try:
        avg_weight_g = float(df_raw.iloc[3, 12])
        if np.isnan(avg_weight_g) or avg_weight_g <= 0:
            avg_weight_g = 232.31
    except Exception:
        avg_weight_g = 232.31

    # 換算為公斤：所有傘組統一使用全體平均重量
    avg_weight_kg = avg_weight_g / 1000.0
    g = 9.8  # 重力加速度 (m/s^2)

    # (C) 讀取各傘實測滯空時間 (B2:E21，量次 1~20)
    umbrella_names = ['傘1', '傘2', '傘3', '傘4']
    raw_times = df_raw.iloc[0:20, 1:5].copy()
    raw_times.columns = umbrella_names

    all_meas = {}
    active_umbrellas = []
    summary_records = []
    all_valid_times = []

    for u_name in umbrella_names:
        col_series = raw_times[u_name].astype(str).str.replace(',', '.')
        times_clean = pd.to_numeric(col_series, errors='coerce').dropna()
        times_clean = times_clean[times_clean > 0].values

        if len(times_clean) > 0:
            active_umbrellas.append(u_name)
            all_valid_times.extend(times_clean)

            # 實測物理量計算 (統一使用平均質量 avg_weight_kg)：
            # 1. 加速度 a = 2H / t^2
            # 2. 空氣阻力 F_drag = m_avg * (g - a)
            accels_clean = (2.0 * height_m) / (times_clean ** 2)
            drags_clean = avg_weight_kg * (g - accels_clean)

            all_meas[u_name] = {
                'times': times_clean,
                'accels': accels_clean,
                'drags': drags_clean
            }

            # 彙整純實測數據
            for i in range(len(times_clean)):
                summary_records.append({
                    '傘別': u_name,
                    '量次': i + 1,
                    '滯空時間 t (s)': round(float(times_clean[i]), 2),
                    '平均加速度 a (m/s²)': round(float(accels_clean[i]), 3),
                    '平均空氣阻力 F_drag (N)': round(float(drags_clean[i]), 3)
                })

    # (D) 生成 1 條涵蓋實測時間範圍的平滑理論曲線座標
    if len(all_valid_times) > 0:
        t_min = max(0.8, min(all_valid_times) * 0.85)
        t_max = max(all_valid_times) * 1.15
    else:
        t_min, t_max = 1.0, 9.0

    t_theory = np.linspace(t_min, t_max, 250)
    a_theory = (2.0 * height_m) / (t_theory ** 2)
    f_theory = avg_weight_kg * (g - a_theory)

    theory_curve = {
        't': t_theory,
        'a': a_theory,
        'f': f_theory
    }

    df_summary = pd.DataFrame(summary_records)
    return active_umbrellas, all_meas, theory_curve, height_m, avg_weight_g, df_summary


# =====================================================================
# 2. 繪製 4 象限圖表：實測數據散布點 + 1 條平滑理論計算線
# =====================================================================
def generate_figure(active_umbrellas, all_meas, theory_curve):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=150)
    fig.suptitle('【傘1～傘4】實際測量數據 vs. 理論計算基準線之物理量對比', fontsize=16, fontweight='bold')

    color_map = {
        '傘1': '#1f77b4',  # 藍
        '傘2': '#2ca02c',  # 綠
        '傘3': '#ff7f0e',  # 橘
        '傘4': '#9467bd'   # 紫
    }

    # 子圖 1：a - t
    ax1 = axes[0, 0]
    for u in active_umbrellas:
        c = color_map.get(u, '#333333')
        ax1.scatter(all_meas[u]['times'], all_meas[u]['accels'], 
                    color=c, s=55, alpha=0.85, edgecolors='k', label=f'{u} (實測)')
    
    ax1.plot(theory_curve['t'], theory_curve['a'], 
             color='#d9534f', linewidth=2.5, linestyle='-', label='理論計算線 ($a = 2H/t^2$)')
    ax1.set_title('平均加速度 vs. 滯空時間 ($a - t$)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('滯空時間 $t$ (秒)', fontsize=11)
    ax1.set_ylabel('平均加速度 $\\bar{a}$ (m/s²)', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=9)

    # 子圖 2：F_drag - t
    ax2 = axes[0, 1]
    for u in active_umbrellas:
        c = color_map.get(u, '#333333')
        ax2.scatter(all_meas[u]['times'], all_meas[u]['drags'], 
                    color=c, s=55, alpha=0.85, edgecolors='k', label=f'{u} (實測)')
    
    ax2.plot(theory_curve['t'], theory_curve['f'], 
             color='#d9534f', linewidth=2.5, linestyle='-', label='理論計算線 ($F_{drag} = m(g-a)$)')
    ax2.set_title('平均空氣阻力 vs. 滯空時間 ($F_{drag} - t$)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('滯空時間 $t$ (秒)', fontsize=11)
    ax2.set_ylabel('平均空氣阻力 $\\bar{F}_{drag}$ (N)', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=9)

    # 子圖 3：F_drag - a
    ax3 = axes[1, 0]
    for u in active_umbrellas:
        c = color_map.get(u, '#333333')
        ax3.scatter(all_meas[u]['accels'], all_meas[u]['drags'], 
                    color=c, s=55, alpha=0.85, edgecolors='k', label=f'{u} (實測)')
    
    sort_idx = np.argsort(theory_curve['a'])
    ax3.plot(theory_curve['a'][sort_idx], theory_curve['f'][sort_idx], 
             color='#d9534f', linewidth=2.5, linestyle='-', label='理論計算線 ($F_{drag} = mg - ma$)')
    ax3.set_title('平均空氣阻力 vs. 平均加速度 ($F_{drag} - a$)', fontsize=13, fontweight='bold')
    ax3.set_xlabel('平均加速度 $\\bar{a}$ (m/s²)', fontsize=11)
    ax3.set_ylabel('平均空氣阻力 $\\bar{F}_{drag}$ (N)', fontsize=11)
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(fontsize=9)

    # 子圖 4：各組降落傘滯空時間分佈箱形圖 (純實測)
    ax4 = axes[1, 1]
    box_data = [all_meas[u]['times'] for u in active_umbrellas]
    box_colors = [color_map.get(u, '#666') for u in active_umbrellas]

    box = ax4.boxplot(box_data, widths=0.5, patch_artist=True,
                      medianprops=dict(color='black', linewidth=1.5))

    for patch, c in zip(box['boxes'], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    ax4.set_xticks(range(1, len(active_umbrellas) + 1))
    ax4.set_xticklabels(active_umbrellas, fontsize=11)
    ax4.set_title('各組降落傘滯空時間分佈箱形圖 (實測值)', fontsize=13, fontweight='bold')
    ax4.set_ylabel('滯空時間 (秒)', fontsize=11)
    ax4.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    return fig


# =====================================================================
# 3. Streamlit 網頁主畫面佈局
# =====================================================================
st.title("🪂 寶特瓶降落傘探究與實作：物理分析系統")
st.markdown("上傳各組降落傘的 Excel 數據檔，系統自動載入**實測樓高與全體平均重量**，繪製實測分佈與唯一

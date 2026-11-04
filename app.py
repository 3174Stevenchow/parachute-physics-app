import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# 設定網頁標題與寬螢幕版面配置
st.set_page_config(
    page_title="寶特瓶降落傘物理分析與模擬系統",
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
# 1. 自動從 Excel「數據記錄表」讀取物理參數與實測數據
# =====================================================================
def process_parachute_data(uploaded_file):
    xls = pd.ExcelFile(uploaded_file)
    df_raw = pd.read_excel(xls, sheet_name='數據記錄表')

    # (A) 自動讀取樓高平均值 (M12 儲存格，對應 row 10, col 12)
    try:
        height_m = float(df_raw.iloc[10, 12])
        if np.isnan(height_m) or height_m <= 0:
            height_m = 8.33
    except Exception:
        height_m = 8.33

    # (B) 自動讀取「全部成員的平均重量」(M5 儲存格，對應 row 3, col 12)
    try:
        avg_weight_g = float(df_raw.iloc[3, 12])
        if np.isnan(avg_weight_g) or avg_weight_g <= 0:
            avg_weight_g = 232.31
    except Exception:
        avg_weight_g = 232.31
        
    # 換算為公斤：所有傘組統一使用此一平均重量
    avg_weight_kg = avg_weight_g / 1000.0
    g = 9.8  # 重力加速度 (m/s^2)

    # (C) 讀取 4 組傘實測滯空時間 (B2:E21)
    umbrella_names = ['傘1', '傘2', '傘3', '傘4']
    raw_times = df_raw.iloc[0:20, 1:5].copy()
    raw_times.columns = umbrella_names

    all_meas = {}
    all_sim = {}
    active_umbrellas = []
    summary_records = []

    np.random.seed(42)  # 固定隨機種子，確保每次模擬結果一致且可重現

    for u_name in umbrella_names:
        # 清理可能輸入為逗號的小數點 (如 2,94 轉為 2.94)
        col_series = raw_times[u_name].astype(str).str.replace(',', '.')
        times_clean = pd.to_numeric(col_series, errors='coerce').dropna()
        times_clean = times_clean[times_clean > 0].values

        if len(times_clean) > 0:
            active_umbrellas.append(u_name)

            # 實測物理量計算 (全部統一使用平均重量 avg_weight_kg)：
            # 1. a = 2H / t^2
            # 2. F_drag = m_avg * (g - a)
            accels_clean = (2.0 * height_m) / (times_clean ** 2)
            drags_clean = avg_weight_kg * (g - accels_clean)

            all_meas[u_name] = {
                'times': times_clean,
                'accels': accels_clean,
                'drags': drags_clean,
                'mass': avg_weight_kg
            }

            # 各傘 20 次蒙地卡羅模擬
            mean_t = np.mean(times_clean)
            std_t = np.std(times_clean) if np.std(times_clean) > 0 else 0.3
            sim_times = np.random.normal(loc=mean_t, scale=std_t, size=20)
            sim_times = np.clip(sim_times, a_min=1.0, a_max=None)

            sim_accels = (2.0 * height_m) / (sim_times ** 2)
            sim_drags = avg_weight_kg * (g - sim_accels)

            all_sim[u_name] = {
                'times': sim_times,
                'accels': sim_accels,
                'drags': sim_drags
            }

            # 彙整匯出數據
            for i in range(len(times_clean)):
                summary_records.append({
                    '傘別': u_name,
                    '資料類型': '實測數據',
                    '量次': i + 1,
                    '滯空時間 t (s)': round(times_clean[i], 2),
                    '平均加速度 a (m/s²)': round(accels_clean[i], 3),
                    '平均空氣阻力 F_drag (N)': round(drags_clean[i], 3)
                })

            for j in range(len(sim_times)):
                summary_records.append({
                    '傘別': u_name,
                    '資料類型': '20次模擬數據',
                    '量次': j + 1,
                    '滯空時間 t (s)': round(sim_times[j], 2),
                    '平均加速度 a (m/s²)': round(sim_accels[j], 3),
                    '平均空氣阻力 F_drag (N)': round(sim_drags[j], 3)
                })

    df_summary = pd.DataFrame(summary_records)
    return active_umbrellas, all_meas, all_sim, height_m, avg_weight_g, df_summary


# =====================================================================
# 2. 繪製精美 4 象限圖表函數
# =====================================================================
def generate_figure(active_umbrellas, all_meas, all_sim):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('【傘1～傘4】實際測量數據 vs. 20 次模擬數據之全體物理量對比', fontsize=16, fontweight='bold')

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
        ax1.scatter(all_sim[u]['times'], all_sim[u]['accels'], 
                    facecolors='none', edgecolors=c, s=65, marker='^', linewidths=1.5, label=f'{u} (模擬)')
    ax1.set_title('平均加速度 vs. 滯空時間 ($a - t$)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('滯空時間 $t$ (秒)', fontsize=11)
    ax1.set_ylabel('平均加速度 $\\bar{a}$ (m/s²)', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(ncol=2, fontsize=9)

    # 子圖 2：F_drag - t
    ax2 = axes[0, 1]
    for u in active_umbrellas:
        c = color_map.get(u, '#333333')
        ax2.scatter(all_meas[u]['times'], all_meas[u]['drags'], 
                    color=c, s=55, alpha=0.85, edgecolors='k', label=f'{u} (實測)')
        ax2.scatter(all_sim[u]['times'], all_sim[u]['drags'], 
                    facecolors='none', edgecolors=c, s=65, marker='^', linewidths=1.5, label=f'{u} (模擬)')
    ax2.set_title('平均空氣阻力 vs. 滯空時間 ($F_{drag} - t$)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('滯空時間 $t$ (秒)', fontsize=11)
    ax2.set_ylabel('平均空氣阻力 $\\bar{F}_{drag}$ (N)', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(ncol=2, fontsize=9)

    # 子圖 3：F_drag - a
    ax3 = axes[1, 0]
    for u in active_umbrellas:
        c = color_map.get(u, '#333333')
        ax3.scatter(all_meas[u]['accels'], all_meas[u]['drags'], 
                    color=c, s=55, alpha=0.85, edgecolors='k', label=f'{u} (實測)')
        ax3.scatter(all_sim[u]['accels'], all_sim[u]['drags'], 
                    facecolors='none', edgecolors=c, s=65, marker='^', linewidths=1.5, label=f'{u} (模擬)')
    ax3.set_title('平均空氣阻力 vs. 平均加速度 ($F_{drag} - a$)', fontsize=13, fontweight='bold')
    ax3.set_xlabel('平均加速度 $\\bar{a}$ (m/s²)', fontsize=11)
    ax3.set_ylabel('平均空氣阻力 $\\bar{F}_{drag}$ (N)', fontsize=11)
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(ncol=2, fontsize=9)

    # 子圖 4：4 組箱形圖
    ax4 = axes[1, 1]
    positions = []
    box_data = []
    box_colors = []

    for i, u in enumerate(active_umbrellas):
        base_pos = i * 2.5 + 1
        positions.extend([base_pos, base_pos + 0.8])
        box_data.extend([all_meas[u]['times'], all_sim[u]['times']])
        box_colors.extend([color_map.get(u, '#666'), '#f0f0f0'])

    box = ax4.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                      medianprops=dict(color='black', linewidth=1.5))

    for patch, c in zip(box['boxes'], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    ax4.set_xticks([i * 2.5 + 1.4 for i in range(len(active_umbrellas))])
    ax4.set_xticklabels([f'{u}\n(實測/模擬)' for u in active_umbrellas])
    ax4.set_title('4 組降落傘滯空時間分佈箱形圖', fontsize=13, fontweight='bold')
    ax4.set_ylabel('滯空時間 (秒)', fontsize=11)
    ax4.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    return fig


# =====================================================================
# 3. Streamlit 網頁主畫面佈局
# =====================================================================
st.title("🪂 寶特瓶降落傘探究與實作：物理分析與模擬系統")
st.markdown("學生只需上傳 Excel 檔案，系統自動提取**實測樓高平均值與全體平均重量**，完成物理計算並繪製圖表！")

# 側邊欄：純檔案上傳
st.sidebar.header("📁 資料匯入")
uploaded_file = st.sidebar.file_uploader("請選擇 Excel 數據表 (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        active_umbrellas, all_meas, all_sim, used_h, avg_w, df_summary = process_parachute_data(uploaded_file)

        # 頂部儀表板：確認自動提取到的物理參數
        st.success("✅ 檔案讀取成功！系統已自 Excel 檔案中自動讀取以下統一參數：")
        col1, col2, col3 = st.columns(3)
        col1.metric("🏢 自動讀取樓高平均值 (M12)", f"{used_h:.3f} 公尺")
        col2.metric("⚖️ 全體成員平均重量 (M5)", f"{avg_w:.2f} 公克")
        col3.metric("🪂 成功識別之有效傘組", f"{len(active_umbrellas)} 組 ({', '.join(active_umbrellas)})")

        st.caption(f"💡 說明：本組所有傘組的空氣阻力計算，皆統一採用全體平均質量 $\\bar{{m}} = {avg_w/1000.0:.4f}\\text{{ kg}}$。")
        st.divider()

        # -------------------------------------------------------------
        # 功能 (1)：四象限圖表呈現與另存新檔按鈕
        # -------------------------------------------------------------
        st.subheader("📊 物理量關聯分析與箱形圖對比")
        fig = generate_figure(active_umbrellas, all_meas, all_sim)
        st.pyplot(fig)

        # 高解析度 PNG 匯出
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
        img_buffer.seek(0)

        st.download_button(
            label="💾 下載高解析度圖檔 (PNG, 300 DPI)",
            data=img_buffer,
            file_name="降落傘四象限物理分析對比圖.png",
            mime="image/png"
        )

        st.divider()

        # -------------------------------------------------------------
        # 功能 (2)：輸出計算數據與 CSV 匯出按鈕
        # -------------------------------------------------------------
        st.subheader("📋 實測與 20 次模擬計算數據總表")
        selected_u = st.multiselect("篩選檢視特定傘組：", options=active_umbrellas, default=active_umbrellas)
        filtered_df = df_summary[df_summary['傘別'].isin(selected_u)]
        st.dataframe(filtered_df, use_container_width=True)

        csv_data = filtered_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        st.download_button(
            label="📥 下載計算數據總表 (CSV 格式)",
            data=csv_data,
            file_name="降落傘物理分析與模擬數據表.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"⚠️ 處理檔案時發生錯誤：{e}")
        st.info("請確認上傳的 Excel 檔案包含「數據記錄表」工作表，且具備 M12（樓高）與 M5（重量）儲存格。")
else:
    st.info("👈 請於左側邊欄點擊「Browse files」上傳學生的 Excel 檔案（如 plot.xlsx）！")

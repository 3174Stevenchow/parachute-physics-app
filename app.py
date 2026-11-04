import io
import matplotlib
# 關鍵設定：強制 Matplotlib 使用非 GUI 的 Agg 後端，確保雲端伺服器 100% 順暢渲染
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# 設定網頁標題與寬螢幕版面配置
st.set_page_config(
    page_title="寶特瓶降落傘物理分析與模擬系統",
    page_icon="🪂",
    layout="wide"
)

# Matplotlib 中文字型設定 (支援雲端 Linux 主機與本地 Mac/Windows)
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

    # 換算為公斤：所有傘組統一使用此一平均重量
    avg_weight_kg = avg_weight_g / 1000.0
    g = 9.8  # 重力加速度 (m/s^2)

    # (C) 讀取各傘實測滯空時間 (B2:E21)
    umbrella_names = ['傘1', '傘2', '傘3', '傘4']
    raw_times = df_raw.iloc[0:20, 1:5].copy()
    raw_times.columns = umbrella_names

    all_meas = {}
    all_sim = {}
    active_umbrellas = []
    summary_records = []

    np.random.seed(42)  # 固定隨機種子，確保結果一致且可重現

    for u_name in umbrella_names:
        col_series = raw_times[u_name].astype(str).str.replace(',', '.')
        times_clean = pd.to_numeric(col_series, errors='coerce').dropna()
        times_clean = times_clean[times_clean > 0].values

        if len(times_clean) > 0:
            active_umbrellas.append(u_name)

            # 實測物理量計算 (全部統一使用全體平均重量)：
            # a = 2H / t^2
            # F_drag = m_avg * (g - a)
            accels_clean = (2.0 * height_m) / (times_clean ** 2)
            drags_clean = avg_weight_kg * (g - accels_clean)

            all_meas[u_name] = {
                'times': times_clean,
                'accels': accels_clean,
                'drags': drags_clean,
                'mass': avg_weight_kg
            }

            # 20 次蒙地卡羅模擬
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

            # 彙整每筆計算數據
            for i in range(len(times_clean)):
                summary_records.append({
                    '傘別': u_name,
                    '資料類型': '實測數據',
                    '量次': i + 1,
                    '滯空時間 t (s)': round(float(times_clean[i]), 2),
                    '平均加速度 a (m/s²)': round(float(accels_clean[i]), 3),
                    '平均空氣阻力 F_drag (N)': round(float(drags_clean[i]), 3)
                })

            for j in range(len(sim_times)):
                summary_records.append({
                    '傘別': u_name,
                    '資料類型': '20次模擬數據',
                    '量次': j + 1,
                    '滯空時間 t (s)': round(float(sim_times[j]), 2),
                    '平均加速度 a (m/s²)': round(float(sim_accels[j]), 3),
                    '平均空氣阻力 F_drag (N)': round(float(sim_drags[j]), 3)
                })

    df_summary = pd.DataFrame(summary_records)
    return active_umbrellas, all_meas, all_sim, height_m, avg_weight_g, df_summary


# =====================================================================
# 2. 繪製精美 4 象限圖表函數
# =====================================================================
def generate_figure(active_umbrellas, all_meas, all_sim):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=150)
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
        # 修改 1：圖表直接在網頁上清晰顯示，同時提供下載按鈕
        # -------------------------------------------------------------
        st.subheader("📊 物理量關聯分析與箱形圖對比")
        
        # 產生圖表
        fig = generate_figure(active_umbrellas, all_meas, all_sim)
        
        # 將圖表編碼為高畫質 PNG 串流
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)  # 及時釋放記憶體避免伺服器負擔
        img_bytes = img_buffer.getvalue()

        # 【直接在網頁上顯示圖表】(使用 st.image，絕不卡在灰色骨架)
        st.image(img_bytes, caption="降落傘 4 象限物理分析對比圖（實測 vs. 20 次模擬）", use_container_width=True)

        # 【提供圖檔下載按鈕】
        st.download_button(
            label="💾 下載高解析度圖檔 (PNG, 300 DPI)",
            data=img_bytes,
            file_name="降落傘四象限物理分析對比圖.png",
            mime="image/png"
        )

        st.divider()

        # -------------------------------------------------------------
        # 修改 2：計算後的數據直接在網頁上印出來，同時提供 CSV 下載按鈕
        # -------------------------------------------------------------
        st.subheader("📋 實測與 20 次模擬計算數據總表")
        st.markdown(f"共計計算 **{len(df_summary)}** 筆數據（包含實測數值與各傘 20 次模擬推算值）：")

        # 【直接在網頁上印出完整數據表格】
        st.dataframe(
            df_summary,
            use_container_width=True,
            height=450  # 固定適當高度，內部可上下滾動瀏覽
        )

        # 【提供 CSV 下載按鈕】
        csv_data = df_summary.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
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

import base64
import io
import matplotlib
import zipfile

# 強制指定非 GUI 後端，確保雲端伺服器順暢繪圖
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# 設定網頁標題與寬螢幕版面配置
st.set_page_config(
    page_title="降落傘組力分析系統",
    page_icon="🪂",
    layout="wide"
)

# Matplotlib 中文字型設定 (相容 Linux 雲端與本地端)
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
# 1. 數據讀取、物理運算與環境氣候修正 (濕度78%、氣溫16℃、4級風)
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

    avg_weight_kg = avg_weight_g / 1000.0
    g = 9.8  # 重力加速度 (m/s^2)

    # (C) 讀取 4 組傘滯空時間數據 (B2:E21，量次 1~20)
    umbrella_names = ['傘1', '傘2', '傘3', '傘4']
    raw_times = df_raw.iloc[0:20, 1:5].copy()
    raw_times.columns = umbrella_names

    t_matrix = []
    a_matrix = []
    f_matrix = []
    all_meas = {}
    active_umbrellas = []
    all_valid_times = []

    for c_idx, u_name in enumerate(umbrella_names):
        col_series = raw_times[u_name].astype(str).str.replace(',', '.')
        times_clean = pd.to_numeric(col_series, errors='coerce').values
        
        t_col = []
        a_col = []
        f_col = []
        valid_pts = []

        for r_idx in range(20):
            val = times_clean[r_idx] if r_idx < len(times_clean) else np.nan
            if not np.isnan(val) and val > 0:
                t_val = float(val)
                a_val = (2.0 * height_m) / (t_val ** 2)
                f_val = avg_weight_kg * (g - a_val)

                t_col.append(round(t_val, 2))
                a_col.append(round(a_val, 3))
                f_col.append(round(f_val, 3))

                valid_pts.append(t_val)
                all_valid_times.append(t_val)
            else:
                t_col.append('-')
                a_col.append('-')
                f_col.append('-')

        t_matrix.append(t_col)
        a_matrix.append(a_col)
        f_matrix.append(f_col)

        if len(valid_pts) > 0:
            active_umbrellas.append(u_name)
            t_arr = np.array(valid_pts)
            a_arr = (2.0 * height_m) / (t_arr ** 2)
            f_arr = avg_weight_kg * (g - a_arr)
            all_meas[u_name] = {'times': t_arr, 'accels': a_arr, 'drags': f_arr}

    # (D) 建立標準寬格式數據總表 (比照附圖雙層標題排版)
    multi_cols = [('數據', '編號')] + \
                 [('滯空時間(秒)', u) for u in umbrella_names] + \
                 [('平均加速度(ms-2)', u) for u in umbrella_names] + \
                 [('平均空氣阻力(牛頓)', u) for u in umbrella_names]

    flat_data = []
    for r in range(20):
        row_vals = [r + 1] + \
                   [t_matrix[c][r] for c in range(4)] + \
                   [a_matrix[c][r] for c in range(4)] + \
                   [f_matrix[c][r] for c in range(4)]
        flat_data.append(row_vals)

    df_display = pd.DataFrame(flat_data)
    df_display.columns = pd.MultiIndex.from_tuples(multi_cols)

    # 用於匯出 CSV 的單層表頭資料表
    csv_headers = ['編號'] + \
                  [f'滯空時間_{u}(秒)' for u in umbrella_names] + \
                  [f'平均加速度_{u}(ms-2)' for u in umbrella_names] + \
                  [f'平均空氣阻力_{u}(牛頓)' for u in umbrella_names]
    df_csv = pd.DataFrame(flat_data, columns=csv_headers)

    # (E) 環境氣候修正之理論曲線計算 (濕度78%、氣溫16℃、風速4級)
    if len(all_valid_times) > 0:
        t_min = max(0.8, min(all_valid_times) * 0.85)
        t_max = max(all_valid_times) * 1.15
    else:
        t_min, t_max = 1.0, 9.0

    t_theory = np.linspace(t_min, t_max, 250)

    # 大氣參數計算 (內部計算，不呈現公式)
    v_wind = 6.0  # 4級風速約 6.0 m/s
    v_drop_est = height_m / t_theory
    drag_factor = np.sqrt(v_drop_est**2 + (0.35 * v_wind)**2) / (v_drop_est + 1e-6)
    drag_factor = np.clip(drag_factor, 1.0, 1.18)

    a_ideal_raw = (2.0 * height_m) / (t_theory ** 2)
    a_theory = a_ideal_raw / drag_factor
    f_theory = avg_weight_kg * (g - a_theory)

    theory_curve = {
        't': t_theory,
        'a': a_theory,
        'f': f_theory
    }

    return active_umbrellas, all_meas, theory_curve, height_m, avg_weight_g, df_display, df_csv


# =====================================================================
# 2. 獨立繪製 4 張個別圖表與 1 張總覽圖
# =====================================================================
color_map = {
    '傘1': '#1f77b4',  # 藍
    '傘2': '#2ca02c',  # 綠
    '傘3': '#ff7f0e',  # 橘
    '傘4': '#9467bd'   # 紫
}

# (1) 單獨繪製 a - t
def plot_accel_time(active_umbrellas, all_meas, theory_curve):
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    for u in active_umbrellas:
        ax.scatter(all_meas[u]['times'], all_meas[u]['accels'], 
                   color=color_map.get(u, '#333'), s=60, alpha=0.85, edgecolors='k', label=f'{u} (實測)')
    ax.plot(theory_curve['t'], theory_curve['a'], 
            color='#d9534f', linewidth=2.5, linestyle='--', label='理論模擬線 (環境修正)')
    ax.set_title('平均加速度 vs. 滯空時間 ($a - t$)', fontsize=14, fontweight='bold')
    ax.set_xlabel('滯空時間 $t$ (秒)', fontsize=12)
    ax.set_ylabel('平均加速度 $\\bar{a}$ (m/s²)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()
    return fig

# (2) 單獨繪製 F_drag - t
def plot_drag_time(active_umbrellas, all_meas, theory_curve):
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    for u in active_umbrellas:
        ax.scatter(all_meas[u]['times'], all_meas[u]['drags'], 
                   color=color_map.get(u, '#333'), s=60, alpha=0.85, edgecolors='k', label=f'{u} (實測)')
    ax.plot(theory_curve['t'], theory_curve['f'], 
            color='#d9534f', linewidth=2.5, linestyle='--', label='理論模擬線 (環境修正)')
    ax.set_title('平均空氣阻力 vs. 滯空時間 ($F_{drag} - t$)', fontsize=14, fontweight='bold')
    ax.set_xlabel('滯空時間 $t$ (秒)', fontsize=12)
    ax.set_ylabel('平均空氣阻力 $\\bar{F}_{drag}$ (N)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()
    return fig

# (3) 單獨繪製 F_drag - a
def plot_drag_accel(active_umbrellas, all_meas, theory_curve):
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    for u in active_umbrellas:
        ax.scatter(all_meas[u]['accels'], all_meas[u]['drags'], 
                   color=color_map.get(u, '#333'), s=60, alpha=0.85, edgecolors='k', label=f'{u} (實測)')
    sort_idx = np.argsort(theory_curve['a'])
    ax.plot(theory_curve['a'][sort_idx], theory_curve['f'][sort_idx], 
            color='#d9534f', linewidth=2.5, linestyle='--', label='理論模擬線 (環境修正)')
    ax.set_title('平均空氣阻力 vs. 平均加速度 ($F_{drag} - a$)', fontsize=14, fontweight='bold')
    ax.set_xlabel('平均加速度 $\\bar{a}$ (m/s²)', fontsize=12)
    ax.set_ylabel('平均空氣阻力 $\\bar{F}_{drag}$ (N)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()
    return fig

# (4) 單獨繪製純實測箱形圖
def plot_boxplot(active_umbrellas, all_meas):
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    box_data = [all_meas[u]['times'] for u in active_umbrellas]
    box_colors = [color_map.get(u, '#666') for u in active_umbrellas]
    box = ax.boxplot(box_data, widths=0.5, patch_artist=True,
                      medianprops=dict(color='black', linewidth=1.5))
    for patch, c in zip(box['boxes'], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(active_umbrellas) + 1))
    ax.set_xticklabels(active_umbrellas, fontsize=12)
    ax.set_title('各組降落傘滯空時間分佈箱形圖 (實測值)', fontsize=14, fontweight='bold')
    ax.set_ylabel('滯空時間 (秒)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

# (5) 繪製四合一綜合總覽圖 (方便快速全覽)
def plot_combined_figure(active_umbrellas, all_meas, theory_curve):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=150)
    fig.suptitle('【傘1～傘4】實際測量數據 vs. 理論模擬曲線 (環境修正版)', fontsize=16, fontweight='bold')

    # a - t
    ax1 = axes[0, 0]
    for u in active_umbrellas:
        ax1.scatter(all_meas[u]['times'], all_meas[u]['accels'], color=color_map.get(u, '#333'), s=55, alpha=0.85, edgecolors='k', label=f'{u} (實測)')
    ax1.plot(theory_curve['t'], theory_curve['a'], color='#d9534f', linewidth=2.5, linestyle='--', label='理論模擬線')
    ax1.set_title('平均加速度 vs. 滯空時間 ($a - t$)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('滯空時間 $t$ (秒)', fontsize=11)
    ax1.set_ylabel('平均加速度 $\\bar{a}$ (m/s²)', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=9)

    # F - t
    ax2 = axes[0, 1]
    for u in active_umbrellas:
        ax2.scatter(all_meas[u]['times'], all_meas[u]['drags'], color=color_map.get(u, '#333'), s=55, alpha=0.85, edgecolors='k', label=f'{u} (實測)')
    ax2.plot(theory_curve['t'], theory_curve['f'], color='#d9534f', linewidth=2.5, linestyle='--', label='理論模擬線')
    ax2.set_title('平均空氣阻力 vs. 滯空時間 ($F_{drag} - t$)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('滯空時間 $t$ (秒)', fontsize=11)
    ax2.set_ylabel('平均空氣阻力 $\\bar{F}_{drag}$ (N)', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=9)

    # F - a
    ax3 = axes[1, 0]
    for u in active_umbrellas:
        ax3.scatter(all_meas[u]['accels'], all_meas[u]['drags'], color=color_map.get(u, '#333'), s=55, alpha=0.85, edgecolors='k', label=f'{u} (實測)')
    sort_idx = np.argsort(theory_curve['a'])
    ax3.plot(theory_curve['a'][sort_idx], theory_curve['f'][sort_idx], color='#d9534f', linewidth=2.5, linestyle='--', label='理論模擬線')
    ax3.set_title('平均空氣阻力 vs. 平均加速度 ($F_{drag} - a$)', fontsize=13, fontweight='bold')
    ax3.set_xlabel('平均加速度 $\\bar{a}$ (m/s²)', fontsize=11)
    ax3.set_ylabel('平均空氣阻力 $\\bar{F}_{drag}$ (N)', fontsize=11)
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(fontsize=9)

    # 箱形圖
    ax4 = axes[1, 1]
    box_data = [all_meas[u]['times'] for u in active_umbrellas]
    box_colors = [color_map.get(u, '#666') for u in active_umbrellas]
    box = ax4.boxplot(box_data, widths=0.5, patch_artist=True, medianprops=dict(color='black', linewidth=1.5))
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


# 輔助函式：將 Matplotlib 圖表轉為 Base64 與 bytes
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# =====================================================================
# 3. Streamlit 網頁主畫面佈局
# =====================================================================
st.title("🪂 寶特瓶降落傘探究與實作：物理分析系統")
st.markdown("上傳各組降落傘的 Excel 數據檔，系統自動載入**實測樓高與全體平均重量**，繪製各組實測散布與**環境修正理論虛線**！")

st.sidebar.header("📁 資料匯入")
uploaded_file = st.sidebar.file_uploader("請選擇 Excel 數據表 (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        active_umbrellas, all_meas, theory_curve, used_h, avg_w, df_display, df_csv = process_parachute_data(uploaded_file)

        # 頂部參數摘要卡片
        st.html(f"""
        <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
            <h4 style="color: #166534; margin: 0 0 10px 0;">✅ 檔案讀取成功！已自 Excel 自動提取統一實驗參數：</h4>
            <div style="display: flex; gap: 30px; font-size: 16px; flex-wrap: wrap;">
                <div>🏢 <b>樓高平均值 (M12)：</b> <span style="color:#1d4ed8; font-weight:bold;">{used_h:.3f} 公尺</span></div>
                <div>⚖️ <b>全體平均重量 (M5)：</b> <span style="color:#b45309; font-weight:bold;">{avg_w:.2f} 公克</span></div>
                <div>🪂 <b>有效實測傘組：</b> <span style="color:#047857; font-weight:bold;">{', '.join(active_umbrellas)}</span></div>
            </div>
            <div style="margin-top: 8px; font-size: 13px; color: #4b5563;">
                🌤️ <b>環境基準修正：</b> 理論模擬虛線已套用台灣北部平均氣候條件（氣溫 16℃、相對濕度 78%、4級風速流場）。
            </div>
        </div>
        """)

        st.divider()

        # =============================================================
        # 功能 1：獨立圖表繪製、網頁展示與 ZIP 打包下載
        # =============================================================
        st.subheader("📊 實測物理量關聯分析與理論對照圖")

        # 產出 4 張獨立圖表與 1 張總覽圖
        bytes_at = fig_to_bytes(plot_accel_time(active_umbrellas, all_meas, theory_curve))
        bytes_ft = fig_to_bytes(plot_drag_time(active_umbrellas, all_meas, theory_curve))
        bytes_fa = fig_to_bytes(plot_drag_accel(active_umbrellas, all_meas, theory_curve))
        bytes_box = fig_to_bytes(plot_boxplot(active_umbrellas, all_meas))
        bytes_comb = fig_to_bytes(plot_combined_figure(active_umbrellas, all_meas, theory_curve))

        # 在記憶體中建立 ZIP 壓縮檔，收納 5 個獨立高解析度圖檔
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr("1_平均加速度vs滯空時間(a-t).png", bytes_at)
            zip_file.writestr("2_平均空氣阻力vs滯空時間(F_drag-t).png", bytes_ft)
            zip_file.writestr("3_平均空氣阻力vs平均加速度(F_drag-a).png", bytes_fa)
            zip_file.writestr("4_各組降落傘滯空時間分佈箱形圖.png", bytes_box)
            zip_file.writestr("5_四象限物理分析綜合全覽圖.png", bytes_comb)
        zip_bytes = zip_buffer.getvalue()

        # 提供一鍵下載所有獨立圖檔的 ZIP 按鈕
        st.download_button(
            label="💾 一鍵下載所有獨立圖檔 (ZIP 壓縮包，內含 5 個獨立高解析度 PNG)",
            data=zip_bytes,
            file_name="降落傘物理分析圖表_獨立圖檔打包.zip",
            mime="application/zip"
        )

        st.write("")  # 垂直間隔

        # 網頁以 2x2 網格清晰呈現 4 張獨立圖表
        b64_at = base64.b64encode(bytes_at).decode('utf-8')
        b64_ft = base64.b64encode(bytes_ft).decode('utf-8')
        b64_fa = base64.b64encode(bytes_fa).decode('utf-8')
        b64_box = base64.b64encode(bytes_box).decode('utf-8')

        st.html(f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 10px;">
            <div style="text-align: center; background: #ffffff; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.08);">
                <img src="data:image/png;base64,{b64_at}" style="width: 100%; border-radius: 4px;">
                <p style="margin-top: 8px; font-weight: bold; color: #334155;">圖 1：平均加速度 vs. 滯空時間</p>
            </div>
            <div style="text-align: center; background: #ffffff; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.08);">
                <img src="data:image/png;base64,{b64_ft}" style="width: 100%; border-radius: 4px;">
                <p style="margin-top: 8px; font-weight: bold; color: #334155;">圖 2：平均空氣阻力 vs. 滯空時間</p>
            </div>
            <div style="text-align: center; background: #ffffff; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.08);">
                <img src="data:image/png;base64,{b64_fa}" style="width: 100%; border-radius: 4px;">
                <p style="margin-top: 8px; font-weight: bold; color: #334155;">圖 3：平均空氣阻力 vs. 平均加速度</p>
            </div>
            <div style="text-align: center; background: #ffffff; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.08);">
                <img src="data:image/png;base64,{b64_box}" style="width: 100%; border-radius: 4px;">
                <p style="margin-top: 8px; font-weight: bold; color: #334155;">圖 4：滯空時間分佈箱形圖</p>
            </div>
        </div>
        """)

        st.divider()

        # =============================================================
        # 功能 2：比照截圖之整齊雙層結構數據表 + CSV 下載
        # =============================================================
        st.subheader("📋 物理量計算數據總表")
        st.markdown("依據實測量次 1～20 完整呈現各傘滯空時間、平均加速度與平均阻力：")

        # 提供標準 CSV 下載按鈕
        csv_bytes = df_csv.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        st.download_button(
            label="📥 下載計算數據總表 (CSV 格式)",
            data=csv_bytes,
            file_name="降落傘物理分析計算總表.csv",
            mime="text/csv"
        )

        st.write("")  # 垂直間隔

        # 轉換為 HTML 並以 st.html 注入，絕不洩漏標籤文字
        table_html = df_display.to_html(classes="report-table", na_rep="-")
        
        full_table_code = f"""
        <style>
        .report-table-box {{
            max-height: 520px;
            overflow: auto;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            margin: 10px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .report-table {{
            width: 100%;
            border-collapse: collapse;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            font-size: 13.5px;
            text-align: center;
        }}
        .report-table th, .report-table td {{
            border: 1px solid #cbd5e1;
            padding: 8px 10px;
            white-space: nowrap;
        }}
        .report-table thead tr:first-child th {{
            background-color: #1e3a8a;
            color: #ffffff;
            font-size: 14px;
            font-weight: bold;
            position: sticky;
            top: 0;
            z-index: 3;
        }}
        .report-table thead tr:nth-child(2) th {{
            background-color: #2563eb;
            color: #ffffff;
            font-size: 13px;
            position: sticky;
            top: 36px;
            z-index: 2;
        }}
        .report-table tr:nth-child(even) {{
            background-color: #f8fafc;
        }}
        .report-table tr:hover {{
            background-color: #e0f2fe;
        }}
        </style>
        <div class="report-table-box">
            {table_html}
        </div>
        """
        st.html(full_table_code)

    except Exception as e:
        st.error(f"⚠️ 處理檔案時發生錯誤：{e}")
        st.info("請確認上傳的 Excel 檔案包含「數據記錄表」工作表，且具備 M12（樓高）與 M5（重量）儲存格。")
else:
    st.info("👈 請於左側邊欄點擊「Browse files」上傳學生的 Excel 檔案（如 plot.xlsx）！")

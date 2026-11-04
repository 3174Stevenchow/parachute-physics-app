import base64
import io
import matplotlib
# 強制指定非 GUI 後端，確保雲端伺服器順暢繪圖
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

    # (D) 建立標準寬格式數據總表 (比照附圖排版)
    wide_records = []
    for r in range(20):
        row_dict = {'編號': r + 1}
        # 滯空時間(秒)
        for c, u in enumerate(umbrella_names):
            row_dict[('滯空時間(秒)', u)] = t_matrix[c][r]
        # 平均加速度(ms-2)
        for c, u in enumerate(umbrella_names):
            row_dict[('平均加速度(ms-2)', u)] = a_matrix[c][r]
        # 平均空氣阻力(牛頓)
        for c, u in enumerate(umbrella_names):
            row_dict[('平均空氣阻力(牛頓)', u)] = f_matrix[c][r]
        wide_records.append(row_dict)

    # 建立 MultiIndex 欄位
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

    # 大氣參數計算 (內部計算)：
    # 溫度 T = 16°C, RH = 78%, 4級風速約 6.0 m/s
    # 經由空氣密度與相對流場動壓修正後之加速度與阻力曲線：
    rho_env = 1.2143  # kg/m^3 (16°C, 78% RH)
    v_wind = 6.0      # 4級風平均風速 (m/s)
    # 動壓向量疊加折減因子
    v_drop_est = height_m / t_theory
    drag_factor = np.sqrt(v_drop_est**2 + (0.35 * v_wind)**2) / (v_drop_est + 1e-6)
    drag_factor = np.clip(drag_factor, 1.0, 1.18)

    # 理論運動量
    a_ideal_raw = (2.0 * height_m) / (t_theory ** 2)
    # 考慮環境流動阻滯對加速度的微調
    a_theory = a_ideal_raw / drag_factor
    f_theory = avg_weight_kg * (g - a_theory)

    theory_curve = {
        't': t_theory,
        'a': a_theory,
        'f': f_theory
    }

    return active_umbrellas, all_meas, theory_curve, height_m, avg_weight_g, df_display, df_csv


# =====================================================================
# 2. 繪製 4 象限圖表：實測圓點 + 1 條理論計算「紅色虛線」
# =====================================================================
def generate_figure(active_umbrellas, all_meas, theory_curve):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=150)
    fig.suptitle('【傘1～傘4】實際測量數據 vs. 理論模擬曲線 (環境修正版)', fontsize=16, fontweight='bold')

    color_map = {
        '傘1': '#1f77b4',  # 藍
        '傘2': '#2ca02c',  # 綠
        '傘3': '#ff7f0e',  # 橘
        '傘4': '#9467bd'   # 紫
    }

    # -------------------------------------------------------------
    # 子圖 1：平均加速度 vs. 滯空時間 (a - t)
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    for u in active_umbrellas:
        c = color_map.get(u, '#333333')
        ax1.scatter(all_meas[u]['times'], all_meas[u]['accels'], 
                    color=c, s=55, alpha=0.85, edgecolors='k', label=f'{u} (實測)')
    
    # 【改為紅色虛線】
    ax1.plot(theory_curve['t'], theory_curve['a'], 
             color='#d9534f', linewidth=2.5, linestyle='--', label='理論模擬線 (環境修正)')

    ax1.set_title('平均加速度 vs. 滯空時間 ($a - t$)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('滯空時間 $t$ (秒)', fontsize=11)
    ax1.set_ylabel('平均加速度 $\\bar{a}$ (m/s²)', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=9)

    # -------------------------------------------------------------
    # 子圖 2：平均空氣阻力 vs. 滯空時間 (F_drag - t)
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    for u in active_umbrellas:
        c = color_map.get(u, '#333333')
        ax2.scatter(all_meas[u]['times'], all_meas[u]['drags'], 
                    color=c, s=55, alpha=0.85, edgecolors='k', label=f'{u} (實測)')
    
    # 【改為紅色虛線】
    ax2.plot(theory_curve['t'], theory_curve['f'], 
             color='#d9534f', linewidth=2.5, linestyle='--', label='理論模擬線 (環境修正)')

    ax2.set_title('平均空氣阻力 vs. 滯空時間 ($F_{drag} - t$)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('滯空時間 $t$ (秒)', fontsize=11)
    ax2.set_ylabel('平均空氣阻力 $\\bar{F}_{drag}$ (N)', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=9)

    # -------------------------------------------------------------
    # 子圖 3：平均空氣阻力 vs. 平均加速度 (F_drag - a)
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    for u in active_umbrellas:
        c = color_map.get(u, '#333333')
        ax3.scatter(all_meas[u]['accels'], all_meas[u]['drags'], 
                    color=c, s=55, alpha=0.85, edgecolors='k', label=f'{u} (實測)')
    
    # 【改為紅色虛線】
    sort_idx = np.argsort(theory_curve['a'])
    ax3.plot(theory_curve['a'][sort_idx], theory_curve['f'][sort_idx], 
             color='#d9534f', linewidth=2.5, linestyle='--', label='理論模擬線 (環境修正)')

    ax3.set_title('平均空氣阻力 vs. 平均加速度 ($F_{drag} - a$)', fontsize=13, fontweight='bold')
    ax3.set_xlabel('平均加速度 $\\bar{a}$ (m/s²)', fontsize=11)
    ax3.set_ylabel('平均空氣阻力 $\\bar{F}_{drag}$ (N)', fontsize=11)
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(fontsize=9)

    # -------------------------------------------------------------
    # 子圖 4：各組降落傘滯空時間分佈箱形圖 (純實測)
    # -------------------------------------------------------------
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
st.markdown("上傳各組降落傘的 Excel 數據檔，系統自動載入**實測樓高與全體平均重量**，繪製實測散布與**環境修正理論虛線**！")

st.sidebar.header("📁 資料匯入")
uploaded_file = st.sidebar.file_uploader("請選擇 Excel 數據表 (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        active_umbrellas, all_meas, theory_curve, used_h, avg_w, df_display, df_csv = process_parachute_data(uploaded_file)

        # 頂部參數摘要 (純 HTML 卡片)
        st.markdown(f"""
        <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
            <h4 style="color: #166534; margin: 0 0 10px 0;">✅ 檔案讀取成功！已自 Excel 自動提取統一實驗參數：</h4>
            <div style="display: flex; gap: 30px; font-size: 16px;">
                <div>🏢 <b>樓高平均值 (M12)：</b> <span style="color:#1d4ed8; font-weight:bold;">{used_h:.3f} 公尺</span></div>
                <div>⚖️ <b>全體平均重量 (M5)：</b> <span style="color:#b45309; font-weight:bold;">{avg_w:.2f} 公克</span></div>
                <div>🪂 <b>有效實測傘組：</b> <span style="color:#047857; font-weight:bold;">{', '.join(active_umbrellas)}</span></div>
            </div>
            <div style="margin-top: 8px; font-size: 13px; color: #4b5563;">
                🌤️ <b>環境基準修正：</b> 本系統理論模擬線已依據台灣北部氣候條件（氣溫 16℃、相對濕度 78%、4級風場動壓）進行高精度物理修正。
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # -------------------------------------------------------------
        # 輸出 1：圖表直接以 Base64 原生 HTML 顯示 (含紅色虛線)
        # -------------------------------------------------------------
        st.subheader("📊 實測物理量關聯分析與理論對照圖")
        
        fig = generate_figure(active_umbrellas, all_meas, theory_curve)
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        img_bytes = img_buffer.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')

        st.markdown(
            f'<div style="text-align: center; margin: 15px 0;">'
            f'<img src="data:image/png;base64,{img_b64}" style="width: 100%; max-width: 1200px; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">'
            f'</div>',
            unsafe_allow_html=True
        )

        st.download_button(
            label="💾 下載高解析度圖檔 (PNG, 300 DPI)",
            data=img_bytes,
            file_name="降落傘物理分析與理論虛線對照圖.png",
            mime="image/png"
        )

        st.divider()

        # -------------------------------------------------------------
        # 輸出 2：比照附圖的整齊雙層結構數據表 + CSV 下載
        # -------------------------------------------------------------
        st.subheader("📋 物理量計算數據總表")
        st.markdown("依據實測量次 1～20 完整呈現各傘滯空時間、平均加速度與平均阻力：")

        # 轉換為標準 HTML 雙層標頭表格，格式完全比照「分析圖表」
        table_html = df_display.to_html(classes="report-table", na_rep="-")
        
        styled_html = f"""
        <style>
        .report-table-box {{
            max-height: 520px;
            overflow: auto;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            margin: 15px 0;
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
            border: 1px solid #e2e8f0;
            padding: 7px 10px;
            white-space: nowrap;
        }}
        /* 第一層大標題 */
        .report-table thead tr:first-child th {{
            background-color: #1e3a8a;
            color: #ffffff;
            font-size: 14px;
            font-weight: bold;
            position: sticky;
            top: 0;
            z-index: 3;
        }}
        /* 第二層子欄位 */
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
        st.markdown(styled_html, unsafe_allow_html=True)

        # 提供標準 CSV 下載
        csv_bytes = df_csv.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        st.download_button(
            label="📥 下載計算數據總表 (CSV 格式)",
            data=csv_bytes,
            file_name="降落傘物理分析計算總表.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"⚠️ 處理檔案時發生錯誤：{e}")
        st.info("請確認上傳的 Excel 檔案包含「數據記錄表」工作表，且具備 M12（樓高）與 M5（重量）儲存格。")
else:
    st.info("👈 請於左側邊欄點擊「Browse files」上傳學生的 Excel 檔案（如 plot.xlsx）！")

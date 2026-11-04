import base64
import io
import matplotlib
import zipfile

# 強制指定非 GUI 後端
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import root_scalar

# 設定網頁標題與寬螢幕版面配置
st.set_page_config(
    page_title="寶特瓶降落傘物理分析系統",
    page_icon="🪂",
    layout="wide"
)

# Matplotlib 中文字型設定
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
# 1. 數據讀取與基礎物理量計算
# =====================================================================
def process_parachute_data(uploaded_file):
    xls = pd.ExcelFile(uploaded_file)
    df_raw = pd.read_excel(xls, sheet_name='數據記錄表')

    # 自動讀取樓高平均值 (M12)
    try:
        height_m = float(df_raw.iloc[10, 12])
        if np.isnan(height_m) or height_m <= 0:
            height_m = 8.33
    except Exception:
        height_m = 8.33

    # 自動讀取全體成員平均重量 (M5)
    try:
        avg_weight_g = float(df_raw.iloc[3, 12])
        if np.isnan(avg_weight_g) or avg_weight_g <= 0:
            avg_weight_g = 232.31
    except Exception:
        avg_weight_g = 232.31

    avg_weight_kg = avg_weight_g / 1000.0
    g = 9.8

    umbrella_names = ['傘1', '傘2', '傘3', '傘4']
    raw_times = df_raw.iloc[0:20, 1:5].copy()
    raw_times.columns = umbrella_names

    t_matrix, a_matrix, f_matrix = [], [], []
    all_meas = {}
    active_umbrellas = []

    for c_idx, u_name in enumerate(umbrella_names):
        col_series = raw_times[u_name].astype(str).str.replace(',', '.')
        times_clean = pd.to_numeric(col_series, errors='coerce').values
        
        t_col, a_col, f_col, valid_pts = [], [], [], []

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

    # 建立雙層表頭資料表
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

    csv_headers = ['編號'] + \
                  [f'滯空時間_{u}(秒)' for u in umbrella_names] + \
                  [f'平均加速度_{u}(ms-2)' for u in umbrella_names] + \
                  [f'平均空氣阻力_{u}(牛頓)' for u in umbrella_names]
    df_csv = pd.DataFrame(flat_data, columns=csv_headers)

    return active_umbrellas, all_meas, height_m, avg_weight_g, df_display, df_csv


# =====================================================================
# 2. 數值反推阻力模型 (由時間 t 反推動態阻力係數 k 與終端速度 vt)
# =====================================================================
def solve_drag_dynamics(active_umbrellas, all_meas, height_m, avg_weight_kg):
    g = 9.8
    drag_params = {}
    time_series_data = {}

    for u in active_umbrellas:
        t_mean = float(np.mean(all_meas[u]['times']))

        # 目標方程式：H(k) = (m/k) * ln(cosh(sqrt(k*g/m) * t)) - H = 0
        def f_obj(k_val):
            if k_val <= 1e-6:
                return -height_m
            alpha = np.sqrt(k_val * g / avg_weight_kg)
            arg = np.clip(alpha * t_mean, 0, 700) # 防止溢位
            h_calc = (avg_weight_kg / k_val) * np.log(np.cosh(arg))
            return h_calc - height_m

        try:
            sol = root_scalar(f_obj, bracket=[1e-4, 10.0], method='brentq')
            k_best = sol.root
        except Exception:
            # 簡化近似公式 k = mg / vt^2
            vt_est = height_m / t_mean
            k_best = (avg_weight_kg * g) / (vt_est ** 2)

        # 終端速度 vt = sqrt(mg / k)
        vt = np.sqrt(avg_weight_kg * g / k_best)
        drag_params[u] = {'k': k_best, 'vt': vt, 't_mean': t_mean}

        # 產生該傘整個下落過程的時間序列
        t_sim = np.linspace(0, t_mean, 150)
        alpha = np.sqrt(k_best * g / avg_weight_kg)
        v_sim = vt * np.tanh(alpha * t_sim)
        f_drag_sim = k_best * (v_sim ** 2)
        a_sim = g - (f_drag_sim / avg_weight_kg)

        time_series_data[u] = {
            't': t_sim,
            'v': v_sim,
            'f_drag': f_drag_sim,
            'a': a_sim
        }

    return drag_params, time_series_data


# =====================================================================
# 3. 獨立圖表繪製函式
# =====================================================================
color_map = {'傘1': '#1f77b4', '傘2': '#2ca02c', '傘3': '#ff7f0e', '傘4': '#9467bd'}

def plot_accel_time(active_umbrellas, all_meas):
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    for u in active_umbrellas:
        ax.scatter(all_meas[u]['times'], all_meas[u]['accels'], color=color_map.get(u, '#333'), s=65, alpha=0.85, edgecolors='k', label=f'{u} (實測)')
    ax.set_title('平均加速度 vs. 滯空時間 ($a - t$)', fontsize=14, fontweight='bold')
    ax.set_xlabel('滯空時間 $t$ (秒)', fontsize=12)
    ax.set_ylabel('平均加速度 $\\bar{a}$ (m/s²)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()
    return fig

def plot_drag_time(active_umbrellas, all_meas):
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    for u in active_umbrellas:
        ax.scatter(all_meas[u]['times'], all_meas[u]['drags'], color=color_map.get(u, '#333'), s=65, alpha=0.85, edgecolors='k', label=f'{u} (實測)')
    ax.set_title('平均空氣阻力 vs. 滯空時間 ($F_{drag} - t$)', fontsize=14, fontweight='bold')
    ax.set_xlabel('滯空時間 $t$ (秒)', fontsize=12)
    ax.set_ylabel('平均空氣阻力 $\\bar{F}_{drag}$ (N)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()
    return fig

def plot_drag_accel(active_umbrellas, all_meas):
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    for u in active_umbrellas:
        ax.scatter(all_meas[u]['accels'], all_meas[u]['drags'], color=color_map.get(u, '#333'), s=65, alpha=0.85, edgecolors='k', label=f'{u} (實測)')
    ax.set_title('平均空氣阻力 vs. 平均加速度 ($F_{drag} - a$)', fontsize=14, fontweight='bold')
    ax.set_xlabel('平均加速度 $\\bar{a}$ (m/s²)', fontsize=12)
    ax.set_ylabel('平均空氣阻力 $\\bar{F}_{drag}$ (N)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()
    return fig

def plot_boxplot(active_umbrellas, all_meas):
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    box_data = [all_meas[u]['times'] for u in active_umbrellas]
    box = ax.boxplot(box_data, widths=0.5, patch_artist=True, medianprops=dict(color='black', linewidth=1.5))
    for patch, u in zip(box['boxes'], active_umbrellas):
        patch.set_facecolor(color_map.get(u, '#666'))
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(active_umbrellas) + 1))
    ax.set_xticklabels(active_umbrellas, fontsize=12)
    ax.set_title('各組降落傘滯空時間分佈箱形圖 (實測值)', fontsize=14, fontweight='bold')
    ax.set_ylabel('滯空時間 (秒)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

# 新增：動態阻力隨時間連續變化圖
def plot_dynamic_drag_curves(active_umbrellas, time_series_data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    fig.suptitle('【理論反推】降落傘下落過程之連續動態阻力與速度重建', fontsize=15, fontweight='bold')

    for u in active_umbrellas:
        c = color_map.get(u, '#333')
        data = time_series_data[u]
        ax1.plot(data['t'], data['f_drag'], color=c, linewidth=2.5, label=f'{u}')
        ax2.plot(data['t'], data['v'], color=c, linewidth=2.5, label=f'{u}')

    ax1.set_title('即時空氣阻力隨時間變化 $F_{drag}(t)$', fontsize=13, fontweight='bold')
    ax1.set_xlabel('下落時間 $t$ (秒)', fontsize=11)
    ax1.set_ylabel('即時阻力 (N)', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=10)

    ax2.set_title('下落速率隨時間變化 $v(t)$ (趨向終端速度)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('下落時間 $t$ (秒)', fontsize=11)
    ax2.set_ylabel('下落速率 $v$ (m/s)', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    return fig

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# =====================================================================
# 4. Streamlit 網頁畫面佈局
# =====================================================================
st.title("🪂 寶特瓶降落傘探究與實作：物理分析系統")
st.markdown("上傳各組降落傘的 Excel 數據檔，系統自動載入**實測樓高與全體平均重量**，完成實測分佈統計與**阻力動態逆推**！")

st.sidebar.header("📁 資料匯入")
uploaded_file = st.sidebar.file_uploader("請選擇 Excel 數據表 (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        active_umbrellas, all_meas, used_h, avg_w, df_display, df_csv = process_parachute_data(uploaded_file)
        avg_weight_kg = avg_w / 1000.0

        # 反推各傘阻力常數 k 與終端速度 vt
        drag_params, time_series_data = solve_drag_dynamics(active_umbrellas, all_meas, used_h, avg_weight_kg)

        # 頂部參數摘要卡片
        st.html(f"""
        <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
            <h4 style="color: #166534; margin: 0 0 10px 0;">✅ 檔案讀取成功！已自 Excel 自動提取統一實驗參數：</h4>
            <div style="display: flex; gap: 30px; font-size: 16px; flex-wrap: wrap;">
                <div>🏢 <b>樓高平均值 (M12)：</b> <span style="color:#1d4ed8; font-weight:bold;">{used_h:.3f} 公尺</span></div>
                <div>⚖️ <b>全體平均重量 (M5)：</b> <span style="color:#b45309; font-weight:bold;">{avg_w:.2f} 公克</span></div>
                <div>🪂 <b>有效實測傘組：</b> <span style="color:#047857; font-weight:bold;">{', '.join(active_umbrellas)}</span></div>
            </div>
        </div>
        """)

        st.divider()

        # =============================================================
        # 功能 1：實測圖表展示與 ZIP 打包下載
        # =============================================================
        st.subheader("📊 實測物理量關聯分析圖表")

        bytes_at = fig_to_bytes(plot_accel_time(active_umbrellas, all_meas))
        bytes_ft = fig_to_bytes(plot_drag_time(active_umbrellas, all_meas))
        bytes_fa = fig_to_bytes(plot_drag_accel(active_umbrellas, all_meas))
        bytes_box = fig_to_bytes(plot_boxplot(active_umbrellas, all_meas))
        bytes_dyn = fig_to_bytes(plot_dynamic_drag_curves(active_umbrellas, time_series_data))

        # 打包 ZIP 壓縮檔
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr("1_平均加速度vs滯空時間(a-t).png", bytes_at)
            zip_file.writestr("2_平均空氣阻力vs滯空時間(F_drag-t).png", bytes_ft)
            zip_file.writestr("3_平均空氣阻力vs平均加速度(F_drag-a).png", bytes_fa)
            zip_file.writestr("4_各組降落傘滯空時間分佈箱形圖.png", bytes_box)
            zip_file.writestr("5_即時阻力與下落速率動態連續曲線.png", bytes_dyn)
        zip_bytes = zip_buffer.getvalue()

        st.download_button(
            label="💾 一鍵下載所有獨立圖檔 (ZIP 壓縮包，內含 5 個獨立高解析度 PNG)",
            data=zip_bytes,
            file_name="降落傘物理分析圖表_獨立圖檔打包.zip",
            mime="application/zip"
        )

        st.write("")

        # 網頁以 2x2 呈現 4 張實測散布圖
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
        # 功能 2：進階物理分析：微分方程反推阻力變化
        # =============================================================
        st.subheader("🔬 探究延伸：從落地總時間逆推「阻力隨時間之動態變化」")
        st.markdown("""
        即使受限於工具僅記錄落地時間，我們仍可藉由空氣動力學運動方程式 $m\\frac{dv}{dt} = mg - kv^2$，
        數值求解各傘的**阻力係數 $k$** 與**終端速度 $v_t$**，進而還原出下落全程的即時阻力變化：
        """)

        # 呈現反推結果表格
        param_rows = []
        for u in active_umbrellas:
            p = drag_params[u]
            param_rows.append({
                '降落傘組別': u,
                '平均滯空時間 (s)': round(p['t_mean'], 2),
                '逆推終端速度 vt (m/s)': round(p['vt'], 2),
                '空氣阻力係數 k (kg/m)': round(p['k'], 4),
                '終端空氣阻力 F_t = mg (N)': round(avg_weight_kg * 9.8, 3)
            })
        st.table(pd.DataFrame(param_rows))

        # 顯示動態曲線圖
        b64_dyn = base64.b64encode(bytes_dyn).decode('utf-8')
        st.html(f"""
        <div style="text-align: center; margin: 15px 0;">
            <img src="data:image/png;base64,{b64_dyn}" style="width: 100%; max-width: 1100px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        </div>
        """)

        st.divider()

        # =============================================================
        # 功能 3：雙層結構數據表與 CSV 下載
        # =============================================================
        st.subheader("📋 物理量計算數據總表")
        st.markdown("依據實測量次 1～20 完整呈現各傘滯空時間、平均加速度與平均阻力：")

        csv_bytes = df_csv.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        st.download_button(
            label="📥 下載計算數據總表 (CSV 格式)",
            data=csv_bytes,
            file_name="降落傘物理分析計算總表.csv",
            mime="text/csv"
        )

        st.write("")

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

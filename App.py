import streamlit as st
import trino
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px

# ========== 資料庫連線設定 ==========
host = 'edwml2'
port = 18081

CATALOG_SPC = 'ctoffspc'
SCHEMA_SPC = 'offspcuser'
TABLE_SPC = 'cfg_spc_spec_l2'

CATALOG_RESULT = 'cimdb'
SCHEMA_RESULT = 'edauser'
TABLE_RESULT = 'met_result_l2'

CATALOG_FB = 'ctfabrpt'
SCHEMA_FB = 'reportuser'
TABLE_FB = 'fb_flow_prod_group_l2'

CATALOG_WAFER = 'cimdb'
SCHEMA_WAFER = 'edauser'
TABLE_WAFER = 'wip_wafer_hist_l2'

def login_form():
    st.title("Trino 資料庫登入")
    with st.form("login_form", clear_on_submit=False):
        input_user = st.text_input("帳號 (user)")
        input_pwd = st.text_input("密碼 (password)", type="password")
        submit = st.form_submit_button("登入")
        if submit:
            if input_user.strip() and input_pwd.strip():
                st.session_state['user'] = input_user.strip()
                st.session_state['password'] = input_pwd.strip()
                st.success("登入成功！")
                st.rerun()
            else:
                st.error("請輸入帳號與密碼")

if 'user' not in st.session_state or 'password' not in st.session_state:
    login_form()
    st.stop()

user = st.session_state['user']
password = st.session_state['password']

@st.cache_data(show_spinner=False)
def get_step_name_list(prod_group, user, password):
    with trino.dbapi.connect(
        host=host, port=port, user=user, catalog=CATALOG_SPC, schema=SCHEMA_SPC,
        http_scheme='https', auth=trino.auth.BasicAuthentication(user, password), verify=False
    ) as conn:
        sql = f"""
            SELECT DISTINCT step_code, step_name
            FROM {TABLE_SPC}
            WHERE product_group = '{prod_group}'
            ORDER BY step_code, step_name
        """
        df = pd.read_sql(sql, conn)
        df['step_show'] = df['step_code'].astype(str) + '-' + df['step_name'].astype(str)
        return df[['step_code', 'step_name', 'step_show']].to_dict('records')

@st.cache_data(show_spinner=False)
def get_parameter_name_list(prod_group, step_name, user, password):
    if not (prod_group and step_name):
        return []
    with trino.dbapi.connect(
        host=host, port=port, user=user, catalog=CATALOG_SPC, schema=SCHEMA_SPC,
        http_scheme='https', auth=trino.auth.BasicAuthentication(user, password), verify=False
    ) as conn:
        sql = f"""
            SELECT DISTINCT parameter_name
            FROM {TABLE_SPC}
            WHERE product_group = '{prod_group}'
              AND step_name = '{step_name}'
            ORDER BY parameter_name
        """
        return pd.read_sql(sql, conn)['parameter_name'].dropna().tolist()

def fetch_final_result_by_name(step_name, parameter_name, dt_start, dt_end, product_group, user, password):
    with trino.dbapi.connect(
        host=host, port=port, user=user, catalog=CATALOG_RESULT, schema=SCHEMA_RESULT,
        http_scheme='https', auth=trino.auth.BasicAuthentication(user, password), verify=False
    ) as conn:
        sql = f"""
            SELECT *
            FROM {TABLE_RESULT}
            WHERE step_id = '{step_name}'
              AND param_id = '{parameter_name}'
              AND param_group LIKE '-{product_group}-%'
              AND met_dt BETWEEN DATE '{dt_start}' AND DATE '{dt_end}'
            ORDER BY met_dt ASC
        """
        return pd.read_sql(sql, conn)

@st.cache_data(show_spinner=False)
def get_step_id_options(prod_group, user, password):
    sql = f"""
        SELECT DISTINCT step_cd, step
        FROM {TABLE_FB}
        WHERE prod_group = '{prod_group}'
        ORDER BY step_cd, step
    """
    with trino.dbapi.connect(
        host=host, port=port, user=user, catalog=CATALOG_FB, schema=SCHEMA_FB,
        http_scheme='https', auth=trino.auth.BasicAuthentication(user, password), verify=False
    ) as conn:
        df = pd.read_sql(sql, conn)
    df['option_label'] = df['step_cd'].astype(str) + df['step'].astype(str)
    return df[['step_cd', 'step', 'option_label']].to_dict('records')

@st.cache_data(show_spinner=False)
def get_wafer_param_batch(wafer_ids, step, param, user, password):
    if not wafer_ids or not step or not param:
        return pd.DataFrame()
    wafer_list = "','".join([str(w) for w in wafer_ids])
    sql = f"""
        SELECT wafer_id, step_id, {param}
        FROM {TABLE_WAFER}
        WHERE wafer_id IN ('{wafer_list}')
          AND step_id = '{step}'
    """
    with trino.dbapi.connect(
        host=host, port=port, user=user, catalog=CATALOG_WAFER, schema=SCHEMA_WAFER,
        http_scheme='https', auth=trino.auth.BasicAuthentication(user, password), verify=False
    ) as conn:
        df = pd.read_sql(sql, conn)
    return df[['wafer_id', param]]

def show_plot_cp_cpk(show_df, color=None):
    required_cols = ['met_dt', 'wafer_id', 'value']
    if not all(col in show_df.columns for col in required_cols):
        st.error(f"缺少欄位：{required_cols}")
        return
    show_df = show_df.dropna(subset=['met_dt', 'wafer_id', 'value'])
    if show_df.empty:
        st.warning("無足夠資料繪圖")
        return
    show_df['met_dt'] = pd.to_datetime(show_df['met_dt']).dt.date.astype(str)
    show_df['x_label'] = show_df['met_dt'] + "_" + show_df['wafer_id'].astype(str)
    color_args = {'color': color} if color and (color in show_df.columns) else {}
    fig = px.line(
        show_df,
        x='x_label',
        y='value',
        markers=True,
        hover_data={
            "wafer_id": True,
            "value": ':.4f',
            "x_label": False
        },
        labels={"x_label": "met_dt + wafer_id", "value": "value"},
        title=f'value vs (met_dt+wafer_id){f" (by {color})" if color else ""}',
        **color_args
    )
    spec_lines = {
        "spec_high": {"style":"dash", "color":"red",   "pos":"top left"},
        "spec_low":  {"style":"dash", "color":"blue",  "pos":"bottom left"},
        "spec_target":{"style":"dot", "color":"green", "pos":"top right"},
    }
    for col, line_info in spec_lines.items():
        if col in show_df.columns:
            val = show_df[col].iloc[0]
            if pd.notnull(val):
                fig.add_hline(
                    y=val, 
                    line_dash=line_info["style"], 
                    line_color=line_info["color"], 
                    annotation_text=col, 
                    annotation_position=line_info["pos"]
                )
    fig.update_traces(marker=dict(size=12, symbol='circle'))
    fig.update_layout(xaxis_tickangle=-60, margin=dict(l=40, r=40, t=40, b=120))
    st.plotly_chart(fig, use_container_width=True)
    value_arr = show_df['value'].dropna().values
    USL = show_df['spec_high'].iloc[0] if 'spec_high' in show_df.columns else None
    LSL = show_df['spec_low'].iloc[0] if 'spec_low' in show_df.columns else None
    mu, sigma, cp, cpk = None, None, None, None
    st.markdown("#### 統計量")
    st.markdown(f"**Average (平均值):** {np.mean(value_arr):.4f}" if len(value_arr) > 0 else "**Average (平均值):** ")
    st.markdown(f"**Stdev (標準差):** {np.std(value_arr, ddof=1):.4f}" if len(value_arr) > 1 else "**Stdev (標準差):** ")
    if len(value_arr) > 1:
        mu = np.mean(value_arr)
        sigma = np.std(value_arr, ddof=1)
        if pd.notnull(USL) and pd.notnull(LSL) and sigma > 0:
            cp = (USL - LSL) / (6 * sigma)
            cpk = min((USL - mu) / (3 * sigma), (mu - LSL) / (3 * sigma))
    st.markdown(f"**CP:** {cp:.4f}" if cp is not None else "**CP:** ")
    st.markdown(f"**CPK:** {cpk:.4f}" if cpk is not None else "**CPK:** ")

def show_box_plot(df, group_col):
    if df.empty:
        st.warning("Box plot 無資料")
        return
    if group_col not in df.columns:
        st.warning(f"找不到分群欄位: {group_col}")
        return
    fig = px.box(
        df,
        x=group_col,
        y='value',
        points="all",
        title=f'Box Plot: value by {group_col}',
        labels={group_col: group_col, 'value': 'value'}
    )
    fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)

# ================== 主程式 ==================
st.set_page_config(page_title="逐級查詢：CP/CPK 與折線圖", layout='wide')
st.title("F24 Inline Trend Chart Monitor")

with st.sidebar:
    st.header("查詢操作")
    prod_group_options = sorted([
    'HAG096', 'TC8251', 'TC8259',
    'HAG110','TC8264','FAG090','FAG091','FAG102','FAG103','FAG109',
    'TC8255','TC8257','TC8258','AAG046','EAG077','EAG080','EAG085','EAG087',
    'EAG104','EAG105','EAG107','EAG108','EAG114','EAG115','EAG116','EAG119',
    'EAG120','EAG121','EAG122','EAG123','EAG124'
])

    prod_group_value = st.selectbox(
        "請選擇產品 (product_group)：",
        prod_group_options,
        key='prod_group'
    )

    with st.spinner("載入 Step Name..."):
        step_records = get_step_name_list(prod_group_value, user, password)
    if step_records:
        step_name = st.selectbox(
            "步驟1：請選擇 Step Name",
            options=step_records,
            format_func=lambda d: d['step_show'],
            key="step_name"
        )['step_name']
    else:
        step_name = ''

    with st.spinner("載入 Parameter Name..."):
        parameter_name_list = get_parameter_name_list(prod_group_value, step_name, user, password)
    parameter_name = st.selectbox(
        "步驟2：請選擇 Parameter Name",
        parameter_name_list if parameter_name_list else [],
        key='parameter_name'
    )

    dt_default_end = datetime.now().date()
    dt_default_start = dt_default_end - timedelta(days=30)
    dt_start = st.date_input("步驟3：起始 met_dt", value=dt_default_start, key='date_input_start')
    dt_end = st.date_input("步驟3：結束 met_dt", value=dt_default_end, key='date_input_end')
    if dt_start > dt_end:
        st.error("起始日期需小於等於結束日期")
        st.stop()

    if st.button('步驟4：查詢'):
        if not (prod_group_value and step_name and parameter_name and dt_start and dt_end):
            st.error("請先完成全部必要選擇，再執行查詢！")
            st.stop()
        with st.spinner("查詢資料中..."):
            df = fetch_final_result_by_name(
                step_name, parameter_name,
                dt_start.strftime('%Y-%m-%d'), dt_end.strftime('%Y-%m-%d'),
                prod_group_value, user, password
            )
        if df.empty:
            st.info("查無結果")
        else:
            df['exclude'] = False
            st.session_state['edited_df'] = df

# ===== 查詢結果區 =====
st.header("查詢結果與分析")
st.divider()
if 'edited_df' in st.session_state:
    st.markdown("#### 資料表（可勾選排除 exclude）")
    df = st.session_state['edited_df'].copy()
    if 'wafer_id' in df.columns and 'exclude' in df.columns:
        # 1. 先全部不排除
        df['exclude'] = False

        # 準備唯一 LotID 與 WaferID（升冪排序）
        df['lot_id'] = df['wafer_id'].astype(str).str[:9]
        unique_lotid = sorted(df['lot_id'].dropna().unique().tolist())
        unique_waferid = sorted(df['wafer_id'].dropna().astype(str).unique().tolist())

        # 互動選單
        exclude_eng = st.checkbox(
            "Exclude ENG (wafer id 以 S 開頭排除)", key="exclude_eng"
        )
        exclude_lotids = st.multiselect(
            "Exclude By LotID (可多選)", options=unique_lotid, key="exclude_lotid_multiselect"
        )
        exclude_waferids = st.multiselect(
            "Exclude By WaferID (可多選)", options=unique_waferid, key="exclude_waferid_multiselect"
        )

        # 依多選結果同步調整排除
        if exclude_eng:
            mask = df['wafer_id'].astype(str).str.startswith('S')
            df.loc[mask, 'exclude'] = True
        if exclude_lotids:
            mask = df['lot_id'].isin(exclude_lotids)
            df.loc[mask, 'exclude'] = True
        if exclude_waferids:
            mask = df['wafer_id'].astype(str).isin(exclude_waferids)
            df.loc[mask, 'exclude'] = True

        st.session_state['edited_df'] = df

    edited_df = st.data_editor(
        st.session_state['edited_df'],
        num_rows="dynamic",
        use_container_width=True,
        column_config={"exclude": st.column_config.CheckboxColumn('Exclude')}
    )
    st.session_state['edited_df'] = edited_df

    show_df = edited_df[edited_df['exclude'] == False].copy()

    # ========= 進階分析方塊，只在主查詢有結果時顯示 ==========
    with st.sidebar:
        st.markdown("---")
        st.subheader("進階分析 (顏色分群)")
        adv_step_options = get_step_id_options(prod_group_value, user, password)
        adv_step_selected = st.selectbox(
            "Step_id (進階分析)", 
            options=adv_step_options,
            format_func=lambda d: d['option_label'],
            key='adv_step_id'
        )
        adv_step_for_query = adv_step_selected['step']
        adv_param_options = ['eqp_id', 'recipe_name']
        selected_adv_param = st.selectbox(
            "參數 (進階分析)", options=adv_param_options, key='adv_param'
        )
        update_pressed = st.button("更新 (進階分析分群)", key="update_adv_plot")

    # ===== 並排顯示分群結果圖、以及可下載表格 =====
    if not update_pressed:
        show_plot_cp_cpk(show_df)
    else:
        wafer_ids = show_df['wafer_id'].dropna().astype(str).unique().tolist()
        adv_param_df = get_wafer_param_batch(
            wafer_ids, adv_step_for_query, selected_adv_param, user, password
        )
        if adv_param_df.empty:
            st.warning("進階分析查無參數結果，圖形維持原色 (uncolored)。")
            show_plot_cp_cpk(show_df)
        else:
            prompt_count = adv_param_df[selected_adv_param].nunique(dropna=True)
            merged_df = show_df.merge(adv_param_df, on='wafer_id', how='left')
            merged_df[selected_adv_param] = merged_df[selected_adv_param].fillna('(Unknown)')
            st.info(f"進階分析：{selected_adv_param} 共分辨 {prompt_count} 種類別")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Trend Chart (依進階分析分群顯示)")
                show_plot_cp_cpk(merged_df, color=selected_adv_param)
            with col2:
                st.markdown("#### Box Plot (依進階分析分群顯示)")
                show_box_plot(merged_df, selected_adv_param)

            # ==== 匯出 CSV 按鈕 ====
            csv = merged_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label='下載目前圖表資料 (CSV)',
                data=csv,
                file_name='trend_boxplot_data.csv',
                mime='text/csv'
            )
else:
    st.info("尚未查詢或無資料")

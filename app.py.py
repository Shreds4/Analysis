import streamlit as st
import pandas as pd
import numpy as np
import io

# --- Backend Functions --- #

def normalize_data(df):
    st.write("Normalizing data...")
    time = df.iloc[:, 0]
    data = df.iloc[:, 1:].replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all')
    numeric = data.select_dtypes(include='number')
    if (numeric.min() <= 0).any():
        st.warning("Warning: Some columns contain zero or negative minimums. Normalization might produce non-finite values.")
    norm = numeric.apply(lambda col: col / col.min(), axis=0)
    norm_df = pd.concat([time, norm], axis=1)
    norm_df.columns = df.columns
    return norm_df

def compute_interval_metrics(time, data, start, end, delta_t):
    mask = (time >= start) & (time <= end + delta_t)
    t_seg = time[mask].reset_index(drop=True)
    data_seg = data[mask].reset_index(drop=True)
    y_min_vals = data_seg.min().values
    auc_recs, trapezoids = [], {c: [] for c in data.columns}
    for i in range(len(t_seg) - 1):
        t1, t2 = t_seg.iloc[i], t_seg.iloc[i+1]
        row = {'Time Start': t1, 'Time End': t2}
        for j, col in enumerate(data.columns):
            y1, y2 = data_seg.iloc[i, j], data_seg.iloc[i+1, j]
            y_min = y_min_vals[j]
            area = ((y1 - y_min) + (y2 - y_min)) / 2 * (t2 - t1)
            row[col] = area
            trapezoids[col].append(area)
        auc_recs.append(row)
    auc_df = pd.DataFrame(auc_recs)
    auc_sums = {c: sum(trapezoids[c]) for c in data.columns}
    amp_vals = (data_seg.max() - data_seg.min()).to_dict()
    mins = (end - start) / 60
    summary = {
        'Average_AUC': np.mean(list(auc_sums.values())),
        'Average_AUC_per_min': np.mean(list(auc_sums.values())) / mins if mins > 0 else 0,
        'Average_Amplitude': np.mean(list(amp_vals.values())),
        'Avg_Amplitude_per_min': np.mean(list(amp_vals.values())) / mins if mins > 0 else 0
    }
    return auc_df, pd.Series(auc_sums), pd.Series(amp_vals), summary

def analyse_intervals(df, cut_pts):
    time = df.iloc[:, 0]
    data = df.iloc[:, 1:]
    dt = time.iloc[1] - time.iloc[0] if len(time) > 1 else 0
    cuts = sorted(cut_pts)
    intervals, s = [], 0
    for c in cuts:
        intervals.append((s, c))
        s = c + dt
    if not time.empty:
        intervals.append((s, time.iloc[-1]))

    auc_tables, auc_sums, amp_sums, meta_rows = {}, [], [], []
    for (a, b) in intervals:
        if a >= b:
            continue
        st.info(f"\U0001F4D0 Processing interval: {a}–{b}")
        auc_df, auc_sum, amp_vals, meta = compute_interval_metrics(time, data, a, b, dt)
        tag = f"{int(a)}-{int(b)}"
        auc_tables[tag] = auc_df
        auc_sum.name, amp_vals.name = tag, tag
        auc_sums.append(auc_sum)
        amp_sums.append(amp_vals)
        meta_rows.append(pd.Series(meta, name=tag))

    df_auc_sums = pd.concat(auc_sums, axis=1).T if auc_sums else pd.DataFrame()
    df_amp_sums = pd.concat(amp_sums, axis=1).T if amp_sums else pd.DataFrame()
    df_meta = pd.DataFrame(meta_rows) if meta_rows else pd.DataFrame()

    return auc_tables, df_auc_sums, df_amp_sums, df_meta, intervals

def compute_max_ratios(df, intervals, numer_idx, denom_idx, threshold):
    time = df.iloc[:, 0]
    data = df.iloc[:, 1:]
    sn, en = intervals[numer_idx]
    sd, ed = intervals[denom_idx]
    mask_n = (time >= sn) & (time <= en)
    mask_d = (time >= sd) & (time <= ed)
    max_n = data[mask_n].max()
    max_d = data[mask_d].max()
    ratio = max_n / max_d
    highlight = ratio > threshold
    return max_n, max_d, ratio, highlight

# --- Streamlit App --- #

st.set_page_config(layout="wide")
st.title("\U0001F52C Advanced Time‑Series Analysis App")

with st.sidebar:
    st.header("⚙️ Analysis Controls")
    analysis_mode = st.radio("Select Analysis Mode", ('Single File Analysis', 'Ratio Analysis'))
    uploaded_file = st.file_uploader("\U0001F4C2 Upload your Excel file", type=["xlsx"])
    cuts_input = st.text_input("✂️ Enter Time Cuts (comma‑separated)", "1230")
    apply_norm = st.toggle("Apply Normalization", value=False)
    run_interval_analysis = st.toggle("Enable Interval Ratio Analysis", value=False)
    st.markdown("---")
    st.subheader("\U0001F4CA Interval Ratio Analysis")
    threshold = st.number_input("\U0001F53A Highlight Threshold for Max Ratio", min_value=0.1, value=1.18, step=0.01)

if uploaded_file is not None:
    try:
        if analysis_mode == 'Single File Analysis':
            df_raw = pd.read_excel(uploaded_file, engine='openpyxl')
            base_df = df_raw
        else:
            xls = pd.ExcelFile(uploaded_file)
            if len(xls.sheet_names) < 2:
                st.error("\u274C Ratio mode requires at least two sheets.")
                st.stop()
            df1 = pd.read_excel(xls, 0)
            df2 = pd.read_excel(xls, 1)
            time_col = df1.iloc[:, 0]; d1 = df1.iloc[:, 1:]; d2 = df2.iloc[:, 1:]
            d2.columns = d1.columns
            base_df = pd.concat([time_col, d1 / d2], axis=1)

        analysis_df = normalize_data(base_df) if apply_norm else base_df
        label = f"{'Normalized' if apply_norm else 'Original'} {'Ratio Data' if analysis_mode != 'Single File Analysis' else 'Data'}"
        st.header(f"Data for Analysis: {label}")
        st.dataframe(analysis_df.head())

        max_ratio_df = pd.DataFrame()
        a_auc_df = pd.DataFrame()
        a_amp_df = pd.DataFrame()
        a_auc_per_min_df = pd.DataFrame()
        a_amp_per_min_df = pd.DataFrame()

        if run_interval_analysis:
            cuts = [int(c.strip()) for c in cuts_input.split(',') if c.strip().isdigit()]
            auc_tbls, auc_sum_df, amp_df, meta_df, intervals = analyse_intervals(analysis_df, cuts)

            interval_labels = [f"Interval {i+1}: {a}–{b}" for i, (a, b) in enumerate(intervals)]
            idx_map = {lbl: i for i, lbl in enumerate(interval_labels)}

            numerator_interval = None
            denominator_interval = None
            interval_selection_enabled = len(interval_labels) >= 2

            if interval_selection_enabled:
                st.sidebar.subheader("\U0001F501 Choose Intervals for Max Ratio")
                numerator_interval = st.sidebar.selectbox("Numerator Interval", interval_labels, index=0)
                denominator_interval = st.sidebar.selectbox("Denominator Interval", interval_labels, index=1)

            if interval_selection_enabled and numerator_interval and denominator_interval:
                try:
                    n_idx = idx_map[numerator_interval]
                    d_idx = idx_map[denominator_interval]
                    m_n, m_d, ratio, mask = compute_max_ratios(
                        analysis_df, intervals, n_idx, d_idx, threshold
                    )
                    max_ratio_df = pd.DataFrame({
                        "Numerator Max": m_n,
                        "Denominator Max": m_d,
                        "Max Ratio": ratio,
                        "Flagged": mask
                    })

                    flagged_cols = ratio[mask].index.tolist()
                    if flagged_cols:
                        sub_df = analysis_df[[analysis_df.columns[0]] + flagged_cols]
                        _, a_auc_df, a_amp_df, a_meta_df, _ = analyse_intervals(sub_df, cuts)
                        a_auc_per_min_df = a_meta_df[['Average_AUC_per_min']]
                        a_amp_per_min_df = a_meta_df[['Avg_Amplitude_per_min']]
                except Exception as e:
                    st.error(f"Interval ratio error: {e}")

        st.success("\u2705 Analysis complete!")

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            ws = writer.book.add_worksheet("Processed_Data")
            writer.sheets["Processed_Data"] = ws
            r = 0

            if analysis_mode == 'Single File Analysis':
                ws.write(r, 0, "Raw Data"); r += 1
                df_raw.to_excel(writer, sheet_name="Processed_Data", startrow=r, index=False)
                r += len(df_raw) + 2
            else:
                ws.write(r, 0, "Raw Data - Sheet 1"); r += 1
                df1.to_excel(writer, sheet_name="Processed_Data", startrow=r, index=False)
                r += len(df1) + 2
                ws.write(r, 0, "Raw Data - Sheet 2"); r += 1
                df2.to_excel(writer, sheet_name="Processed_Data", startrow=r, index=False)
                r += len(df2) + 2

            ws.write(r, 0, label); r += 1
            analysis_df.to_excel(writer, sheet_name="Processed_Data", startrow=r, index=False)
            r += len(analysis_df) + 2

            if run_interval_analysis:
                if not max_ratio_df.empty:
                    ws.write(r, 0, f"Max Ratio Table: {numerator_interval} \u00F7 {denominator_interval}"); r += 1
                    max_ratio_df.to_excel(writer, sheet_name="Processed_Data", startrow=r, index=True)
                    fmt = writer.book.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
                    col_idx = list(max_ratio_df.columns).index("Max Ratio")
                    ws.conditional_format(r, col_idx+1, r+len(max_ratio_df), col_idx+1,
                                          {'type': 'cell', 'criteria': '>', 'value': threshold, 'format': fmt})
                    r += len(max_ratio_df) + 2

                for idx, (tag, auc_df_int) in enumerate(auc_tbls.items(), start=1):
                    ws.write(r, 0, f"AUC Data - Interval {idx} ({tag})"); r += 1
                    auc_df_int.rename(columns={'Time Start': 'Time'}).drop(columns=['Time End'], errors='ignore') \
                              .to_excel(writer, sheet_name="Processed_Data", startrow=r, index=False)
                    r += len(auc_df_int) + 2
                    ws.write(r, 0, f"AUC Sums - Interval {idx}"); r += 1
                    auc_sum_df.loc[tag:tag].to_excel(writer, sheet_name="Processed_Data", startrow=r)
                    r += 2
                    ws.write(r, 0, f"AUC Average - Interval {idx}"); r += 1
                    ws.write(r, 0, meta_df.loc[tag, 'Average_AUC']); r += 2
                    ws.write(r, 0, f"AUC per Minute - Interval {idx}"); r += 1
                    ws.write(r, 0, meta_df.loc[tag, 'Average_AUC_per_min']); r += 2
                    ws.write(r, 0, f"Amplitude - Interval {idx}"); r += 1
                    amp_df.loc[tag:tag].to_excel(writer, sheet_name="Processed_Data", startrow=r)
                    r += len(amp_df.loc[tag:tag]) + 2
                    ws.write(r, 0, f"Average Amplitude - Interval {idx}"); r += 1
                    ws.write(r, 0, meta_df.loc[tag, 'Average_Amplitude']); r += 2
                    ws.write(r, 0, f"Amp per Minute - Interval {idx}"); r += 1
                    ws.write(r, 0, meta_df.loc[tag, 'Avg_Amplitude_per_min']); r += 2

                if not a_auc_df.empty:
                    r += 2
                    ws.write(r, 0, "A‑Cell AUC Data"); r += 1
                    a_auc_df.to_excel(writer, sheet_name="Processed_Data", startrow=r, index=True); r += len(a_auc_df) + 2
                    ws.write(r, 0, "A‑Cell Amplitude Data"); r += 1
                    a_amp_df.to_excel(writer, sheet_name="Processed_Data", startrow=r, index=True); r += len(a_amp_df) + 2
                    ws.write(r, 0, "A‑Cell AUC per Minute"); r += 1
                    a_auc_per_min_df.to_excel(writer, sheet_name="Processed_Data", startrow=r, index=True); r += len(a_auc_per_min_df) + 2
                    ws.write(r, 0, "A‑Cell Amplitude per Minute"); r += 1
                    a_amp_per_min_df.to_excel(writer, sheet_name="Processed_Data", startrow=r, index=True); r += len(a_amp_per_min_df) + 2

        st.download_button(
            label="\U0001F4E5 Download Analysis Excel File",
            data=output.getvalue(),
            file_name="complete_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"\u26D4 Error: {e}")
        st.warning("Double check your uploads, time cuts, and interval selections.")

import streamlit as st
import pandas as pd
import numpy as np
import io

# --- Core Backend Functions (These are the calculation engines) ---

def normalize_data(df):
    """Normalizes the numeric columns of a DataFrame by dividing by the column minimum."""
    st.write("Normalizing data...") # Show progress in the app
    time = df.iloc[:, 0]
    # Clean data before normalization to avoid errors with infinity or missing values
    data = df.iloc[:, 1:].replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all')
    numeric = data.select_dtypes(include='number')
    
    # Check for non-positive minimums which can cause issues
    if (numeric.min() <= 0).any():
        st.warning("Warning: Some columns contain zero or negative minimums. Normalization might produce non-finite values.")

    norm = numeric.apply(lambda col: col / col.min(), axis=0)
    norm_df = pd.concat([time, norm], axis=1)
    norm_df.columns = df.columns
    return norm_df

def compute_interval_metrics(time, data, start, end, delta_t):
    """Calculates AUC and Amplitude for a single interval."""
    mask = (time >= start) & (time <= end + delta_t)
    t_seg = time[mask].reset_index(drop=True)
    data_seg = data[mask].reset_index(drop=True)
    y_seg = data_seg.values
    
    y_min_vals = data_seg.min().values
    
    auc_recs, trapezoids = [], {c: [] for c in data.columns}

    for i in range(len(t_seg) - 1):
        t1, t2 = t_seg.iloc[i], t_seg.iloc[i + 1]
        row = {'Time Start': t1, 'Time End': t2}
        for j, col in enumerate(data.columns):
            y1, y2 = y_seg[i, j], y_seg[i + 1, j]
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
    """Processes all intervals for the given dataframe."""
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
        if a >= b: continue
        st.info(f"📐 Processing interval: {a}–{b}")
        auc_df, auc_sum, amp_vals, meta = compute_interval_metrics(time, data, a, b, dt)
        tag = f"{int(a)}-{int(b)}"
        auc_tables[tag] = auc_df
        auc_sum.name = tag
        amp_vals.name = tag
        auc_sums.append(auc_sum)
        amp_sums.append(amp_vals)
        meta_rows.append(pd.Series(meta, name=tag))

    if not auc_sums:
        return {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    return (
        auc_tables, pd.concat(auc_sums, axis=1).T,
        pd.concat(amp_sums, axis=1).T, pd.DataFrame(meta_rows)
    )

# --- Streamlit User Interface (This is what the user sees and interacts with) ---

st.set_page_config(layout="wide")
st.title("🔬 Advanced Time-Series Analysis App")

# --- Sidebar for All User Controls ---
with st.sidebar:
    st.header("⚙️ Analysis Controls")
    
    # 1. Mode Selection
    analysis_mode = st.radio(
        "Select Analysis Mode",
        ('Single File Analysis', 'Ratio Analysis'),
        help="**Single File:** Performs analysis on one dataset. **Ratio:** Calculates the ratio of two datasets, then performs analysis."
    )

    # 2. Conditional File Uploader
    if analysis_mode == 'Single File Analysis':
        uploaded_file = st.file_uploader("📂 Upload your Excel file (with one sheet)", type=["xlsx"])
    else:
        uploaded_file = st.file_uploader("📂 Upload your Excel file (must have two sheets)", type=["xlsx"])

    # 3. Text Input for Cuts
    cuts_input = st.text_input("✂️ Enter Time Cuts (comma-separated)", "1230")

    # 4. Universal Normalization Toggle
    apply_norm = st.toggle("Apply Normalization", value=False, help="If ON, normalization will be applied before the main analysis.")

# --- Main Page for Processing and Results ---
if uploaded_file is not None:
    try:
        # --- Data Loading and Preparation ---
        base_df = None
        original_df_single = None # To hold original for single mode output
        sheet1_df, sheet2_df = None, None # To hold originals for ratio mode output
        
        if analysis_mode == 'Single File Analysis':
            original_df_single = pd.read_excel(uploaded_file, engine='openpyxl')
            base_df = original_df_single
            st.success("✅ Single file loaded successfully.")
        
        elif analysis_mode == 'Ratio Analysis':
            xls = pd.ExcelFile(uploaded_file)
            if len(xls.sheet_names) < 2:
                st.error("Error: The uploaded Excel file must contain at least two sheets for Ratio Analysis.")
                st.stop()
            sheet1_df = pd.read_excel(xls, sheet_name=0)
            sheet2_df = pd.read_excel(xls, sheet_name=1)
            time_col = sheet1_df.iloc[:, 0]
            data1 = sheet1_df.iloc[:, 1:]
            data2 = sheet2_df.iloc[:, 1:]
            data2.columns = data1.columns
            ratio_data = data1 / data2
            base_df = pd.concat([time_col, ratio_data], axis=1)
            st.success("✅ Two sheets loaded and ratio calculated.")

        # --- Conditional Normalization ---
        data_label = ""
        if apply_norm:
            analysis_df = normalize_data(base_df)
            data_label = "Normalized " + ("Data" if analysis_mode == 'Single File Analysis' else "Ratio Data")
        else:
            analysis_df = base_df
            data_label = "Original Data" if analysis_mode == 'Single File Analysis' else "Ratio Data"
        
        st.header(f"Data for Analysis: {data_label}")
        st.dataframe(analysis_df.head())
        
        # --- Run Core Analysis ---
        cuts = [int(c.strip()) for c in cuts_input.split(',')]
        auc_tbls, auc_sum_df, amp_df, meta_df = analyse_intervals(analysis_df, cuts)
        st.success("✅ Analysis Complete!")

        # --- Prepare Excel file for download ---
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            sh_name = 'Processed_Data'
            r = 0 # Row counter

            # Write initial data based on mode
            if analysis_mode == 'Single File Analysis':
                ws = writer.book.add_worksheet(sh_name)
                writer.sheets[sh_name] = ws
                ws.write(r, 0, "Original Data"); r += 1
                original_df_single.to_excel(writer, sheet_name=sh_name, startrow=r, index=False)
                r += len(original_df_single) + 3
                if apply_norm:
                    ws.write(r, 0, "Normalized Data"); r += 1
                    analysis_df.to_excel(writer, sheet_name=sh_name, startrow=r, index=False)
                    r += len(analysis_df) + 3
            else: # Ratio Analysis
                # Write a separate sheet for the raw input data
                ws_raw = writer.book.add_worksheet('Raw_Data')
                ws_raw.write(0, 0, "Raw Data from Sheet 1"); sheet1_df.to_excel(writer, sheet_name='Raw_Data', startrow=1, index=False)
                ws_raw.write(len(sheet1_df) + 3, 0, "Raw Data from Sheet 2"); sheet2_df.to_excel(writer, sheet_name='Raw_Data', startrow=len(sheet1_df) + 4, index=False)
                # Write the processed data to its own sheet
                ws = writer.book.add_worksheet(sh_name)
                writer.sheets[sh_name] = ws
                ws.write(r, 0, data_label); r += 1 # Dynamic label (Ratio Data or Normalized Ratio Data)
                analysis_df.to_excel(writer, sheet_name=sh_name, startrow=r, index=False)
                r += len(analysis_df) + 3

            # --- Write Detailed Per-Interval Blocks ---
            for i, (tag, auc_df_interval) in enumerate(auc_tbls.items(), 1):
                ws.write(r, 0, f"AUC Data - Interval {i} ({tag})"); r += 1; auc_df_interval.drop(columns=['Time End'], errors='ignore').rename(columns={'Time Start': 'Time'}).to_excel(writer, sheet_name=sh_name, startrow=r, index=False); r += len(auc_df_interval) + 2
                ws.write(r, 0, f"AUC Sums - Interval {i}"); r += 1; auc_sum_df.loc[tag:tag].to_excel(writer, sheet_name=sh_name, startrow=r, index=True); r += 3
                ws.write(r, 0, f"AUC Average - Interval {i}"); r+=1; ws.write(r, 0, meta_df.loc[tag, 'Average_AUC']); r += 2
                ws.write(r, 0, f"AUC per Minute - Interval {i}"); r+=1; ws.write(r, 0, meta_df.loc[tag, 'Average_AUC_per_min']); r += 2
                ws.write(r, 0, f"Amplitude - Interval {i}"); r += 1; amp_df.loc[tag:tag].to_excel(writer, sheet_name=sh_name, startrow=r, index=True); r += 3
                ws.write(r, 0, f"Average Amplitude - Interval {i}"); r+=1; ws.write(r, 0, meta_df.loc[tag, 'Average_Amplitude']); r += 2
                ws.write(r, 0, f"Avg Amplitude per Minute - Interval {i}"); r+=1; ws.write(r, 0, meta_df.loc[tag, 'Avg_Amplitude_per_min']); r += 3
            
            # --- Write Final Summary Tables ---
            r += 2
            ws.write(r, 0, "Summary of AUC Sums (All Intervals)"); r += 1; auc_sum_df.to_excel(writer, sheet_name=sh_name, startrow=r, index=True); r += len(auc_sum_df) + 3
            ws.write(r, 0, "Summary of Amplitudes (All Intervals)"); r += 1; amp_df.to_excel(writer, sheet_name=sh_name, startrow=r, index=True); r += len(amp_df) + 3
            ws.write(r, 0, "Summary of Average AUC per Minute (All Intervals)"); r += 1; meta_df[['Average_AUC_per_min']].to_excel(writer, sheet_name=sh_name, startrow=r, index=True); r += len(meta_df) + 3
            ws.write(r, 0, "Summary of Average Amplitude per Minute (All Intervals)"); r += 1; meta_df[['Avg_Amplitude_per_min']].to_excel(writer, sheet_name=sh_name, startrow=r, index=True)

        # --- Download Button ---
        st.header("📥 Download Your Results")
        st.download_button(
            label="Download Analysis Excel File",
            data=output.getvalue(),
            file_name="complete_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.warning("Please ensure your file format is correct and time cuts are valid numbers. For Ratio mode, the Excel file must contain two sheets.")

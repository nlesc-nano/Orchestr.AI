import streamlit as st
from io import BytesIO
import pandas as pd
import os
import time
import argparse
import sys

from extract_metrics import extract_metrics_data
from utils import validate_columns, get_sample_schnet_csv, get_sample_nequip_csv
from plot_utils import custom_plot_schnet, custom_plot_nequip, custom_plotly_schnet, custom_plotly_nequip

# --- Parse Command-Line Arguments ---
def parse_args():
    parser = argparse.ArgumentParser(description="Metrics Extraction & Plotting Dashboard with CLI and GUI support.")
    subparsers = parser.add_subparsers(dest="command")

    # Subparser for extract_metrics
    extract_parser = subparsers.add_parser("extract_metrics", help="Extract metrics from a TensorBoard event file.")
    extract_parser.add_argument("-p", "--path", required=True, help="Path to the TensorBoard event file.")
    extract_parser.add_argument("-o", "--output", default="metrics_output.csv", help="Output CSV filename (default: metrics_output.csv).")

    # Subparser for plot
    plot_parser = subparsers.add_parser("plot", help="Plot metrics from a CSV file.")
    plot_parser.add_argument("--platform", required=True, choices=["schnet", "nequip"], help="Platform to plot (schnet or nequip).")
    plot_parser.add_argument("--file", required=True, help="Path to the metrics CSV file.")
    plot_parser.add_argument("--cols", type=int, default=2, help="Number of subplot columns (default: 2).")
    plot_parser.add_argument("--out", default="plot_output.png", help="Output plot filename (default: plot_output.png).")

    return parser.parse_args()

# --- CLI Mode: Extract Metrics ---
def run_extract_metrics(args):
    try:
        metrics_df = extract_metrics_data(args.path)
        if metrics_df.empty:
            print("No metrics data extracted from the event file.")
        else:
            metrics_df.to_csv(args.output, index=False)
            print(f"Metrics extracted successfully and saved to {args.output}")
    except Exception as e:
        print(f"Error: {e}")
    sys.exit(0)

# --- CLI Mode: Plot ---
def run_plot(args):
    try:
        df = pd.read_csv(args.file)
        
        # Validate columns based on platform
        platform = args.platform.lower()
        if platform == "schnet":
            required_cols = ['Step', 'Metric', 'Value']
        else:  # nequip
            required_cols = ['epoch', 'training_loss', 'validation_loss',
                             'training_f_mae', 'validation_f_mae',
                             'training_e_mae', 'validation_e_mae']
        
        is_valid, error_message = validate_columns(df, required_cols, platform.capitalize())
        if not is_valid:
            print(f"Error: {error_message}")
            sys.exit(1)

        # Default plot settings for CLI
        metrics_to_plot = ["Loss", "Energy MAE", "Forces MAE"] if platform == "schnet" else ["Total Loss", "F MAE", "E MAE"]
        title_inputs = {metric: f"{platform.capitalize()} - {metric}" for metric in metrics_to_plot}
        train_color = "#1f77b4"
        val_color = "#ff7f0e"
        log_scale = (platform == "schnet")
        grid = True
        fig_width = 8
        fig_height = 4
        main_title = f"{platform.capitalize()} Performance"

        # Generate plot
        fig = None
        n_rows = 0
        if platform == "schnet":
            fig, n_rows = custom_plot_schnet(df, args.cols, metrics_to_plot, train_color, val_color, log_scale, grid, fig_width, fig_height, title_inputs, main_title)
        else:
            fig, n_rows = custom_plot_nequip(df, args.cols, metrics_to_plot, train_color, val_color, log_scale, grid, fig_width, fig_height, title_inputs, main_title)

        if fig:
            format_map = {"PNG": "png", "JPG": "jpg", "PDF": "pdf", "SVG": "svg"}
            ext = args.out.split('.')[-1].lower()
            if ext not in format_map.values():
                print(f"Error: Unsupported file extension in {args.out}. Supported formats: PNG, JPG, PDF, SVG.")
                sys.exit(1)
            fig.savefig(args.out, format=ext, bbox_inches="tight", dpi=300 if ext in ["png", "jpg"] else None)
            print(f"Plot generated successfully and saved to {args.out}")
        else:
            print("Error: Plot generation failed unexpectedly.")
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    sys.exit(0)

# --- Check for CLI Arguments ---
args = parse_args()
if args.command == "extract_metrics":
    run_extract_metrics(args)
elif args.command == "plot":
    run_plot(args)

# --- If no CLI arguments, proceed with GUI ---
# Streamlit App Configuration
st.set_page_config(page_title="Metrics Extraction & Plotting", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Styling (Including Sliders)
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 5px;}
    .stButton>button:hover {background-color: #45a049;}
    .stSelectbox>label {font-weight: bold;}
    
    /* Custom Slider Styling */
    .stSlider [type="range"] {
        -webkit-appearance: none;
        background: #ddd; /* Track color */
        height: 8px; /* Track height */
        border-radius: 5px;
    }
    .stSlider [type="range"]::-webkit-slider-thumb {
        -webkit-appearance: none;
        background: #4CAF50; /* Thumb color */
        height: 20px; /* Thumb height */
        width: 20px; /* Thumb width */
        border-radius: 50%; /* Circular thumb */
        cursor: pointer;
        box-shadow: 0 0 5px rgba(0, 0, 0, 0.2); /* Subtle shadow */
    }
    .stSlider [type="range"]::-webkit-slider-thumb:hover {
        background: #45a049; /* Darker on hover */
    }
    .stSlider [type="range"]::-moz-range-thumb {
        background: #4CAF50; /* Thumb color for Firefox */
        height: 20px;
        width: 20px;
        border-radius: 50%;
        cursor: pointer;
        box-shadow: 0 0 5px rgba(0, 0, 0, 0.2);
    }
    .stSlider [type="range"]::-moz-range-thumb:hover {
        background: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if 'metrics_df' not in st.session_state:
    st.session_state.metrics_df = None

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
st.sidebar.markdown("Explore the app:")
page = st.sidebar.radio("", ["🏠 Home", "📊 Extract Metrics", "📈 Plot Results"], label_visibility="collapsed")

# Add Resources section with GitHub link
st.sidebar.markdown("### Resources")
st.sidebar.markdown("[Visit NeuralForceField for Training Machine Learning Force Fields](https://github.com/learningmatter-mit/NeuralForceField)")

# Add Help section
st.sidebar.markdown("### Help")
st.sidebar.markdown("""
**How to Use This App:**
1. **Extract Metrics**: Upload a TensorBoard event file to extract metrics and download them as a CSV.
2. **Plot Results**: Upload a CSV file with metrics to visualize training/validation data.
   - **SchNet CSV Format**: Requires columns `Step`, `Metric`, `Value`.
   - **Nequip CSV Format**: Requires columns `epoch`, `training_loss`, `validation_loss`, `training_f_mae`, `validation_f_mae`, `training_e_mae`, `validation_e_mae`.
3. Use the "Download Sample SchNet CSV" or "Download Sample NequIP CSV" buttons to get example data.

**About SchNet and NequIP:**
- **SchNet**: A deep learning model for predicting molecular energies and forces, often used in quantum chemistry.
- **Nequip**: A neural equivariant interatomic potential model for molecular dynamics simulations.
""")

# --- Header ---
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Metrics Extraction & Plotting Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Analyze your TensorBoard data with ease!</p>", unsafe_allow_html=True)

# --- Home Page ---
if page == "🏠 Home":
    st.subheader("Welcome!")
    st.write("""
    This dashboard helps you:
    - **Extract Metrics**: Upload a TensorBoard event file, preview metrics, and download them as CSV.
    - **Plot Results**: Upload a CSV, customize plots, and download visualizations.
    
    Use the sidebar to get started!
    """)
    st.info("Tip: All processing happens in your browser—no files are saved on the server.")

# --- Extract Metrics Page ---
elif page == "📊 Extract Metrics":
    st.subheader("Extract Metrics from TensorBoard Event File")
    
    with st.expander("Upload & Options", expanded=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded_file = st.file_uploader(
                "Upload TensorBoard Event File", 
                type=None, 
                help="Upload a TensorBoard event file (e.g., events.out.tfevents.xxx)."
            )
        with col2:
            output_filename = st.text_input(
                "Output CSV Filename", 
                "metrics_output.csv", 
                help="Specify the name for your downloaded CSV file."
            )
            preview = st.checkbox(
                "Preview Metrics", 
                value=True, 
                help="Check to see a table of extracted metrics before downloading."
            )

    if uploaded_file and st.button("Extract Metrics", key="extract"):
        progress_bar = st.progress(0)
        with st.spinner("Extracting metrics..."):
            try:
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                progress_bar.progress(25)
                metrics_df = extract_metrics_data(temp_path)
                st.session_state.metrics_df = metrics_df
                progress_bar.progress(75)
                
                if metrics_df.empty:
                    st.warning("No metrics data extracted from the event file.")
                else:
                    if preview:
                        st.write("### Preview of Extracted Metrics")
                        st.dataframe(metrics_df, use_container_width=True)
                    
                    csv_buffer = BytesIO()
                    metrics_df.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)
                    
                    st.success("✅ Metrics extracted successfully!")
                    st.download_button(
                        label="Download CSV",
                        data=csv_buffer,
                        file_name=output_filename,
                        mime="text/csv",
                        key="download_csv"
                    )
                
                os.remove(temp_path)
                progress_bar.progress(100)
                time.sleep(0.5)  # Brief delay to show completion
                progress_bar.empty()
            except Exception as e:
                st.error(f"❌ Error: {e}")
                progress_bar.empty()

    if st.button("Clear", key="clear_extract"):
        st.session_state.metrics_df = None
        st.rerun()

# --- Plot Results Page ---
elif page == "📈 Plot Results":
    st.subheader("Plot Results from CSV")
    
    with st.expander("Upload & Plot Settings", expanded=True):
        col1, col2 = st.columns([1, 2])
        with col1:
            uploaded_csv = st.file_uploader(
                "Upload CSV File", 
                type=["csv"], 
                help="Upload a CSV with metrics (SchNet: Step, Metric, Value; NequIP: epoch, training_loss, etc.)."
            )
            platform = st.selectbox(
                "Select Platform", 
                ["SchNet", "Nequip"], 
                help="Choose the platform your CSV corresponds to (SchNet for molecular energies, NequIP for equivariant potentials)."
            )
            n_cols = st.slider(
                "Columns in Plot", 
                1, 3, 2, 
                help="Number of subplot columns in the generated plot (e.g., 2 means 2 subplots per row)."
            )
            output_plot_name = st.text_input(
                "Output Plot Filename", 
                "plot_output.png", 
                help="Specify the name for your downloaded plot file (only for Matplotlib plots)."
            )
            export_format = st.selectbox(
                "Export Format", 
                ["PNG", "JPG", "PDF", "SVG"], 
                help="Choose the format for downloading the plot (only available for Matplotlib plots)."
            )
            plot_type = st.selectbox(
                "Plot Type", 
                ["Static (Matplotlib)", "Interactive (Plotly)"], 
                help="Choose between static (Matplotlib, can be saved) or interactive (Plotly, display only) plots."
            )
            st.download_button(
                label=f"Download Sample {platform} CSV",
                data=get_sample_schnet_csv() if platform == "SchNet" else get_sample_nequip_csv(),
                file_name=f"sample_{platform.lower()}_metrics.csv",
                mime="text/csv",
                key="sample_csv"
            )
        
        with col2:
            st.write("### Plot Customization")
            main_title = st.text_input(
                "Main Plot Title", 
                f"{platform} Performance", 
                help="Set an overarching title for the entire figure, displayed above all subplots."
            )
            if platform == "SchNet":
                metrics_to_plot = st.multiselect(
                    "Metrics to Plot", 
                    ["Loss", "Energy MAE", "Forces MAE"], 
                    default=["Loss", "Energy MAE", "Forces MAE"],
                    help="Choose which metrics to include in the plot (e.g., Loss, Energy MAE, Forces MAE for SchNet)."
                )
            else:  # Nequip
                metrics_to_plot = st.multiselect(
                    "Metrics to Plot", 
                    ["Total Loss", "F MAE", "E MAE"], 
                    default=["Total Loss", "F MAE", "E MAE"],
                    help="Choose which metrics to include in the plot (e.g., Total Loss, F MAE, E MAE for NequIP)."
                )
            title_inputs = {}
            for metric in metrics_to_plot:
                default_title = f"{platform} - {metric}"
                title_inputs[metric] = st.text_input(
                    f"Title for {metric}", 
                    default_title, 
                    key=f"title_{metric}",
                    help=f"Enter a custom title for the {metric} subplot (default: {default_title})."
                )
            train_color = st.color_picker(
                "Train Line Color", 
                "#1f77b4", 
                help="Pick a color for the training data lines (e.g., blue for training loss)."
            )
            val_color = st.color_picker(
                "Validation Line Color", 
                "#ff7f0e", 
                help="Pick a color for the validation data lines (e.g., orange for validation loss)."
            )
            log_scale = st.checkbox(
                "Log Scale (Y-Axis)", 
                value=(platform == "SchNet"), 
                help="Apply a logarithmic scale to the Y-axis (recommended for SchNet to better visualize small values)."
            )
            grid = st.checkbox(
                "Show Grid", 
                value=True, 
                help="Toggle grid lines on the plot for better readability of data points."
            )
            fig_width = st.slider(
                "Figure Width", 
                5, 15, 8, 
                help="Set the width of the plot in inches (affects the overall plot size)."
            )
            fig_height = st.slider(
                "Figure Height", 
                3, 10, 4, 
                help="Set the height of each subplot row in inches (affects the height of each metric plot)."
            )

    if uploaded_csv and st.button("Generate Plot", key="plot"):
        progress_bar = st.progress(0)
        with st.spinner("Generating plot..."):
            try:
                df = pd.read_csv(uploaded_csv)
                progress_bar.progress(25)
                
                # Validate columns based on platform
                if platform == "SchNet":
                    required_cols = ['Step', 'Metric', 'Value']
                else:  # Nequip
                    required_cols = ['epoch', 'training_loss', 'validation_loss',
                                     'training_f_mae', 'validation_f_mae',
                                     'training_e_mae', 'validation_e_mae']
                
                is_valid, error_message = validate_columns(df, required_cols, platform)
                if not is_valid:
                    st.error(f"❌ {error_message}")
                    progress_bar.empty()
                else:
                    progress_bar.progress(50)
                    # Plotting logic
                    if plot_type == "Static (Matplotlib)":
                        fig = None
                        n_rows = 0
                        if platform == "SchNet":
                            fig, n_rows = custom_plot_schnet(df, n_cols, metrics_to_plot, train_color, val_color, log_scale, grid, fig_width, fig_height, title_inputs, main_title)
                        else:
                            fig, n_rows = custom_plot_nequip(df, n_cols, metrics_to_plot, train_color, val_color, log_scale, grid, fig_width, fig_height, title_inputs, main_title)

                        st.pyplot(fig)

                        # Export logic for Matplotlib only
                        progress_bar.progress(75)
                        if fig:
                            try:
                                img_buffer = BytesIO()
                                format_map = {"PNG": "png", "JPG": "jpg", "PDF": "pdf", "SVG": "svg"}
                                fig.savefig(img_buffer, format=format_map[export_format], bbox_inches="tight", dpi=300 if export_format in ["PNG", "JPG"] else None)
                                img_buffer.seek(0)
                                st.success(f"✅ {platform} plot generated successfully!")
                                st.download_button(
                                    label=f"Download Plot as {export_format}",
                                    data=img_buffer,
                                    file_name=output_plot_name.replace(".png", f".{format_map[export_format].lower()}"),
                                    mime=f"image/{format_map[export_format]}" if export_format in ["PNG", "JPG"] else f"application/{format_map[export_format]}",
                                    key="download_plot"
                                )
                            except Exception as e:
                                st.error(f"❌ Error generating downloadable image: {e}")
                        else:
                            st.error("❌ Plot generation failed unexpectedly.")

                    else:  # Interactive (Plotly) - Display only
                        fig = None
                        n_rows = 0
                        if platform == "SchNet":
                            fig, n_rows = custom_plotly_schnet(df, n_cols, metrics_to_plot, train_color, val_color, log_scale, grid, fig_width, fig_height, title_inputs, main_title)
                        else:
                            fig, n_rows = custom_plotly_nequip(df, n_cols, metrics_to_plot, train_color, val_color, log_scale, grid, fig_width, fig_height, title_inputs, main_title)

                        st.plotly_chart(fig, use_container_width=True)
                        st.success(f"✅ {platform} plot generated successfully! (Plotly plots are display-only. Switch to Matplotlib to download.)")

                    progress_bar.progress(100)
                    time.sleep(0.5)
                    progress_bar.empty()
            except Exception as e:
                st.error(f"❌ Error: {e}")
                progress_bar.empty()

    if st.button("Clear", key="clear_plot"):
        st.session_state.metrics_df = None
        st.rerun()

if __name__ == "__main__":
    pass
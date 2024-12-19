# Import libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# matplotlib dark mode
plt.style.use('dark_background')

# styling
plt.rcParams.update({
    "font.size": 10
})

# page title
st.set_page_config(page_title="Exploratory Data Analysis", page_icon='📊')
st.title("📊 Cassini Dust Astronomy Model")

# App Description
with st.expander('About this app'):
    st.markdown('**What can this app do?**')
    st.info('This app shows the data analysis and ML models for Cassini calibration tools.')
    st.markdown('**How to use the app?**')
    st.warning('To engage with the app:\n1. Select amplitude of your interest in the drop-down selection box.\n2. Select the target.\n3. Specify the number of plots and configure each plot.\n4. The result will generate side-by-side plots.')

# Sidebar for filter tools
st.sidebar.header('Filter Tools')

# Load the data
cal_df = pd.read_pickle("data/level1/CDA__CAT_IID_cal_data.pkl")

# Create a dropdown menu for target selector in the sidebar
tar_list = cal_df.TAR.unique()
tar_selection = st.sidebar.multiselect("Select target", tar_list)

tar_filter = cal_df[cal_df.TAR.isin(tar_selection)]
with st.expander("Filtered by target"):
    st.write(tar_filter)

# Sidebar for specifying the number of plots
st.sidebar.header('Plot Configuration')
num_plots = st.sidebar.number_input("Number of plots", min_value=1, max_value=6, value=2)
num_histograms = st.sidebar.number_input("Number of histograms", min_value=1, max_value=6, value=2)

# Generate plots based on user input
plots = []
for i in range(num_plots):
    with st.sidebar.expander(f'Configure Plot {i+1}', expanded=(i == 0)):
        # Plot-specific configuration
        x_element = st.selectbox(f"Select X-axis element for Plot {i+1}", options=tar_filter.columns, key=f"x_{i}")
        y_element = st.selectbox(f"Select Y-axis element for Plot {i+1}", options=tar_filter.columns, key=f"y_{i}")
        size = st.slider(f"Select size of points for Plot {i+1}", min_value=2, max_value=100, value=10, key=f"size_{i}")
        alpha_scatter = st.slider(f"Select transparency of points for Plot {i+1}", min_value=0.0, max_value=1.0, value=0.35, key=f"alpha_scatter{i}")
        color = st.color_picker(f"Select color of points for Plot {i+1}", "#00f900", key=f"color_{i}")
        log_x = st.checkbox(f"Log scale for X-axis for Plot {i+1}", key=f"log_x_{i}")
        log_y = st.checkbox(f"Log scale for Y-axis for Plot {i+1}", key=f"log_y_{i}")

        # Create the plot
        fig, ax = plt.subplots()
        ax.scatter(tar_filter[x_element], tar_filter[y_element], s=size, alpha=alpha_scatter, color=color)
        plt.grid(linestyle="dashed", alpha=0.2)
        ax.set_xlabel(x_element)
        ax.set_ylabel(y_element)
        ax.set_title(f'Scatter plot of {x_element} vs {y_element}')

        # Apply logarithmic scaling if selected
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")

        plots.append(fig)

# display histograms based on user input
histograms = []
for i in range(num_histograms):
    with st.sidebar.expander(f"Configure Histogram {i+1}", expanded= (i==0)):
        element = st.selectbox(f"Search element for histogram {i+1}", options=tar_filter.columns, key=f"hist_element_{i}")

        # min_bin_val = st.slider(f"Min value for histogram bin {i+1}", min_value=element.min(), max_value=element.max(), value=0.0, key=f"hist_min_bin_{i}")
        # max_bin_val = st.slider(f"Max value for histogram bin {i+1}", min_value=element.min(), max_value=element.max(), value=100.0, key=f"hist_max_bin_{i}")

        tar_filter[element] = tar_filter[element].dropna()
        # Extract the numerical values of the selected element
        if np.issubdtype(tar_filter[element].dtype, np.number):  # Check if the column is numeric
            natural_min = tar_filter[element].min()
            natural_max = tar_filter[element].max()
            print(natural_min)
        else:
            st.warning(f"The selected element '{element}' is not numeric. Defaulting to a range of 0-100.")
            natural_min = 0.0
            natural_max = 100.0

        # Sliders for bin values
        min_bin_val = st.slider(
            f"Min value for histogram bin {i+1}",
            min_value=float(natural_min),
            max_value=float(natural_max),
            value=float(natural_min),
            key=f"hist_min_bin_{i}"
        )
        max_bin_val = st.slider(
            f"Max value for histogram bin {i+1}",
            min_value=float(natural_min),
            max_value=float(natural_max),
            value=float(natural_max),
            key=f"hist_max_bin_{i}"
        )


        alpha_histogram = st.slider(f"Select transparency of histogram {i+1}", min_value=0.0, max_value=1.0, value=0.35, key=f"alpha_histogram{i}")
        step = st.slider(f"Step size for Histogram {i+1}", min_value=float(-1000), max_value=float(1000), value=1.0, key=f"hist_step_{i}")
        color = st.color_picker(f"Select color for Histogram {i+1}", "#0073e6", key=f"hist_color_{i}")
        log_y = st.checkbox(f"Log scale for Histogram {i+1}", key=f"hist_log_{i}")

        # Create the histogram
        fig, ax = plt.subplots(layout="constrained")
        ax.hist(tar_filter[element],
                # bins=np.arange(min_bin_val, max_bin_val, step),
                color=color,
                alpha=alpha_histogram,
                edgecolor='white')

        ax.set_xlabel(element)
        ax.set_ylabel("Frequency")
        ax.set_title(f'Histogram of {element}')

        # Apply logarithmic scaling if selected
        if log_y:
            ax.set_yscale("log")

        histograms.append(fig)


# Display plots side by side (max two per row)
with st.expander("Plots"):
    all_plots = plots + histograms
    for i in range(0, len(all_plots), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(all_plots):
                with col:
                    st.pyplot(all_plots[i + j])

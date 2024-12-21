# Import libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

import plotly.graph_objects as go

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


st.sidebar.header('Filter Tools')
cal_df = pd.read_pickle("data/level1/CDA__CAT_IID_cal_data.pkl")

# Create a dropdown menu for target selector in the sidebar
tar_list = cal_df.TAR.unique()
tar_selection = st.sidebar.multiselect("Select target", tar_list, ["CAT"])

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
        alpha_histogram = st.slider(f"Select transparency of histogram {i+1}",
                                                min_value=0.0,
                                                max_value=1.0, value=0.35,
                                                key=f"alpha_histogram{i}")

        # Extract the numerical values of the selected element
        if np.issubdtype(tar_filter[element].dtype, np.number):  # Check if the column is numeric
            tar_filter[element] = tar_filter[element].dropna()
            if not tar_filter[element].empty:
                natural_min = tar_filter[element].min()
                natural_max = tar_filter[element].max()

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

            # work on step size count to make it more intuitive maybe standard deviation
            step = st.slider(f"Step size for Histogram {i+1}", min_value=min_bin_val, max_value=((max_bin_val + min_bin_val) / 2), value=1.0, key=f"hist_step_{i}")
            color = st.color_picker(f"Select color for Histogram {i+1}", "#0073e6", key=f"hist_color_{i}")
            log_y = st.checkbox(f"Log scale for Histogram {i+1}", key=f"hist_log_{i}")

            # Create the histogram
            fig, ax = plt.subplots(layout="constrained")
            ax.hist(tar_filter[element],
                    bins=np.arange(min_bin_val, max_bin_val, step),
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

        else:
            # Group by unique values and count occurrences
            value_counts = tar_filter[element].value_counts()

            # Create the bar plot
            fig, ax = plt.subplots(layout="constrained")
            value_counts.plot.bar(color=color, alpha=alpha_histogram, ax=ax)
            ax.set_xlabel("Categories")
            ax.set_ylabel("Frequency")
            ax.set_title(f'Bar Plot of {element}')

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


# ml model splitting

# Sidebar configuration
st.sidebar.title("Machine Learning Integration")

# Multiselect for features
feature_options = list(tar_filter.columns)
selected_features = st.sidebar.multiselect(
    "Select Features for Prediction",
    options=feature_options,
    default=feature_options[:2]  # Default to the first two columns
)

# Single select for target
selected_target = st.sidebar.selectbox(
    "Select Target Variable",
    options=feature_options,
    index=len(feature_options) - 1  # Default to the last column
)

# Test size slider
test_size = st.sidebar.slider("Test Size (for Train-Test Split)", min_value=0.1, max_value=0.5, step=0.05, value=0.2)

# Button to split the data
if st.sidebar.button("Split Data"):
    # Ensure features and target are selected properly
    if selected_features and selected_target not in selected_features:
        # Extract the data
        X = tar_filter[selected_features].to_numpy()  # Selected features
        y = tar_filter[selected_target].to_numpy()  # Target variable

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X,  # Features
            y,  # Target
            test_size=test_size,
            random_state=42
        )

        # Visualizing histograms for each feature
        num_selected_features = len(selected_features)
        fig, axes = plt.subplots(1, num_selected_features + 1, figsize=(15, 5), layout="constrained")

        for i, ax in enumerate(axes[:-1]):  # Loop over selected features
            bins = np.linspace(X_train[:, i].min(), X_train[:, i].max(), 20)
            ax.hist(X_train[:, i],
                    bins=bins,
                    alpha=0.5,
                    color="blue",
                    label="Train")

            ax.hist(X_test[:, i],
                    bins=bins,
                    alpha=0.5,
                    color="green",
                    label="Test")
            ax.set_yscale("log")
            ax.set_xlabel(selected_features[i])
            ax.legend(loc="upper right")

        # Target variable
        bins = np.linspace(y_train.min(), y_train.max(), 20)

        axes[-1].hist(y_train,
                    bins=bins,
                    alpha=0.5,
                    color="blue",
                    label="Train")

        axes[-1].hist(y_test,
                     bins=bins,
                     alpha=0.5,
                     color="green",
                     label="Test")

        axes[-1].set_yscale("log")
        axes[-1].set_xlabel(selected_target)
        axes[-1].legend(loc="upper right")

        # Display plots side by side (max two per row)
        with st.expander("Split plot"):
            st.pyplot(fig)

        ## Metrics
        # Standardization of data
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_train_scaled = X_scaler.fit_transform(X_train)
        X_test_scaled = X_scaler.transform(X_test)

        y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
        y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))

        # Neural network model with 2 hidden layers
        inputs = tf.keras.Input(shape=(X_train_scaled.shape[1],), name="tar_filter")
        layer = tf.keras.layers.Dense(units=64, activation="relu")(inputs)
        layer = tf.keras.layers.Dense(units=32, activation="relu")(layer)
        output = tf.keras.layers.Dense(units=1, name="psu_vel")(layer)

        # Compile the model
        model = tf.keras.Model(inputs=inputs, outputs=output)
        model.compile(optimizer="adam", loss="mean_squared_error",
                      metrics=[tf.keras.metrics.RootMeanSquaredError(), "mean_absolute_error"])

        # Train the model
        history = model.fit(X_train_scaled, y_train_scaled, batch_size=32, epochs=150,
                            validation_split=0.25, verbose=0)

        # Plot the training history
        st.subheader("Model Loss and Validation Loss")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history.history["loss"], label="Training Loss")
        ax.plot(history.history["val_loss"], label="Validation Loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()

        with st.expander("Loss & Validation error"):
            st.pyplot(fig)

        # Evaluate the model on test data
        st.subheader("Model Evaluation on Test Data")
        test_loss, rmse_scaled, mae_scaled = model.evaluate(X_test_scaled, y_test_scaled)
        st.write(f"Loss scaled: {test_loss:.4f}")
        st.write(f"RMSE scaled: {rmse_scaled:.4f} \
                & RMSE on target variabale : \
                {y_scaler.inverse_transform([[rmse_scaled]])[0][0]}")
        st.write(f"Test MAE: {mae_scaled:.4f} \
                & MAE on target variable: {y_scaler.inverse_transform([[mae_scaled]])[0][0]}")

        # Rescale the data to fit the 3D mesh for a hyperplane
        st.sidebar.subheader("3D Mesh Grid Configuration")

        # Define default min and max ranges based on selected features
        default_min_values = [1, 0.5]
        default_max_values = [60, 2.6]

        # Allow users to dynamically adjust the range for mesh grid scaling
        min_values = []
        max_values = []

        if len(selected_features) == 2:
            for i, feature in enumerate(selected_features):
                min_val = st.sidebar.number_input(f"Min value for {feature}", value=default_min_values[i])
                max_val = st.sidebar.number_input(f"Max value for {feature}", value=default_max_values[i])
                min_values.append(min_val)
                max_values.append(max_val)

            # Transform min and max values using the scaler
            min_scaled = X_scaler.transform([min_values])[0]
            max_scaled = X_scaler.transform([max_values])[0]

            # Generate linspaced values in between to fit the grid
            scaled_ranges = [np.linspace(min_scaled[i], max_scaled[i], 100) for i in range(len(selected_features))]

            # Create meshgrid for the selected features
            meshgrid_scaled = np.meshgrid(*scaled_ranges)


            # plotting the z-axis with the help of meshgrid
            feature_one = []
            feature_two = []
            pred_val = []

            for _f1, _f2 in zip(*meshgrid_scaled):
                stacked_scaled = np.hstack([_f1.reshape(-1,1),
                                            _f2.reshape(-1,1)])

                # rise time and log_fc scalings
                feature_one.append(X_scaler.inverse_transform(stacked_scaled)[:, 0])
                feature_two.append(X_scaler.inverse_transform(stacked_scaled)[:,1])

                # pred. vel. stacked scaling
                temp_results = model.predict(x=stacked_scaled)
                pred_val.append(y_scaler.inverse_transform(temp_results.reshape(1,-1))[0])

            pred_val = np.array(pred_val)

            print(feature_one)
            print("a")
            print(feature_two)
            print("a")
            print(pred_val)

            # # 3D plot

            x_feature, y_feature = selected_features
            tar_filter = tar_filter[tar_filter[x_feature] >= 0.0]
            tar_filter = tar_filter[tar_filter[y_feature] >= 0.0]

            # Create 3D scatter plot with Plotly
            fig = go.Figure()

            # Scatter plot
            fig.add_trace(
                go.Scatter3d(
                    x=tar_filter[x_feature],
                    y=np.log10(tar_filter[y_feature]),
                    z=tar_filter[selected_target],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=tar_filter[selected_target],
                        colorscale="Viridis",
                        opacity=0.8,
                    ),
                    name="Scatter Points",
                )
            )

            # Regression surface
            fig.add_trace(
                go.Surface(
                    x=np.array(feature_one),
                    y=np.array(np.log10(feature_two)),
                    z=pred_val,
                    colorscale="Viridis",
                    opacity=0.7,
                    name="Regression Surface",
                )
            )

            # Customize layout
            fig.update_layout(
                scene=dict(
                    xaxis_title=f"{x_feature}",
                    yaxis_title=f"{y_feature}",
                    zaxis_title=f"{selected_target}",
                ),
                title=f"3D Scatter Plot and Regression Surface for ",
                margin=dict(l=0, r=0, t=30, b=0),
            )

            # Display interactive plot in Streamlit
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.sidebar.warning("Please select at least one feature and ensure the target variable is not among selected features.")
else:
    st.sidebar.info("Select features and a target variable, then click the button to split the data and visualize distributions.")



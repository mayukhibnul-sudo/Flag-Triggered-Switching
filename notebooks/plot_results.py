import marimo

__generated_with = "0.23.13"
app = marimo.App()


@app.cell
def introduction():
    import marimo as marimo

    return (marimo,)


@app.cell
def imports():
    import os
    import sys

    # 1. Dynamically locate the directory where this notebook script lives
    # Marimo populates __file__ when running/editing notebooks
    try:
        notebook_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Fallback if executing in an interactive shell context without __file__
        notebook_dir = os.getcwd()

    # 2. Go up one level to the project root (C:\qec_decoder_switch)
    project_root = os.path.abspath(os.path.join(notebook_dir, ".."))
    src_path = os.path.join(project_root, "src")

    # 3. Inject paths cleanly if they exist
    for path in [project_root, src_path]:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)

    import numpy as np
    from qec_sim.main import evaluate_decoder

    return evaluate_decoder, np


@app.cell
def control_panel(marimo):
    # Dropdown to select the code geometry type matching our new main.py factory
    code_selector = marimo.ui.dropdown(
        options=["Generalized Bicycle (d=5)", "Bivariate Bicycle [[144,12,12]]"],
        value="Bivariate Bicycle [[144,12,12]]",
        label="QLDPC Code Architecture"
    )

    # Simulation sample size adjustment
    shots_slider = marimo.ui.slider(start=50, stop=500, step=50, value=100, label="Simulation Shots")

    # Combine them into a layout object so Marimo renders them on screen
    ui_layout = marimo.vstack([
        marimo.md("### FTS Simulation Controls"),
        code_selector,
        shots_slider
    ])

    # Returning these variables ensures they are drawn and accessible by other cells
    return code_selector, shots_slider


@app.cell
def run_simulation_sweep(code_selector, evaluate_decoder, np, shots_slider):
    # Sweep physical error rates from 1% to 10% across 5 data points
    physical_rates = np.linspace(0.01, 0.10, 5)
    logical_rates = []
    fallback_rates = []

    # Run the circuit-level backend for each noise point
    for p in physical_rates:
        log_e, fallback = evaluate_decoder(
            code_type=code_selector.value, 
            physical_error_rate=p, 
            num_shots=shots_slider.value
        )
        logical_rates.append(log_e)
        fallback_rates.append(fallback * 100) # Convert to %

    return fallback_rates, logical_rates, physical_rates


@app.cell
def plot_results(fallback_rates, logical_rates, physical_rates):
    import matplotlib.pyplot as plt
    import marimo as mo

    # 1. Clear out any background figure buffers
    plt.close('all')

    # 2. Guard rail: assign a fallback warning if data isn't ready
    if len(physical_rates) == 0:
        output_to_display = "No simulation data available yet. Please run the simulation sweep first!"
    else:
        # 3. Build the canvas elements
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Logical vs Physical Performance
        ax1.plot(physical_rates, logical_rates, 'o-', color='tab:blue', label="Logical Error")
        ax1.set_xlabel("Physical Error Rate (p)")
        ax1.set_ylabel("Logical Error Rate")
        ax1.set_title("Decoder Threshold Tracking")
        ax1.grid(True)

        # Plot 2: OSD Workload Transferred (Flag-Triggered Switching efficiency)
        ax2.plot(physical_rates, fallback_rates, 's--', color='tab:red', label="OSD Fallback %")
        ax2.set_xlabel("Physical Error Rate (p)")
        ax2.set_ylabel("OSD Fallback Rate (%)")
        ax2.set_title("Workload Transferred to OSD")
        ax2.grid(True)

        plt.tight_layout()

        # Assign the final figure object to our variable
        output_to_display = fig

    # 4. CRITICAL: Print the output variable block at the root level for Marimo to render it
    output_to_display

    return


if __name__ == "__main__":
    app.run()

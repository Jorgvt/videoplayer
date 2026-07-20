# /// script
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "numpy",
#   "plotly",
# ]
# ///

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

CSV_PATH = Path("profile_results.csv")

def generate_raster_plot():
    # Check if a custom CSV file path was passed on the command line
    csv_file = Path("profile_results.csv")
    for arg in sys.argv[1:]:
        if arg.endswith(".csv"):
            csv_file = Path(arg)
            break
            
    if not csv_file.exists():
        print(f"Error: Could not find {csv_file.resolve()}")
        print("Please run 'uv run profile_triple_player.py' or 'uv run benchmark_dataset.py' first to generate profiling data.")
        sys.exit(1)
        
    print(f"Reading profiling data from {csv_file}...")
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=["SwapTimestamp_sec", "GetImageDuration_ms"]).copy()
    num_frames = len(df)
    print(f"Loaded {num_frames} frames of profiling data.")
    
    # Convert timestamps to relative time in seconds starting from 0.0
    t0_first = df["SwapTimestamp_sec"].iloc[0] - (df["TotalLoopDuration_ms"].iloc[0] / 1000.0)
    df["RelSwapTime_sec"] = df["SwapTimestamp_sec"] - t0_first
    
    # Reconstruct stage milestone timestamps for each frame i
    df["t3"] = df["RelSwapTime_sec"]
    df["t2"] = df["t3"] - (df["FlipDuration_ms"] / 1000.0)
    df["t1"] = df["t2"] - (df["UploadDrawDuration_ms"] / 1000.0)
    df["t0"] = df["t1"] - (df["GetImageDuration_ms"] / 1000.0)
    
    df["InterFrameInterval_numeric"] = pd.to_numeric(df["InterFrameInterval_ms"], errors="coerce")
    
    # Continuous colormap for color coding frame index i
    colors_mpl = cm.turbo(np.linspace(0, 1, num_frames))
    
    # =========================================================================
    # 1. MATPLOTLIB RASTER PLOT WITH VERTICAL EVENT TICKS (PNG Export)
    # =========================================================================
    plt.style.use("dark_background")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 11), sharex=True, gridspec_kw={'height_ratios': [2.2, 1.2, 1.2]})
    fig.patch.set_facecolor("#0b0c10")
    
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor("#14161d")
        ax.grid(True, linestyle="--", alpha=0.25, color="#444444")

    fig.subplots_adjust(left=0.10, right=0.86, top=0.93, bottom=0.07, hspace=0.25)

    # SUBPLOT 1: Vertical Line Event Raster Ticks
    ax1.set_title("Rendering Pipeline Event Raster Plot (Vertical Line Ticks)", fontsize=14, fontweight="bold", pad=10, color="#9d4edd")
    ax1.set_ylabel("Rendering Stage", fontsize=11, fontweight="bold")
    ax1.set_yticks([1, 2, 3, 4])
    ax1.set_yticklabels([
        "1. Fetch Frame (RAM)", 
        "2. Upload & Draw (GPU)", 
        "3. VSync Flip Wait", 
        "4. Swap Complete"
    ])
    ax1.set_ylim(0.5, 4.5)
    
    # Plot vertical line event ticks for every frame on each stage row
    for i in range(num_frames):
        col = colors_mpl[i]
        t0_i = df["t0"].iloc[i]
        t1_i = df["t1"].iloc[i]
        t2_i = df["t2"].iloc[i]
        t3_i = df["t3"].iloc[i]
        
        ax1.vlines(t0_i, 0.65, 1.35, color=col, linewidth=1.2, alpha=0.55, zorder=3)
        ax1.vlines(t1_i, 1.65, 2.35, color=col, linewidth=1.2, alpha=0.55, zorder=3)
        ax1.vlines(t2_i, 2.65, 3.35, color=col, linewidth=1.2, alpha=0.55, zorder=3)
        ax1.vlines(t3_i, 3.65, 4.35, color=col, linewidth=1.2, alpha=0.55, zorder=3)

    # SUBPLOT 2: Inter-Frame Presentation Intervals
    ax2.set_title("Inter-Frame Presentation Interval vs Time (Stimulus Pacing)", fontsize=11, fontweight="bold", pad=6, color="#e0e0e0")
    ax2.set_ylabel("Interval (ms)", fontsize=11, fontweight="bold")
    
    for i in range(num_frames):
        val = df["InterFrameInterval_numeric"].iloc[i]
        if not np.isnan(val):
            t3_i = df["t3"].iloc[i]
            col = colors_mpl[i]
            ax2.scatter([t3_i], [val], color=[col], s=10, alpha=0.6, zorder=3)

    ax2.axhline(4.167, color="#ff4757", linestyle="--", linewidth=1.5, label="Target 240Hz (4.167ms)")
    ax2.legend(loc="upper right", frameon=True, facecolor="#1a1c23", edgecolor="#333333")

    # SUBPLOT 3: GPU Upload / Processing Duration
    ax3.set_title("GPU Upload & Processing Duration vs Time", fontsize=11, fontweight="bold", pad=6, color="#e0e0e0")
    ax3.set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Duration (ms)", fontsize=11, fontweight="bold")
    
    for i in range(num_frames):
        draw_dur = df["UploadDrawDuration_ms"].iloc[i]
        t3_i = df["t3"].iloc[i]
        col = colors_mpl[i]
        ax3.scatter([t3_i], [draw_dur], color=[col], s=10, alpha=0.6, zorder=3)

    # Dedicated Colorbar on far right
    cax = fig.add_axes([0.89, 0.12, 0.018, 0.76])
    sm = plt.cm.ScalarMappable(cmap=cm.turbo, norm=plt.Normalize(vmin=0, vmax=num_frames - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Frame Index", fontsize=11, fontweight="bold", color="#e0e0e0")
    cbar.ax.tick_params(labelsize=10, colors="#e0e0e0")

    output_png = "profile_raster_plot.png"
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    print(f"\nMatplotlib raster plot saved to: {Path(output_png).resolve()}")

    # Check command-line flags for optional Plotly HTML export
    enable_plotly = "--plotly" in sys.argv or "--html" in sys.argv
    
    if enable_plotly:
        try:
            import webbrowser
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            print("\nGenerating optional Plotly Interactive HTML Dashboard...")
            fig_plotly = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=(
                    "Rendering Pipeline Event Raster Plot (Vertical Line Ticks)",
                    "Inter-Frame Presentation Interval vs Time (Stimulus Pacing)",
                    "GPU Upload & Processing Duration vs Time"
                ),
                row_heights=[0.5, 0.25, 0.25]
            )

            stages_info = [
                ("1. Fetch Frame (RAM)", "t0", 1),
                ("2. Upload & Draw (GPU)", "t1", 2),
                ("3. VSync Flip Wait", "t2", 3),
                ("4. Swap Complete", "t3", 4)
            ]
            
            for label, col_time, y_val in stages_info:
                fig_plotly.add_trace(
                    go.Scatter(
                        x=df[col_time],
                        y=[y_val] * num_frames,
                        mode="markers",
                        marker=dict(
                            symbol="line-ns-open",
                            size=12,
                            line=dict(width=1.5),
                            color=df["FrameIndex"],
                            colorscale="Turbo",
                            showscale=True if y_val == 1 else False,
                            colorbar=dict(title="Frame Index", len=0.8, x=1.02) if y_val == 1 else None,
                            opacity=0.75
                        ),
                        text=[f"Frame #{idx}<br>Stage: {label}<br>Time: {t:.4f}s" for idx, t in zip(df["FrameIndex"], df[col_time])],
                        hoverinfo="text",
                        name=label
                    ),
                    row=1, col=1
                )

            fig_plotly.add_trace(
                go.Scatter(
                    x=df["t3"],
                    y=df["InterFrameInterval_numeric"],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=df["FrameIndex"],
                        colorscale="Turbo",
                        opacity=0.7
                    ),
                    text=[f"Frame #{idx}<br>Interval: {val:.3f}ms" if not np.isnan(val) else "" for idx, val in zip(df["FrameIndex"], df["InterFrameInterval_numeric"])],
                    hoverinfo="text",
                    showlegend=False
                ),
                row=2, col=1
            )
            
            fig_plotly.add_hline(y=4.167, line_dash="dash", line_color="red", annotation_text="Target 240Hz (4.167ms)", row=2, col=1)

            fig_plotly.add_trace(
                go.Scatter(
                    x=df["t3"],
                    y=df["UploadDrawDuration_ms"],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=df["FrameIndex"],
                        colorscale="Turbo",
                        opacity=0.7
                    ),
                    text=[f"Frame #{idx}<br>Draw Duration: {dur:.3f}ms" for idx, dur in zip(df["FrameIndex"], df["UploadDrawDuration_ms"])],
                    hoverinfo="text",
                    showlegend=False
                ),
                row=3, col=1
            )

            fig_plotly.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0b0c10",
                plot_bgcolor="#14161d",
                height=900,
                title_text="GAIM240 Perception Profiling Interactive Dashboard",
                title_x=0.5,
                title_font=dict(size=18, color="#9d4edd")
            )
            
            fig_plotly.update_yaxes(
                tickvals=[1, 2, 3, 4],
                ticktext=["1. RAM Fetch", "2. GPU Draw", "3. VSync Flip", "4. Swap Complete"],
                row=1, col=1
            )
            fig_plotly.update_yaxes(title_text="Interval (ms)", row=2, col=1)
            fig_plotly.update_yaxes(title_text="Duration (ms)", row=3, col=1)
            fig_plotly.update_xaxes(title_text="Time (seconds)", row=3, col=1)

            output_html = "profile_interactive.html"
            fig_plotly.write_html(output_html)
            print(f"Interactive HTML dashboard saved to: {Path(output_html).resolve()}")
            
            try:
                webbrowser.open(Path(output_html).resolve().as_uri())
            except Exception:
                pass
        except Exception as e:
            print(f"Could not generate Plotly dashboard: {e}")
    else:
        print("\nTip: Pass '--plotly' (or '--html') to also generate and open an interactive Plotly HTML dashboard.")
        
    print("Displaying Matplotlib window...")
    plt.show()

if __name__ == "__main__":
    generate_raster_plot()

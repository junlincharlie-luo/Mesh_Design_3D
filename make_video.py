"""Generate a video of the laser single-track simulation results.

Reads VTU output files and renders the top-surface (z=0) temperature
field as an animation (MP4).
"""
import os, glob, argparse
import numpy as np
import meshio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import Normalize, PowerNorm
from matplotlib import cm


def load_top_surface(vtu_path):
    """Load VTU file and extract top-surface (z ~ 0) points and temperature."""
    mesh = meshio.read(vtu_path)
    pts = mesh.points
    T = mesh.point_data["Temperature"]
    tol = 1e-10
    mask = np.abs(pts[:, 2]) < tol
    x = pts[mask, 0] * 1e3   # convert to mm
    y = pts[mask, 1] * 1e3
    temp = T[mask]
    return x, y, temp


def main():
    parser = argparse.ArgumentParser(description="Generate video from laser3d VTU results")
    parser.add_argument("--input_dir", default="output_single_track_dx0.000125_sigma2e-04",
                        help="Directory containing T*.vtu files")
    parser.add_argument("--output", default="laser3d_simulation.mp4",
                        help="Output video filename")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second")
    parser.add_argument("--dpi", type=int, default=180, help="Figure DPI")
    parser.add_argument("--t_min", type=float, default=290, help="Colorbar min temperature [K]")
    parser.add_argument("--t_max", type=float, default=3500, help="Colorbar max temperature [K]")
    args = parser.parse_args()

    vtu_files = sorted(glob.glob(os.path.join(args.input_dir, "T*.vtu")))
    if not vtu_files:
        print(f"No VTU files found in {args.input_dir}")
        return
    print(f"Found {len(vtu_files)} VTU files")

    # Simulation parameters
    dt = 0.125e-3
    savefreq = 5
    dt_frame = dt * savefreq

    # Laser path in mm
    laser_x_start, laser_x_end = -7.5, 7.5
    laser_t_start, laser_t_end = 0.0, 0.025

    # Load first frame for triangulation
    x, y, _ = load_top_surface(vtu_files[0])
    tri = mtri.Triangulation(x, y)

    frame_dir = "frames_tmp"
    os.makedirs(frame_dir, exist_ok=True)

    # Use PowerNorm (gamma < 1) to spread out the low end and make the
    # hot region pop against the cool background
    norm = PowerNorm(gamma=0.35, vmin=args.t_min, vmax=args.t_max)
    levels = np.linspace(args.t_min, args.t_max, 300)

    print("Rendering frames...")
    for i, vtu_file in enumerate(vtu_files):
        _, _, temp = load_top_surface(vtu_file)
        t_sim = i * dt_frame

        fig, ax = plt.subplots(figsize=(13, 5.5))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")

        temp_disp = np.clip(temp, args.t_min, args.t_max)

        tcf = ax.tricontourf(tri, temp_disp, levels=levels, cmap="inferno",
                             norm=norm, extend="max")

        # Colorbar
        cbar = fig.colorbar(tcf, ax=ax, label="Temperature [K]", shrink=0.92,
                            pad=0.02, aspect=25)
        cbar.ax.yaxis.label.set_color("white")
        cbar.ax.yaxis.label.set_fontsize(12)
        cbar.ax.tick_params(colors="white")
        # Nice tick values
        cbar.set_ticks([300, 500, 1000, 1500, 2000, 2500, 3000, 3500])

        # Mark laser position
        if laser_t_start <= t_sim <= laser_t_end:
            frac = (t_sim - laser_t_start) / (laser_t_end - laser_t_start)
            laser_x = laser_x_start + frac * (laser_x_end - laser_x_start)
            ax.plot(laser_x, 0, "o", color="cyan", markersize=10,
                    markeredgecolor="white", markeredgewidth=1.5, zorder=10)
            ax.annotate("Laser", (laser_x, 0), textcoords="offset points",
                        xytext=(0, 14), ha="center", fontsize=9,
                        color="cyan", fontweight="bold")

        # Melt pool contour (Ti-6Al-4V solidus ~ 1878 K)
        try:
            ax.tricontour(tri, temp, levels=[1878], colors=["lime"],
                          linewidths=1.0, linestyles="--")
        except Exception:
            pass

        ax.set_xlabel("x [mm]", color="white", fontsize=12)
        ax.set_ylabel("y [mm]", color="white", fontsize=12)
        ax.set_title(
            f"Laser Single Track  |  Top Surface Temperature  |  "
            f"t = {t_sim*1e3:5.2f} ms  |  T$_{{max}}$ = {temp.max():.0f} K",
            color="white", fontsize=13, fontweight="bold", pad=10)
        ax.set_aspect("equal")
        ax.set_xlim(-13, 13)
        ax.set_ylim(-5, 5)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")

        frame_path = os.path.join(frame_dir, f"frame_{i:04d}.png")
        fig.savefig(frame_path, dpi=args.dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)

        print(f"  Frame {i+1:>2}/{len(vtu_files)}: t = {t_sim*1e3:5.2f} ms, "
              f"T_max = {temp.max():.0f} K", flush=True)

    # Assemble video
    print(f"\nAssembling video: {args.output}")
    ffmpeg_cmd = (
        f"ffmpeg -y -framerate {args.fps} "
        f"-i {frame_dir}/frame_%04d.png "
        f"-c:v libx264 -pix_fmt yuv420p -crf 18 "
        f"-vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' "
        f"{args.output}"
    )
    os.system(ffmpeg_cmd)

    # Clean up frames
    for f in glob.glob(os.path.join(frame_dir, "*.png")):
        os.remove(f)
    os.rmdir(frame_dir)

    if os.path.exists(args.output):
        size_mb = os.path.getsize(args.output) / 1e6
        dur = len(vtu_files) / args.fps
        print(f"Video saved: {args.output} ({size_mb:.1f} MB, {dur:.1f}s @ {args.fps} fps)")
    else:
        print("Error: video was not created")


if __name__ == "__main__":
    main()

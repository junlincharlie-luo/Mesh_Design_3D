# Job Submission Guide: GPU Cluster via SLURM

## Quick Start

From the project directory, submit the job with:

```bash
cd /home/darve/junlin25/Mesh_Design_3D
sbatch run_laser3d.sh
```

---

## SLURM Job Script: `run_laser3d.sh`

### Resource Configuration

| Directive | Value | Description |
|---|---|---|
| `--job-name` | `laser3d` | Job name shown in the queue |
| `--partition` | `gpu-ampere` | Target the GPU Ampere node partition |
| `--nodes` | 1 | Single compute node |
| `--ntasks` | 1 | Single task (serial execution) |
| `--cpus-per-task` | 8 | 8 CPU cores allocated |
| `--mem` | 64G | 64 GB of memory |
| `--time` | 02:00:00 | 2-hour maximum wall time |
| `--output` | `laser3d_%j.out` | Standard output log (`%j` = job ID) |
| `--error` | `laser3d_%j.err` | Standard error log (`%j` = job ID) |

### Script Workflow

1. **Print job info** — logs the job ID, hostname, start time, and CPU count
2. **Activate conda environment** — sources miniconda and activates the `fenics` environment
3. **Set library path** — ensures FEniCS shared libraries are found
4. **Change to submit directory** — `cd $SLURM_SUBMIT_DIR`
5. **Run the simulation** — executes `laser_single_track.py` with the following parameters:

| Parameter | Value | Description |
|---|---|---|
| `--dx` | 0.125e-3 | Mesh spacing: 0.125 mm |
| `--dt` | 0.125e-3 | Time step: 0.125 ms |
| `--t_final` | 0.025 | Total simulation time: 25 ms |
| `--laser_sigma` | 2e-4 | Laser beam sigma: 0.2 mm |
| `--output_folder` | `output_single_track` | Output directory for VTU files |
| `--savefreq` | 5 | Save every 5 time steps |

---

## Monitoring Jobs

### Check job status

```bash
squeue -u junlin25
```

### Cancel a running job

```bash
scancel <job_id>
```

### View completed job details

```bash
sacct -j <job_id>
```

### View job output in real time

```bash
tail -f laser3d_<job_id>.out
tail -f laser3d_<job_id>.err
```

---

## Output Files

After the job completes, you will find:

| File | Description |
|---|---|
| `laser3d_<job_id>.out` | Job stdout — simulation parameters and progress |
| `laser3d_<job_id>.err` | Job stderr — progress bar and any warnings |
| `output_single_track_dx*_sigma*/T*.vtu` | Temperature field snapshots (viewable in ParaView) |

---

## Customizing the Run

To modify simulation parameters, either edit `run_laser3d.sh` directly or override arguments on the command line. All available options:

```bash
python3 laser_single_track.py --help
```

Key parameters you may want to adjust:

| Parameter | Flag | Example |
|---|---|---|
| Mesh resolution | `--dx` | `--dx 0.0625e-3` (finer mesh) |
| Time step | `--dt` | `--dt 0.0625e-3` |
| Laser power | `--laser_path` | Change the last value in the list |
| Scan speed | `--laser_path` | Adjust start/end time and positions |
| Save frequency | `--savefreq` | `--savefreq 1` (every step) |
| Surface refinement | `--refine_surface` | Add flag to enable |

---

## Notes

- The simulation runs in **serial** (single process) — FEniCS handles the solve with CG + Hypre AMG preconditioner using the allocated CPUs for parallel linear algebra
- Typical runtime: ~5 minutes for the default parameters on 8 CPUs
- VTU output files are large (~168 MB each) — adjust `--savefreq` to control disk usage

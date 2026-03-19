# How the Laser 3D Single Track Simulation Is Computed

## 1. The Physics — What PDE is being solved?

The code solves the **3D transient heat equation** for a laser scanning across a metal block (Ti-6Al-4V):

$$
\rho \, c \, \frac{\partial u}{\partial t} = k \, \nabla^2 u
$$

where $u(\mathbf{x}, t)$ is temperature [K], with:

| Symbol | Meaning | Value | Unit |
|---|---|---|---|
| $k$ | Thermal conductivity | 6.7 | W/(m·K) |
| $c$ | Specific heat | 0.560 | J/(g·K) |
| $\rho$ | Density | 4.43×10⁶ | g/m³ |
| $\alpha = k / (c \rho)$ | Thermal diffusivity | 2.70×10⁻⁶ | m²/s |

**Boundary conditions:**
- **Top surface** ($z = 0$): Neumann BC — a moving Gaussian laser heat flux $g(\mathbf{x}, t)$ is applied
- **All other surfaces**: Zero flux (insulated), i.e. $\frac{\partial u}{\partial n} = 0$

**Initial condition:** $u(\mathbf{x}, 0) = 300$ K (room temperature everywhere)

## 2. The Laser Source — How the heat flux is defined

The laser is modeled as a **2D Gaussian** moving along the top surface:

$$
g(\mathbf{x}, t) = \frac{P_{\text{abs}}}{c \, \rho} \cdot \frac{1}{2\pi\sigma^2} \, \exp\!\left( -\frac{(x - x_L(t))^2 + (y - y_L(t))^2}{2\sigma^2} \right)
$$

where the laser center moves linearly:

$$
x_L(t) = x_0 + v_x \cdot t, \quad y_L(t) = y_0 + v_y \cdot t
$$

| Parameter | Value | Meaning |
|---|---|---|
| $P$ | 250 W | Laser power |
| Absorptivity | 1.0 | Fraction absorbed |
| $\sigma$ | 0.2 mm | Gaussian beam radius |
| $(x_0, y_0)$ | (-7.5, 0) mm | Start position |
| $(x_1, y_1)$ | (7.5, 0) mm | End position |
| $v_x$ | 0.6 m/s | Scan speed (15 mm in 25 ms) |

The `time_subdiv` parameter (0 in our run) controls sub-stepping within $\Delta t$:
- `0`: evaluate $g$ at time $t$ only (cheapest)
- `1`: trapezoidal average at $t$ and $t + \Delta t$
- `2` / `4`: Simpson's / Boole's rule for more accurate time integration of the moving source

*(Code reference: `setup_laser_expression()`, lines 127–176)*

## 3. Spatial Discretization — Finite Element Method

### 3.1 Mesh generation

`BoxMesh` creates a structured 3D mesh of the domain $[-12.5, 12.5] \times [-4.5, 4.5] \times [-5, 0]$ mm:

```
BoxMesh(Point(-0.5*Lx, -0.5*Ly, -Lz), Point(0.5*Lx, 0.5*Ly, 0), nx, ny, nz)
```

Each rectangular brick is split into **6 tetrahedra**:

$$
N_{\text{tet}} = 200 \times 72 \times 40 \times 6 = 3{,}456{,}000 \text{ tetrahedra}
$$

*(Code reference: `setup_mesh_and_function_space()`, line 101–103)*

### 3.2 Function space

P1 (piecewise-linear Lagrange) elements — one DOF per mesh vertex:

$$
N_{\text{DOFs}} = (200+1) \times (72+1) \times (40+1) = 201 \times 73 \times 41 = 601{,}593
$$

*(Code reference: line 116)*

### 3.3 Boundary marking

The top surface ($z = 0$) is tagged as boundary ID `1` to apply the laser flux. All other boundaries default to tag `0` (natural zero-flux BC).

*(Code reference: `Top_surface` class, lines 8–11; marking at lines 118–121)*

## 4. Time Discretization — Backward Euler

Dividing the PDE by $c \rho$ and using backward Euler in time:

$$
\frac{u^{n+1} - u^n}{\Delta t} = \alpha \, \nabla^2 u^{n+1} + g^{n+1}(\mathbf{x}) \quad \text{on } \Gamma_{\text{top}}
$$

Rearranging:

$$
u^{n+1} - \Delta t \, \alpha \, \nabla^2 u^{n+1} = u^n + \Delta t \, g^{n+1}
$$

| Parameter | Value |
|---|---|
| $\Delta t$ | 0.125 ms |
| $t_{\text{final}}$ | 25 ms |
| Number of steps | 200 |

*(Code reference: line 49)*

## 5. Weak Form — What FEniCS actually assembles

Multiply by test function $v$ and integrate by parts. The code writes this on **one line** (line 182):

```python
F = u*v*dx + dt*alpha*dot(grad(u), grad(v))*dx - u_n*v*dx - dt*g*v*ds(1)
```

This corresponds to the weak form:

$$
\underbrace{\int_\Omega u^{n+1} v \, d\mathbf{x}}_{\text{mass term}} + \underbrace{\Delta t \, \alpha \int_\Omega \nabla u^{n+1} \cdot \nabla v \, d\mathbf{x}}_{\text{stiffness term}} = \underbrace{\int_\Omega u^n \, v \, d\mathbf{x}}_{\text{previous solution}} + \underbrace{\Delta t \int_{\Gamma_\text{top}} g \, v \, ds}_{\text{laser flux}}
$$

FEniCS splits this into:

| Side | Matrix/Vector | What it contains |
|---|---|---|
| **Left-hand side** `a` | $\mathbf{A} \mathbf{u}^{n+1}$ | Mass matrix $\mathbf{M}$ + $\Delta t \, \alpha$ × Stiffness matrix $\mathbf{K}$ |
| **Right-hand side** `L` | $\mathbf{b}$ | $\mathbf{M} \mathbf{u}^n$ + $\Delta t$ × laser flux vector |

Key point: **$\mathbf{A}$ is assembled once** (line 186) because $M$ and $K$ don't change between time steps. Only the RHS vector $\mathbf{b}$ is reassembled each step (because $u^n$ and the laser position change).

*(Code reference: `setup_variational_problem()`, lines 178–186)*

## 6. Linear Solve — What happens each time step

Each of the 200 time steps executes this loop (lines 213–223):

```python
for n in range(num_steps):       # 200 iterations
    b = assemble(L)              # Step A: reassemble RHS vector
    solver.solve(u.vector(), b)  # Step B: solve A * u = b
    t += dt                      # Step C: advance time
    g.t = t                      # Step D: move the laser
    u_n.assign(u)                # Step E: store solution for next step
```

### Step A: RHS Assembly (~30% of time per step)
- Loop over all 3.46M tetrahedra
- For each element, compute local contribution using quadrature
- The laser expression `g` is evaluated at quadrature points on the top-surface facets only
- Scatter into global vector $\mathbf{b}$ (601,593 entries)

### Step B: Linear Solve (~60% of time per step)
Solve $\mathbf{A} \mathbf{u}^{n+1} = \mathbf{b}$ using:

| Component | Method | What it does |
|---|---|---|
| **Solver** | Conjugate Gradient (CG) | Krylov iterative method for symmetric positive-definite systems |
| **Preconditioner** | hypre AMG | Algebraic Multigrid — builds a hierarchy of coarser systems to accelerate convergence |

$\mathbf{A}$ is a **601,593 × 601,593 sparse matrix** with ~30 non-zero entries per row (from the tetrahedral P1 stencil). CG+AMG typically converges in **O(10–30) iterations**, independent of mesh size.

### Step C–E: Time update (~10% of time per step)
- Update laser position parameter `g.t = t` (moves the Gaussian center)
- Copy $u^{n+1} \to u^n$ for the next time step
- Optionally write a `.vtu` file (every 5 steps → 40 output files)

*(Code reference: `run()`, lines 200–224; `setup_solver()`, lines 188–193)*

## 7. Computational Cost Summary

### Per time step

| Operation | FLOPs estimate | Wall time |
|---|---|---|
| RHS assembly (3.46M elements) | ~10⁸ | ~0.5 s |
| CG+AMG solve (601K unknowns, ~20 iters) | ~10⁹ | ~0.8 s |
| I/O (write 168 MB VTU, every 5 steps) | — | ~0.2 s (amortized) |
| **Total per step** | | **~1.5 s** |

### Full simulation

| Metric | Value |
|---|---|
| Time steps | 200 |
| Wall time | 5 min 48 sec |
| Output | 40 VTU files, 9.9 GB total |
| Peak memory | ~2 GB |

### Execution pipeline diagram

```
[Once at startup]
  BoxMesh (200×72×40 → 3.46M tets)
    ↓
  FunctionSpace (P1, 601K DOFs)
    ↓
  Assemble A = M + Δt·α·K   ← done ONCE, reused every step
    ↓
  Setup CG + AMG preconditioner

[Repeated 200 times]
  ┌─────────────────────────────────────────────────┐
  │  1. Assemble b = M·uⁿ + Δt·∫ g·v ds            │
  │  2. Solve A·u^{n+1} = b   (CG + hypre AMG)      │
  │  3. Update laser position: g.t += Δt              │
  │  4. u^n ← u^{n+1}                                │
  │  5. (every 5 steps) Write VTU file                │
  └─────────────────────────────────────────────────┘
```

## 8. SLURM Resource Allocation

| Resource | Allocated | Why |
|---|---|---|
| Partition | `gpu-ampere` | 48 CPUs, 180 GB RAM, nodes were idle |
| Nodes | 1 | Problem fits in single-node memory |
| Tasks | 1 | Serial run (FEniCS MPI not configured) |
| CPUs | 8 | For threaded BLAS/LAPACK inside PETSc |
| Memory | 64 GB | ~30× working set for safety margin |
| Wall time | 2 hours | Conservative; actual was 6 min |

**Note:** FEniCS legacy does not use GPUs. The gpu-ampere partition was chosen for its CPU count and availability, not for GPU compute.

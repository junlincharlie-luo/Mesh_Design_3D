# 3D Laser Single-Track Simulation: Calculation Process

## Problem Overview

This simulation models a **laser single-track scan** on a Ti-6Al-4V rectangular block, solving the transient 3D heat conduction equation with a moving Gaussian laser heat source on the top surface. It is implemented using the **FEniCS** finite element library.

---

## 1. Governing Equation

The 3D transient heat equation:

$$\rho c \frac{\partial T}{\partial t} = k \nabla^2 T$$

Dividing both sides by $\rho c$ gives:

$$\frac{\partial T}{\partial t} = \alpha \nabla^2 T$$

where $\alpha = k / (\rho c)$ is the thermal diffusivity.

### Material Properties (Ti-6Al-4V)

| Property | Symbol | Value | Unit |
|---|---|---|---|
| Thermal conductivity | $k$ | 6.7 | W/(m·K) |
| Specific heat | $c$ | 0.56 | J/(g·K) |
| Density | $\rho$ | 4.43 × 10⁶ | g/m³ |
| Initial temperature | $T_0$ | 300 | K |

---

## 2. Domain and Mesh

The simulation domain is a rectangular box:

- **Dimensions**: $L_x \times L_y \times L_z$ = 25 mm × 9 mm × 5 mm
- **Mesh spacing**: $dx = dy = dz$ = 0.125 mm (uniform)
- **Grid points**: $n_x \times n_y \times n_z$ = 200 × 72 × 40

The domain is meshed with tetrahedral elements using FEniCS `BoxMesh`. The top surface ($z = 0$) is identified as a boundary subdomain where the laser heat flux is applied.

### Finite Element Space

- Element type: **P1** (linear Lagrange)
- Polynomial degree: 1

---

## 3. Boundary Conditions

### Top Surface ($z = 0$): Laser Heat Flux (Neumann BC)

A moving 2D Gaussian heat flux is applied on the top surface:

$$g(x, y, t) = \frac{P_{\text{absorbed}}}{2\pi\sigma^2 \rho c} \exp\left( -\frac{(x - x_L(t))^2 + (y - y_L(t))^2}{2\sigma^2} \right)$$

where:
- $P_{\text{absorbed}} = P \cdot \eta$ is the absorbed laser power ($P$ = 250 W, absorptivity $\eta$ = 1.0)
- $\sigma$ = 0.2 mm is the laser beam radius parameter
- $(x_L(t), y_L(t))$ is the laser center position, which moves linearly in time

### Laser Path

The laser travels in a straight line from $(x_0, y_0)$ to $(x_1, y_1)$ at constant velocity:

$$x_L(t) = x_0 + v_x \cdot t, \quad y_L(t) = y_0 + v_y \cdot t$$

- Scan speed: 0.6 m/s (in x-direction)
- Scan duration: 25 ms

### Other Surfaces: Zero-Flux (Neumann BC)

All other boundaries have a natural zero-flux condition (adiabatic), meaning no heat enters or leaves through the sides or bottom.

### Initial Condition

$$T(x, y, z, 0) = T_0 = 300 \text{ K}$$

---

## 4. Time Discretization: Implicit Euler

The time derivative is discretized using the **implicit (backward) Euler** method:

$$\frac{T^{n+1} - T^n}{\Delta t} = \alpha \nabla^2 T^{n+1} + \text{(source terms at } t^{n+1}\text{)}$$

This scheme is **unconditionally stable**, which is important given the sharp spatial gradients from the concentrated laser source.

### Time-Step Subdivision (Optional)

To improve accuracy in representing the moving laser within a single time step, the code supports **trapezoidal-rule averaging** of the Gaussian at multiple sub-positions:

| Subdivision level | Quadrature points | Method |
|---|---|---|
| 0 | 1 | Evaluate at $t$ only |
| 1 | 2 | Average at $t$ and $t + \Delta t$ |
| 2 | 3 | Simpson's rule at $t$, $t + \Delta t/2$, $t + \Delta t$ |
| 4 | 5 | Composite trapezoidal at $t$, $t + \Delta t/4$, ..., $t + \Delta t$ |

---

## 5. Weak (Variational) Formulation

Multiplying the heat equation by a test function $v$ and integrating by parts gives the weak form. After implicit Euler discretization:

$$\int_\Omega T^{n+1} v \, d\Omega + \Delta t \cdot \alpha \int_\Omega \nabla T^{n+1} \cdot \nabla v \, d\Omega = \int_\Omega T^n v \, d\Omega + \Delta t \int_{\Gamma_{\text{top}}} g \cdot v \, d\Gamma$$

This can be written in matrix form as:

$$\mathbf{A} \mathbf{u} = \mathbf{b}$$

where:
- **A** (left-hand side): mass matrix + $\Delta t \cdot \alpha \times$ stiffness matrix — **constant**, assembled once
- **b** (right-hand side): previous solution contribution + laser flux — **reassembled every time step** as the laser moves

---

## 6. Linear Solver

Each time step requires solving the linear system $\mathbf{A} \mathbf{u} = \mathbf{b}$:

- **Solver**: Conjugate Gradient (CG) — suitable because $\mathbf{A}$ is symmetric positive definite
- **Preconditioner**: Hypre AMG (Algebraic Multigrid) — provides efficient convergence for elliptic-type problems on structured meshes

Since $\mathbf{A}$ is constant throughout the simulation, it is assembled and factored only once, and only $\mathbf{b}$ is updated at each step.

---

## 7. Time-Stepping Loop

The simulation advances through **200 time steps** ($\Delta t$ = 0.125 ms, total = 25 ms):

```
Initialize: T = 300 K everywhere
Assemble matrix A (once)

For n = 0, 1, ..., 199:
    1. Assemble load vector b (depends on current laser position)
    2. Solve A · u = b using CG + AMG
    3. Advance time: t ← t + Δt
    4. Update laser position in the source expression
    5. Save temperature field T to VTU file
    6. Update: T_n ← T for the next step
```

---

## 8. Output and Post-Processing

- **VTU files**: One per time step (41 files saved), each containing the full 3D temperature field. These can be visualized in ParaView.
- **Video**: `make_video.py` renders the temperature field into `laser3d_simulation.mp4`.
- **Preview**: `frame_preview.png` shows a single frame snapshot.

---

## 9. Simulation Parameters Summary

| Parameter | Value |
|---|---|
| Domain size | 25 × 9 × 5 mm |
| Mesh spacing | 0.125 mm |
| Grid points | 200 × 72 × 40 |
| Finite element | P1 (linear Lagrange) |
| Time step | 0.125 ms |
| Total time | 25 ms |
| Number of steps | 200 |
| Laser power | 250 W |
| Laser sigma | 0.2 mm |
| Scan speed | 0.6 m/s |
| Solver | CG + Hypre AMG |
| Runtime | ~5 min 21 sec (8 CPUs, GPU node) |

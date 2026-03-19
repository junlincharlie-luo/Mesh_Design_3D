'''Single track laser scanning on rectangular block'''
# adapted from https://gitlab.com/micronano/FEniCS/-/blob/master/legacy/examples/heat/heat_rect_block.py
import os, sys, argparse, numpy as np, ast
from fenics import *
from tqdm import tqdm


class Top_surface(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-14
        return on_boundary and near(x[2], 0, tol)

class LaserSingleTrack:
    def __init__(self, dx=0.125e-3, dt=0.125e-3, time_subdiv=0, t_final=0.025,
                 Lx=25e-3, Ly=9e-3, Lz=5e-3, laser_sigma=1e-5,
                 laser_path=[0.0, -7.5e-3, 0.0, 0.025, 7.5e-3, 0.0, 250.0],
                 laser_absorptivity=1.0, k=6.7, c=0.560, rho=4.43e6, u_0=300,
                 finite_element_type='P', polynomial_degree=1,
                 linear_solver="cg", preconditioner="hypre_amg",
                 log_level=50, output_folder="output_single_track",
                 output_file_name='T.pvd', savefreq=1,
                 refine_surface=False, refine_levels=1):
        self.dx = dx
        self.dt = dt
        self.time_subdiv = time_subdiv
        self.t_final = t_final
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.laser_sigma = laser_sigma
        self.laser_path = laser_path
        self.laser_absorptivity = laser_absorptivity
        self.k, self.c, self.rho = k, c, rho
        self.u_0 = u_0
        self.finite_element_type = finite_element_type
        self.polynomial_degree = polynomial_degree
        self.linear_solver = linear_solver
        self.preconditioner = preconditioner
        self.log_level = log_level
        self.output_folder = output_folder
        self.output_file_name = output_file_name
        self.savefreq = savefreq
        self.refine_surface = refine_surface
        self.refine_levels = refine_levels

        set_log_level(self.log_level)

        self.output_folder = f'{self.output_folder}_dx{str(self.dx)}_sigma{self.laser_sigma:.0e}'
        self.alpha = self.k / (self.c * self.rho)

        self.num_steps = int(self.t_final / self.dt)
        self.nx = int(np.round(self.Lx / self.dx))
        self.ny = int(np.round(self.Ly / self.dx))
        self.nz = int(np.round(self.Lz / self.dx))

        if np.allclose([self.Lx, self.Ly, self.Lz], 
                      [self.nx*self.dx, self.ny*self.dx, self.nz*self.dx], atol=1e-12):
            print(f"  dx={self.dx} divide [{self.Lx}, {self.Ly}, {self.Lz}] exactly")
            self.dy = self.dx
            self.dz = self.dx
        else:
            print(f". dx={self.dx} does not divide [{self.Lx}, {self.Ly}, {self.Lz}] exactly")
            self.dy = self.Ly / self.ny
            self.dz = self.Lz / self.nz

        self.laser_t_start, self.laser_t_end = self.laser_path[0], self.laser_path[3]
        self.laser_x_start, self.laser_x_end = self.laser_path[1], self.laser_path[4]
        self.laser_y_start, self.laser_y_end = self.laser_path[2], self.laser_path[5]
        self.laser_power = self.laser_path[6]
        self.laser_power_absorbed = self.laser_power * self.laser_absorptivity
        self.laser_distance = np.linalg.norm(
            np.array([self.laser_x_end, self.laser_y_end]) - 
            np.array([self.laser_x_start, self.laser_y_start]))
        self.laser_vx = (self.laser_x_end - self.laser_x_start) / (self.laser_t_end - self.laser_t_start)
        self.laser_vy = (self.laser_y_end - self.laser_y_start) / (self.laser_t_end - self.laser_t_start)
        self.laser_I = self.laser_power_absorbed / (self.c * self.rho)

        self._print_info()

    def _print_info(self):
        print(f'Simulation time step size dt: {self.dt} [s]')
        print(f' time step subdivision: {self.time_subdiv}')
        print(f'Total simulation time: {self.t_final} [s]')
        print(f'Number of simulation time steps: {self.num_steps}')
        print('')
        print(f'Simulation domain size Lx, Ly, Lz: {self.Lx}, {self.Ly}, {self.Lz} [m]')
        print(f'Mesh size dx, dy, dz: {self.dx}, {self.dy}, {self.dz} [m]')
        print(f'Number of grid points nx, ny, nz: {self.nx}, {self.ny}, {self.nz}')
        print(f'FE type: {self.finite_element_type}, polynomial degree: {self.polynomial_degree}')
        print(f'Surface refinement: {"enabled" if self.refine_surface else "disabled"}' + 
              (f' ({self.refine_levels} level(s))' if self.refine_surface else ''))
        print('')
        print(f'Laser path: {self.laser_path}')
        print(f'Thermal conductivity: {self.k} [W/(m*K)]')
        print(f'Specific heat: {self.c} [J/(g*K)]')
        print(f'Density: {self.rho:.2e} [g/m^3]')
        print(f'Initial temperature: {self.u_0} [K]')
        print(f"Laser scan speed: {self.laser_vx:.2e}, {self.laser_vy:.2e} m/s")
        print(f"Laser power: {self.laser_power:.1f} W, absorptivity: {self.laser_absorptivity:.2f}, absorbed: {self.laser_power_absorbed:.1f} W")
        print(f"Total simulation time period: {self.t_final:.2e} s")

    def setup_mesh_and_function_space(self):
        initial_mesh = BoxMesh(Point(-0.5*self.Lx, -0.5*self.Ly, -self.Lz), 
                               Point(0.5*self.Lx, 0.5*self.Ly, 0), 
                               self.nx, self.ny, self.nz)
        if self.refine_surface:
            threshold_z = -self.Lz / self.nz * 1.01
            for level in range(self.refine_levels):
                cell_markers = MeshFunction("bool", initial_mesh, initial_mesh.topology().dim())
                cell_markers.set_all(False)
                for cell in cells(initial_mesh):
                    p = cell.midpoint()
                    if p.z() > threshold_z:
                        cell_markers[cell] = True
                initial_mesh = refine(initial_mesh, cell_markers)
            print(f"Applied {self.refine_levels} level(s) of surface refinement (z > {threshold_z:.6e})")
        self.mesh = initial_mesh
        self.V = FunctionSpace(self.mesh, self.finite_element_type, self.polynomial_degree)

        top_surface = Top_surface()
        self.boundaries = MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1, 0)
        top_surface.mark(self.boundaries, 1)
        self.ds = Measure("ds", domain=self.mesh, subdomain_data=self.boundaries)

    def setup_initial_condition(self):
        u_n = Constant(self.u_0)
        self.u_n = interpolate(u_n, self.V)

    def setup_laser_expression(self):
        # TODO: currently assumes laser stays on from t to t+dt
        expr_dict = {0: '(t >= tmin && t<=tmax) ? '
                        'prefac * '
                        'exp(- ( (x[0]-(x0_0+vx*(t-tmin)))*(x[0]-(x0_0+vx*(t-tmin)) ) '
                        '      + (x[1]-(y0_0+vy*(t-tmin)))*(x[1]-(y0_0+vy*(t-tmin)) ) ) / (2*sigma*sigma) ) '
                        ': 0',
                    1:  '(t >= tmin && t<=tmax) ? '
                        'prefac * ('
                        '  0.5 * exp(- ( (x[0]-(x0_0+vx*(t-tmin)))*(x[0]-(x0_0+vx*(t-tmin)) ) '
                        '              + (x[1]-(y0_0+vy*(t-tmin)))*(x[1]-(y0_0+vy*(t-tmin)) ) ) / (2*sigma*sigma) ) '
                        '+ 0.5 * exp(- ( (x[0]-(x0_0+vx*(t+dts-tmin)))*(x[0]-(x0_0+vx*(t+dts-tmin)) ) '
                        '              + (x[1]-(y0_0+vy*(t+dts-tmin)))*(x[1]-(y0_0+vy*(t+dts-tmin)) ) ) / (2*sigma*sigma) ) '
                        ') '
                        ': 0',
                     2: '(t >= tmin && t<=tmax) ? '
                        'prefac * ('
                        '  0.5 * exp(- ( (x[0]-(x0_0+vx*(t-tmin)))*(x[0]-(x0_0+vx*(t-tmin)) ) '
                        '              + (x[1]-(y0_0+vy*(t-tmin)))*(x[1]-(y0_0+vy*(t-tmin)) ) ) / (2*sigma*sigma) ) '
                        '+ 1.0 * exp(- ( (x[0]-(x0_0+vx*(t+dts*0.5-tmin)))*(x[0]-(x0_0+vx*(t+dts*0.5-tmin)) ) '
                        '              + (x[1]-(y0_0+vy*(t+dts*0.5-tmin)))*(x[1]-(y0_0+vy*(t+dts*0.5-tmin)) ) ) / (2*sigma*sigma) ) '
                        '+ 0.5 * exp(- ( (x[0]-(x0_0+vx*(t+dts-tmin)))*(x[0]-(x0_0+vx*(t+dts-tmin)) ) '
                        '              + (x[1]-(y0_0+vy*(t+dts-tmin)))*(x[1]-(y0_0+vy*(t+dts-tmin)) ) ) / (2*sigma*sigma) ) '
                        ') / 2.0'
                        ': 0',
                     4: '(t >= tmin && t<=tmax) ? '
                        'prefac * ('
                        '  0.5 * exp(- ( (x[0]-(x0_0+vx*(t-tmin)))*(x[0]-(x0_0+vx*(t-tmin)) ) '
                        '              + (x[1]-(y0_0+vy*(t-tmin)))*(x[1]-(y0_0+vy*(t-tmin)) ) ) / (2*sigma*sigma) ) '
                        '+ 1.0 * exp(- ( (x[0]-(x0_0+vx*(t+dts*0.25-tmin)))*(x[0]-(x0_0+vx*(t+dts*0.25-tmin)) ) '
                        '              + (x[1]-(y0_0+vy*(t+dts*0.25-tmin)))*(x[1]-(y0_0+vy*(t+dts*0.25-tmin)) ) ) / (2*sigma*sigma) ) '
                        '+ 1.0 * exp(- ( (x[0]-(x0_0+vx*(t+dts*0.5-tmin)))*(x[0]-(x0_0+vx*(t+dts*0.5-tmin)) ) '
                        '              + (x[1]-(y0_0+vy*(t+dts*0.5-tmin)))*(x[1]-(y0_0+vy*(t+dts*0.5-tmin)) ) ) / (2*sigma*sigma) ) '
                        '+ 1.0 * exp(- ( (x[0]-(x0_0+vx*(t+dts*0.75-tmin)))*(x[0]-(x0_0+vx*(t+dts*0.75-tmin)) ) '
                        '              + (x[1]-(y0_0+vy*(t+dts*0.75-tmin)))*(x[1]-(y0_0+vy*(t+dts*0.75-tmin)) ) ) / (2*sigma*sigma) ) '
                        '+ 0.5 * exp(- ( (x[0]-(x0_0+vx*(t+dts-tmin)))*(x[0]-(x0_0+vx*(t+dts-tmin)) ) '
                        '              + (x[1]-(y0_0+vy*(t+dts-tmin)))*(x[1]-(y0_0+vy*(t+dts-tmin)) ) ) / (2*sigma*sigma) ) '
                        ') / 4.0'
                        ': 0' }

        if self.time_subdiv not in expr_dict.keys():
            raise ValueError(f"time_subdiv ({self.time_subdiv}) not in ({expr_dict.keys()})")

        expr_str = expr_dict.get(self.time_subdiv)
        self.g = Expression(expr_str, degree=2,
                           tmin=self.laser_t_start, tmax=self.laser_t_end, dts=self.dt,
                           x0_0=self.laser_x_start, vx=self.laser_vx,
                           y0_0=self.laser_y_start, vy=self.laser_vy,
                           prefac=self.laser_I/(2.0*np.pi*self.laser_sigma**2),
                           sigma=self.laser_sigma, t=0.0)

    def setup_variational_problem(self):
        u = TrialFunction(self.V)
        v = TestFunction(self.V)

        F = u*v*dx + self.dt*self.alpha*dot(grad(u), grad(v))*dx - (self.u_n)*v*dx - self.dt*(self.g)*v*self.ds(1)
        a, L = lhs(F), rhs(F)
        self.a = a
        self.L = L
        self.A = assemble(a)

    def setup_solver(self):
        if self.linear_solver == 'lu':
            self.solver = LUSolver(self.A)
        else:
            self.solver = KrylovSolver(self.linear_solver, self.preconditioner)
            self.solver.set_operator(self.A)

    def setup_output(self):
        vtkfilename = os.path.join(self.output_folder, self.output_file_name)
        self.vtkfile = File(vtkfilename)
        return vtkfilename

    def run(self):
        self.setup_mesh_and_function_space()
        self.setup_initial_condition()
        self.setup_laser_expression()
        self.setup_variational_problem()
        self.setup_solver()
        vtkfilename = self.setup_output()

        u = Function(self.V)

        print('')
        print(f'Starting simulation... results saved to {vtkfilename}')
        t = 0
        for n in tqdm(range(self.num_steps), desc="Time-stepping", unit="step"):
            b = assemble(self.L)
            self.solver.solve(u.vector(), b)

            t += self.dt
            self.g.t = t

            if self.savefreq > 0 and ((n+1) % self.savefreq == 0 or n == self.num_steps-1):
                u.rename("Temperature", "")
                self.vtkfile << (u, t)
            self.u_n.assign(u)
        print(f'Simulation complete!')
        return u

    @classmethod
    def obtain_args(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--dx", type=float, default=0.125e-3, help="Mesh size in x [m]")
        parser.add_argument("--dt", type=float, default=0.125e-3, help='Time step [s]')
        parser.add_argument("--time_subdiv", type=int, default=0, help='Time step subdivision number')
        parser.add_argument("--t_final", type=float, default=0.025, help="total simulation time [s]")
        parser.add_argument("--Lx", type=float, default=25e-3, help='Sample dimension in x [m]')
        parser.add_argument("--Ly", type=float, default=9e-3, help='Sample dimension in y [m]')
        parser.add_argument("--Lz", type=float, default=5e-3, help='Sample dimension in z [m]')
        parser.add_argument("--laser_sigma", type=float, default=2e-4, help='Laser sigma [m]')
        parser.add_argument("--laser_path", type=ast.literal_eval, default=[0.0, -7.5e-3, 0.0,  0.025, 7.5e-3, 0.0, 250.0],
                            help='Laser path list: [t0 [s], x0 [m], y0 [m], t1 [s], x1 [m], y1 [m], power [W]]')
        parser.add_argument("--laser_absorptivity", type=float, default=1.0, help='Laser absorptivity (fraction of power absorbed, 0-1)')
        parser.add_argument("--k", type=float, default=6.7, help='Thermal conductivity [W/(m*K)] (Ti-6Al-4V)')
        parser.add_argument("--c", type=float, default=0.560, help='Specific heat [J/(g*K)] (Ti-6Al-4V ≈ 560 J/kg/K)')
        parser.add_argument("--rho", type=float, default=4.43e6, help='Density [g/m^3] (Ti-6Al-4V ≈ 4430 kg/m^3)')
        parser.add_argument("--u_0", type=float, default=300, help='Initial temperature [K]')
        parser.add_argument("--finite_element_type", type=str, default='P', help='FEniCS Finite Element type')
        parser.add_argument("--polynomial_degree", type=int, default=1, help='FEniCS polynomial degree for Finite Element')
        parser.add_argument("--linear_solver", type=str, default="cg", help='FEniCS linear solver')
        parser.add_argument("--preconditioner", type=str, default="hypre_amg", help='FEniCS conditioner')
        parser.add_argument("--log_level", type=int, default=50, help='FEniCS log level')
        parser.add_argument("--output_folder", default="output_single_track",)
        parser.add_argument("--output_file_name", type=str, default='T.pvd', help='Output file name')
        parser.add_argument("--savefreq", type=int, default=1, help="Save file every this number of steps")
        parser.add_argument("--refine_surface", action='store_true', help='Enable surface mesh refinement near z=0 (default: False)')
        parser.add_argument("--refine_levels", type=int, default=1, help='Number of refinement levels for surface mesh (default: 1)')
        return parser


def main(sys_args):
    global args, simulation, u

    p = LaserSingleTrack.obtain_args()
    args = p.parse_args(sys_args)

    simulation = LaserSingleTrack(
        dx=args.dx, dt=args.dt, time_subdiv=args.time_subdiv, t_final=args.t_final,
        Lx=args.Lx, Ly=args.Ly, Lz=args.Lz, laser_sigma=args.laser_sigma,
        laser_path=args.laser_path, laser_absorptivity=args.laser_absorptivity,
        k=args.k, c=args.c, rho=args.rho, u_0=args.u_0,
        finite_element_type=args.finite_element_type, polynomial_degree=args.polynomial_degree,
        linear_solver=args.linear_solver, preconditioner=args.preconditioner,
        log_level=args.log_level, output_folder=args.output_folder,
        output_file_name=args.output_file_name, savefreq=args.savefreq,
        refine_surface=args.refine_surface, refine_levels=args.refine_levels)
    u = simulation.run()
    return True


if __name__ == "__main__":
    main(sys.argv[1:])
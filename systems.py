import math
import numpy as np
from numpy.fft import fft2, ifft2
from scipy.integrate import solve_ivp

integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12


def get_polylib_size(order, dim, include_bias=True):
    size = 1 if include_bias else 0
    for o in range(1, order+1):
        size += math.comb(dim + o - 1, o)
    return size


class System:
    def __init__(self):
        self.name = ''
        self.is_pde = False
        self.initial_condition = (0, 0)
        self.forward_fn = None
        self.true_coefs = []

    def simulate(self, duration=10, dt=0.01, x0=None):
        x0 = self.initial_condition if x0 is None else x0
        t_train = np.arange(0, duration, dt)
        t_train_span = (t_train[0], t_train[-1])
        x_train = solve_ivp(self.forward_fn, t_train_span,
                            x0, t_eval=t_train, **integrator_keywords).y.T
        return t_train, x_train

    def correct_invalid_initial_conditions(self, x0s):
        return x0s

    def sample_initial_conditions(self, n=10, seed=None, duration=10, closeness=10):
        if seed is not None:
            np.random.seed(seed)
        _, canonical_trajectory = self.simulate(duration)
        std = np.std(canonical_trajectory, axis=0)
        l = np.arange(len(canonical_trajectory))
        starting_ixs = np.random.choice(l, size=n, replace=False)
        starting_points = canonical_trajectory[starting_ixs, :]
        x0s = np.random.normal(starting_points, std / closeness)
        return self.correct_invalid_initial_conditions(x0s)


class Lorenz(System):
    def __init__(self, p=(10, 8 / 3, 28), library_size=56):
        super(Lorenz).__init__()
        self.name = 'Lorenz'
        self.initial_condition = (-8, 8, 27)
        sigma, beta, rho = p
        self.forward_fn = lambda t, x: [
            sigma * (x[1] - x[0]),
            x[0] * (rho - x[2]) - x[1],
            x[0] * x[1] - beta * x[2],
        ]
        true_coefs = np.zeros((3, library_size))
        true_coefs[0, [1, 2]] = [-sigma, sigma]
        true_coefs[1, [1, 2, 6]] = [rho, -1, -1]
        true_coefs[2, [3, 5]] = [-beta, 1]
        self.true_coefs = true_coefs

    def sample_initial_conditions(self, n=10, seed=None, duration=10, closeness=10):
        if seed is not None:
            np.random.seed(seed)
        x = np.random.uniform(-5, 5, size=n)
        y = np.random.uniform(-5, 5, size=n)
        z = np.random.uniform(10, 40, size=n)
        return np.column_stack([x, y, z])


class Lorenz96(System):
    def __init__(self, p=8, library_size=56):
        super(Lorenz96).__init__()
        self.name = 'Lorenz96'
        F = p
        N = 5
        self.initial_condition = (F, F, F, F, F-1)
        self.forward_fn = lambda t, x: [
            (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
            for i in range(N)
        ]
        true_coefs = np.zeros((N, library_size))
        true_coefs[0, [0, 1, 14, 19]] = [F, -1, 1, -1]
        true_coefs[1, [0, 2, 8, 10]] = [F, -1, 1, -1]
        true_coefs[2, [0, 3, 7, 13]] = [F, -1, -1, 1]
        true_coefs[3, [0, 4, 12, 17]] = [F, -1, -1, 1]
        true_coefs[4, [0, 5, 9, 16]] = [F, -1, 1, -1]
        self.true_coefs = true_coefs


class VanderPol(System):
    def __init__(self, p=0.5, library_size=21):
        super(VanderPol).__init__()
        self.name = 'Van der Pol'
        self.initial_condition = (1, 0)
        self.p = p
        self.forward_fn = lambda t, x: [
            x[1],
            p * (1 - x[0] ** 2) * x[1] - x[0]
        ]
        true_coefs = np.zeros((2, library_size))
        true_coefs[0, 2] = 1
        true_coefs[1, [1, 2, 7]] = [-1, p, -p]
        self.true_coefs = true_coefs

    def sample_initial_conditions(self, n=10, seed=None, duration=10, closeness=5):
        if seed is not None:
            np.random.seed(seed)
        p = self.p
        x = np.random.uniform(-1, 1, size=n)
        y = np.random.uniform(-p, p, size=n)
        return np.column_stack([x, y])


class Duffing(System):
    def __init__(self, p=(-1, 1), library_size=10):
        super(Duffing).__init__()
        self.name = 'Duffing'
        self.initial_condition = (2.1, 1.1, 2.1, 1.5)
        omega, alpha = p
        self.forward_fn = lambda t, x: [
            x[2],
            x[3],
            -omega*x[0] - alpha*(x[0]**3 + x[0]*x[1]**2),
            -omega*x[1] - alpha*(x[0]**2*x[1] + x[1]**3)
        ]
        true_coefs = np.zeros((2, library_size))
        true_coefs[0, [1, 6, 8]] = [-omega, -alpha, -alpha]
        true_coefs[1, [2, 7, 9]] = [-omega, -alpha, -alpha]
        self.true_coefs = true_coefs

    def sample_initial_conditions(self, n=10, seed=None, duration=10, closeness=5):
        if seed is not None:
            np.random.seed(seed)
        return np.random.uniform(-math.pi, math.pi, size=(n, 4))


class Lotka(System):
    def __init__(self, p=(1, 10), library_size=21):
        super(Lotka).__init__()
        self.name = 'Lotka'
        self.initial_condition = (0.8, 0.4)
        self.forward_fn = lambda t, x: [
            p[0] * x[0] - p[1] * x[0] * x[1],
            p[1] * x[0] * x[1] - 2 * p[0] * x[1]
        ]
        true_coefs = np.zeros((2, library_size))
        true_coefs[0, [1, 4]] = [p[0], -p[1]]
        true_coefs[1, [2, 4]] = [-2 * p[0], p[1]]
        self.true_coefs = true_coefs

    def sample_initial_conditions(self, n=10, seed=None, duration=10, closeness=5):
        if seed is not None:
            np.random.seed(seed)
        return np.random.uniform(0, 1, size=(n, 2))


class Hopf(System):
    def __init__(self, p=(-0.05, 1, 1), library_size=21):
        super(Hopf).__init__()
        self.name = 'Hopf'
        self.initial_condition = (1, 0)
        mu, omega, A = p
        self.forward_fn = lambda t, x: [
            mu * x[0] - omega * x[1] - A * x[0] * (x[0] ** 2 + x[1] ** 2),
            omega * x[0] + mu * x[1] - A * x[1] * (x[0] ** 2 + x[1] ** 2),
        ]
        true_coefs = np.zeros((2, library_size))
        true_coefs[0, [1, 2, 6, 8]] = [mu, -omega, -A, -A]
        true_coefs[1, [1, 2, 7, 9]] = [omega, mu, -A, -A]
        self.true_coefs = true_coefs

    def sample_initial_conditions(self, n=10, seed=None, duration=10, closeness=5):
        if seed is not None:
            np.random.seed(seed)
        theta = np.random.rand(n) * 2 * math.pi
        x0 = np.column_stack([np.cos(theta), np.sin(theta)])
        return (x0.T * np.random.uniform(0.75, 1.25, n)).T


class Rossler(System):
    def __init__(self, p=(0.2, 0.2, 5.7), library_size=56):
        super(Rossler).__init__()
        self.name = 'Rossler'
        self.initial_condition = (5, 3, 0)
        self.forward_fn = lambda t, x: [
            -x[1] - x[2],
            x[0] + p[0] * x[1],
            p[1] + (x[0] - p[2]) * x[2]
        ]
        true_coefs = np.zeros((3, library_size))
        true_coefs[0, [2, 3]] = [-1, -1]
        true_coefs[1, [1, 2]] = [1, p[0]]
        true_coefs[2, [0, 3, 6]] = [p[1], -p[2], 1]
        self.true_coefs = true_coefs

    def correct_invalid_initial_conditions(self, x0s):
        x0s[:, -1] = np.abs(x0s[:, -1])
        return x0s


class Meanfield(System):
    def __init__(self, p=(0.1, 1, -1, 1), library_size=56):
        super(Meanfield).__init__()
        self.name = 'Meanfield'
        mu, omega, A, lambd = p
        self.initial_condition = (mu, mu, 0)
        self.forward_fn = lambda t, x: [
            mu * x[0] - omega * x[1] + A * x[0] * x[2],
            omega * x[0] + mu * x[1] + A * x[1] * x[2],
            lambd * (-x[2] + x[0] ** 2 + x[1] ** 2),
        ]
        true_coefs = np.zeros((3, library_size))
        true_coefs[0, [1, 2, 6]] = [mu, -omega, A]
        true_coefs[1, [1, 2, 8]] = [omega, mu, A]
        true_coefs[2, [3, 4, 7]] = [-lambd, lambd, lambd]
        self.true_coefs = true_coefs


class AtmosphericOscillator(System):
    def __init__(self, p=(0.05, -0.01, 3.0, -2.0, -5.0, 1.1), library_size=56):
        super(AtmosphericOscillator).__init__()
        self.name = 'Atmospheric Oscillator'
        self.initial_condition = (0.2, 0.1,  0.4)
        mu1, mu2, omega, alpha, beta, sigma = p
        self.forward_fn = lambda t, x: [
            mu1 * x[0] + sigma * x[0] * x[1],
            mu2 * x[1] + (omega + alpha * x[1] + beta * x[2]) * x[2] - sigma * x[0] ** 2,
            mu2 * x[2] - (omega + alpha * x[1] + beta * x[2]) * x[1],
        ]
        true_coefs = np.zeros((3, library_size))
        true_coefs[0, [1, 5]] = [mu1, sigma]
        true_coefs[1, [3, 4, 5, 8, 9]] = [omega, -sigma, -mu2, alpha, beta]
        true_coefs[2, [2, 6, 7, 8]] = [-omega, -mu2, -alpha, -beta]
        self.true_coefs = true_coefs


class MHD(System):
    def __init__(self, p=(0, 0), library_size=84):
        super(MHD).__init__()
        self.name = 'MHD'
        self.initial_condition = (1, -1, 0.5, -0.5, -1, 1)
        nu, mu = p
        self.forward_fn = lambda t, x: [
            -2 * nu * x[0] + 4.0 * (x[1] * x[2] - x[4] * x[5]),
            -5 * nu * x[1] - 7.0 * (x[0] * x[2] - x[3] * x[5]),
            -9 * nu * x[2] + 3.0 * (x[0] * x[1] - x[3] * x[4]),
            -2 * mu * x[3] + 2.0 * (x[5] * x[1] - x[2] * x[4]),
            -5 * mu * x[4] + 5.0 * (x[2] * x[3] - x[0] * x[5]),
            -9 * mu * x[5] + 9.0 * (x[4] * x[0] - x[1] * x[3]),
        ]
        true_coefs = np.zeros((6, library_size))
        true_coefs[0, [1, 14, 26]] = [-2 * nu, 4, -4]
        true_coefs[1, [2, 9, 24]] = [-5 * nu, -7, 7]
        true_coefs[2, [3, 8, 23]] = [-9 * nu, 3, -3]
        true_coefs[3, [4, 17, 20]] = [-2 * mu, 2, -2]
        true_coefs[4, [5, 12, 19]] = [-5 * mu, -5, 5]
        true_coefs[5, [6, 11, 15]] = [-9 * mu, 9, -9]
        self.true_coefs = true_coefs

    def sample_initial_conditions(self, n=10, seed=None, duration=10, closeness=5):
        if seed is not None:
            np.random.seed(seed)
        return np.random.uniform(-1.5, 1.5, size=(n, 6))


# Code adapted from https://github.com/E-Renshaw/kuramoto-sivashinsky/blob/master/ksequ.py
class KuramotoSivashinsky(System):
    def __init__(self, N=1024, library_size=19):
        super(KuramotoSivashinsky).__init__()
        self.name = 'Kuramoto-Sivashinsky'
        self.is_pde = True
        self.N = N
        cos = np.cos(np.linspace(0, 2 * math.pi, N))
        sin = np.sin(np.linspace(0, 2 * math.pi, N) * 2) / 2
        self.initial_condition = cos + sin

        true_coefs = np.zeros((1, library_size))
        true_coefs[0, [4, 6, 7]] = [-1, -1, -1]
        self.true_coefs = true_coefs

    def simulate(self, duration=20, dt=0.1, x0=None):
        # Initial condition and grid setup
        N = self.N
        x = np.transpose(np.conj(np.arange(1, N + 1))) / N * 100
        a = -1
        b = 1
        u = self.initial_condition if x0 is None else x0  # np.cos(x/16)*(1+np.sin(x/16))
        v = np.fft.fft(u)
        # scalars for ETDRK4
        h = dt
        k = np.transpose(np.conj(np.concatenate((np.arange(0, N / 2), np.array([0]), np.arange(-N / 2 + 1, 0))))) / 16
        L = k ** 2 - k ** 4
        E = np.exp(h * L)
        E_2 = np.exp(h * L / 2)
        M = 16
        r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
        LR = h * np.transpose(np.repeat([L], M, axis=0)) + np.repeat([r], N, axis=0)
        Q = h * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
        f1 = h * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, axis=1))
        f2 = h * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, axis=1))
        f3 = h * np.real(np.mean((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3, axis=1))
        # main loop
        uu = np.array([u])
        tt = 0
        tmax = duration
        nmax = round(tmax / h)
        g = -0.5j * k
        for n in range(1, nmax + 1):
            t = n * h
            Nv = g * np.fft.fft(np.real(np.fft.ifft(v)) ** 2)
            a = E_2 * v + Q * Nv
            Na = g * np.fft.fft(np.real(np.fft.ifft(a)) ** 2)
            b = E_2 * v + Q * Na
            Nb = g * np.fft.fft(np.real(np.fft.ifft(b)) ** 2)
            c = E_2 * a + Q * (2 * Nb - Nv)
            Nc = g * np.fft.fft(np.real(np.fft.ifft(c)) ** 2)
            v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
            u = np.real(np.fft.ifft(v))
            uu = np.append(uu, np.array([u]), axis=0)
            tt = np.hstack((tt, t))

        u = uu.T.reshape(uu.shape[1], uu.shape[0], 1)
        return u

    def sample_initial_conditions(self, n=10, seed=None, duration=100, closeness=5):
        if seed is not None:
            np.random.seed(seed)
        x0s = []
        for _ in range(n):
            r = np.random.rand(2)
            cos = np.cos(r[0] + np.linspace(0, 2 * math.pi, self.N))
            sin = np.sin(np.linspace(0, 2 * math.pi, self.N) * (4 * r[1]))
            init = (cos + sin) / np.max(np.abs(cos + sin))
            x0s.append(init)
        return self.correct_invalid_initial_conditions(x0s)

    def make_mesh_grid(self, duration, dt):
        t = np.linspace(0, duration, round(duration/dt) + 1)
        N = self.N
        x = np.transpose(np.conj(np.arange(1, N + 1))) / N * 100
        X, T = np.meshgrid(x, t)
        XT = np.asarray([X, T]).T
        return XT


# Code adapted from pySINDy example 12
class ReactionDiffusion(System):
    def __init__(self, grid_size=64, n_spirals=1, domain_size=20, library_size=109):
        super(ReactionDiffusion).__init__()
        self.name = 'Reaction Diffusion'
        self.is_pde = True
        self.n = grid_size
        self.n_spirals = n_spirals
        self.domain_size = domain_size
        true_coefs = np.zeros((2, library_size))
        true_coefs[0, [0, 4, 5, 7, 8, 11, 17]] = [1, -1, 1, 1, -1, 0.1, 0.1]
        true_coefs[1, [1, 4, 5, 7, 8, 12, 18]] = [1, -1, -1, -1, -1, 0.1, 0.1]
        self.true_coefs=true_coefs

    def simulate(self, duration=10, dt=0.1, x0=None):
        def forward_fn(t, uvt, K22, d1, d2, beta, n, N):
            ut = np.reshape(uvt[:N], (n, n))
            vt = np.reshape(uvt[N: 2 * N], (n, n))
            u = np.real(ifft2(ut))
            v = np.real(ifft2(vt))
            u3 = u ** 3
            v3 = v ** 3
            u2v = (u ** 2) * v
            uv2 = u * (v ** 2)
            utrhs = np.reshape((fft2(u - u3 - uv2 + beta * u2v + beta * v3)), (N, 1))
            vtrhs = np.reshape((fft2(v - u2v - v3 - beta * u3 - beta * uv2)), (N, 1))
            uvt_reshaped = np.reshape(uvt, (len(uvt), 1))
            uvt_updated = np.squeeze(
                np.vstack(
                    (-d1 * K22 * uvt_reshaped[:N] + utrhs,
                     -d2 * K22 * uvt_reshaped[N:] + vtrhs)
                )
            )
            return uvt_updated

        # Generate the data
        t = np.linspace(0, duration, round(duration/dt))
        d1 = 0.1
        d2 = 0.1
        beta = 1.0
        L = self.domain_size  # Domain size in X and Y directions
        n = self.n  # Number of spatial points in each direction
        N = n * n
        x_uniform = np.linspace(-L / 2, L / 2, n + 1)
        x = x_uniform[:n]
        y = x_uniform[:n]
        n2 = int(n / 2)
        # Define Fourier wavevectors (kx, ky)
        kx = (2 * np.pi / L) * np.hstack((np.linspace(0, n2 - 1, n2),
                                          np.linspace(-n2, -1, n2)))
        ky = kx
        # Get 2D meshes in (x, y) and (kx, ky)
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX ** 2 + KY ** 2
        K22 = np.reshape(K2, (N, 1))

        # define our solution vectors
        u = np.zeros((len(x), len(y), len(t)))
        v = np.zeros((len(x), len(y), len(t)))

        # Initial conditions
        initial_uv = x0 if x0 is not None else self.sample_initial_conditions(1, random=False)[0]
        u[:, :, 0] = initial_uv[0]
        v[:, :, 0] = initial_uv[1]

        # uvt is the solution vector in Fourier space, so below
        # we are initializing the 2D FFT of the initial condition, uvt0
        uvt0 = np.squeeze(
            np.hstack(
                (np.reshape(fft2(u[:, :, 0]), (1, N)),
                 np.reshape(fft2(v[:, :, 0]), (1, N)))
            )
        )

        # Solve the PDE in the Fourier space, where it reduces to system of ODEs
        uvsol = solve_ivp(
            forward_fn,
            (t[0], t[-1]),
            y0=uvt0,
            t_eval=t,
            args=(K22, d1, d2, beta, n, N)
        )
        uvsol = uvsol.y

        # Reshape things and ifft back into (x, y, t) space from (kx, ky, t) space
        for j in range(len(t)):
            ut = np.reshape(uvsol[:N, j], (n, n))
            vt = np.reshape(uvsol[N:, j], (n, n))
            u[:, :, j] = np.real(ifft2(ut))
            v[:, :, j] = np.real(ifft2(vt))

        u_sol = u
        v_sol = v
        u = np.zeros((n, n, len(t), 2))
        u[:, :, :, 0] = u_sol
        u[:, :, :, 1] = v_sol

        return u

    def sample_initial_conditions(self, n=10, seed=None, duration=10, closeness=5, random=True):
        if seed is not None:
            np.random.seed(seed)
        L = self.domain_size  # Domain size in X and Y directions
        grid_size = self.n  # Number of spatial points in each direction
        x_uniform = np.linspace(-L / 2, L / 2, grid_size + 1)
        x = x_uniform[:grid_size]
        y = x_uniform[:grid_size]
        # Get 2D meshes in (x, y) and (kx, ky)
        X, Y = np.meshgrid(x, y)
        m = self.n_spirals

        x0s = []
        for _ in range(n):
            # Initial conditions
            spacing = np.random.uniform(.95, 1.05) if random else 1
            angle_offset = np.random.uniform(0, 2*math.pi) if random else 0

            u_0 = np.tanh(np.sqrt(X ** 2 + Y ** 2)) * np.cos(
                (m * np.angle(X + 1j * Y) + angle_offset)
                - (np.sqrt(X ** 2 + Y ** 2) * spacing)
            )
            v_0 = np.tanh(np.sqrt(X ** 2 + Y ** 2)) * np.sin(
                (m * np.angle(X + 1j * Y) + angle_offset)
                - (np.sqrt(X ** 2 + Y ** 2) * spacing)
            )
            x0s.append((u_0, v_0))
        return x0s

    def make_mesh_grid(self, duration=10, dt=0.1):
        t = np.linspace(0, duration, round(duration/dt))
        L = self.domain_size
        n = self.n
        x_uniform = np.linspace(-L / 2, L / 2, n + 1)
        x = x_uniform[:n]
        y = x_uniform[:n]
        X, Y, T = np.meshgrid(x, y, t, indexing='ij')
        XYT = np.transpose([X, Y, T], [1, 2, 3, 0])
        return XYT


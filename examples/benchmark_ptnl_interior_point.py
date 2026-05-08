import torch
import pytorch_nonlinear as ptnl

from benchcaddy import Sweep, observe


DTYPE = torch.float64


def objective(state, params=None):
    x0, x1, x2 = state.unbind()
    return (
        torch.exp(x0 - 0.8)
        + 0.4 * (x1 - 0.2).pow(4)
        + torch.sin(x2 + 0.3).pow(2)
        + 0.15 * x0 * x2
    )


def equality_constraint(state, params=None):
    return state[0] + state[1] - 1.0


def inequality_constraint(state, params=None):
    x0, x1, x2 = state.unbind()
    return x0 * x2 + 0.25 * x1.pow(2) - 0.45  # PTNL expects g(x) <= 0


problem = ptnl.ConstrainedNLPProblem(
    objective=objective,
    constraints=[
        ptnl.EqualityConstraint(equality_constraint),
        ptnl.InequalityConstraint(inequality_constraint),
    ],
    bounds=ptnl.Bounds(
        lower=torch.tensor([0.05, 0.05, -0.8], dtype=DTYPE),
        upper=torch.tensor([1.25, 0.95, 0.9], dtype=DTYPE),
    ),
)

x0 = torch.tensor([0.55, 0.45, 0.10], dtype=DTYPE)
config = ptnl.SolverConfig(method="interior_point", max_iter=80, tol=1e-8)


@observe("ptnl_interior_point")
def solve_once():
    result = ptnl.solve(problem, x0=x0, config=config)
    if not result.success:
        raise RuntimeError(result.summary())
    return result


def main():
    Sweep(
        target=solve_once,
        params={},
        suite_name="ptnl-interior-point",
        samples=5,
        warmup_iterations=1,
    ).run()


if __name__ == "__main__":
    main()

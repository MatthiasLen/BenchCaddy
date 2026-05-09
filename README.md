We all tell ourselves we’re going to use Scalene,PyInstrument or TorchProfile - tools that produce traces so complex and beautiful they belong in a modern art gallery. But let’s be real: most days, "benchmarking" is just us sprinkling time.time() across our code like frantic seasoning on a failing dish. You’re staring at the terminal, trying to remember if the last run was actually faster or if you just happen to be in a better mood, only to realize you’ve already lost the thread. *"Wait, when did I change the naming convention of the log files? Is 'results_v2_final' newer than 'results_new_test'?"*


**BenchCaddy** is the humble sidekick for those of us living in that chaotic middle ground. It replaces "vibes-based" timing with stabilized sweeps and environment metadata, tucking everything into a neat database before your brain can wander. It won’t map your entire soul, but it will save you from your own memory and provide a summary clean enough to make you look like the organized professional your friends think you are. No traces to decipher, no lost logs, and no more gaslighting yourself - just actual proof your code is getting faster.

# Something missing ?

BenchCaddy is intentionally lean - a sidekick, not a supervisor. I built it to curb my own "log-file-chaos," but I’m curious how you manage yours. If you’ve got a feature idea, a bug that’s getting on your nerves, or a suggestion for an export format that actually belongs in this decade, open an issue. I’m not trying to build a bloated enterprise behemoth; I just want this to be the best way to track performance without ever having to name a file timings_final_v4_fixed_REALLY.log again.


## Example: benchmark a PTNL interior-point solve

Install the solver package first:

```bash
pip install ptnl
```

Then run this example to benchmark a compact nonlinear program with mixed
constraints:

```python
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


Sweep(
    target=solve_once,
    params={},
    suite_name="ptnl-interior-point",
    samples=5,
    warmup_iterations=1,
).run()
```

BenchCaddy writes the samples, medians, and environment metadata to
`benchcaddy.db` in the current working directory.

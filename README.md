We all tell ourselves we’re going to use Scalene,PyInstrument or TorchProfile - tools that produce traces so complex and beautiful they belong in a modern art gallery. But let’s be real: most days, "benchmarking" is just us sprinkling time.time() across our code like frantic seasoning on a failing dish. You’re staring at the terminal, trying to remember if the last run was actually faster or if you just happen to be in a better mood, only to realize you’ve already lost the thread. *"Wait, when did I change the naming convention of the log files? Is 'results_v2_final' newer than 'results_new_test'?"*

**BenchCaddy** is the humble sidekick for those of us living in that chaotic middle ground. It replaces "vibes-based" timing with stabilized sweeps and environment metadata, tucking everything into a neat SQLite database before your brain can wander. It won’t map your entire soul, but it will save you from your own memory and provide a CLI summary clean enough to make you look like the organized professional your friends think you are. No traces to decipher, no lost logs, and no more gaslighting yourself - just actual proof your code is getting faster.

```python
from benchcaddy import Sweep, observe

@observe("heavy_computation")
def my_function():
    # Your complex logic, PyTorch solver, or life's work here
    return sum(i * i for i in range(10**6))

# The "Sweep" does the heavy lifting
Sweep(target=my_function, samples=5, warmup_iterations=1).run()
```


# Something missing ?

BenchCaddy is intentionally lean—a sidekick, not a supervisor. I built it to curb my own "log-file-chaos," but I’m curious how you manage yours. If you’ve got a feature idea, a bug that’s getting on your nerves, or a suggestion for an export format that actually belongs in this decade, open an issue. I’m not trying to build a bloated enterprise behemoth; I just want this to be the best way to track performance without ever having to name a file timings_final_v4_fixed_REALLY.log again.
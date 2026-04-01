<div align="center" style="height:250px;width:400px">
<img src="assets/logo.png" alt="logo"></img>
</div>

# paramore

paramore is a package to implement parametric statistical models for High Energy Physics, built on JAX and [evermore](https://github.com/pfackeldey/evermore/tree/main) primitives.

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/your-org/paramore.git
cd paramore
uv pip install -e .
```

## Example

Minimal example: fit a Gaussian PDF using `evermore` parameters and `optimistix`

```

import jax
import jax.numpy as jnp
import evermore as evm
import optimistix as optx
from evermore.parameters import filter as evm_filter
from flax import nnx
import paramore as pm

jax.config.update("jax_enable_x64", True)

lower, upper = 110.0, 140.0

# Sample toy data from a Gaussian with known parameters
true_pdf = pm.Gaussian(mu=125.0, sigma=2.0, lower=lower, upper=upper)
key = jax.random.PRNGKey(42)
data = true_pdf.sample(key, n_events=1000)

# Define free parameters
class Params(nnx.Pytree):
    def __init__(self, mu: evm.Parameter, sigma: evm.Parameter):
        self.mu = mu
        self.sigma = sigma

params = Params(
    mu=evm.Parameter(value=120.0, name="mu"),
    sigma=evm.Parameter(value=3.0, name="sigma"),
)

# Split into differentiable and static parts
graphdef, diffable, static = nnx.split(params, evm_filter.is_dynamic_parameter, ...)

# Define NLL
def nll(diffable, args):
    graphdef, static, data = args
    p = nnx.merge(graphdef, diffable, static)
    pdf = pm.Gaussian(mu=p.mu.get_value(), sigma=p.sigma.get_value(), lower=lower, upper=upper)
    sum_pdf = pm.SumPDF([pdf], [jnp.array(float(len(data)))], lower=lower, upper=upper)
    return pm.create_nll(p, sum_pdf, data)

# Fit with BFGS
solver = optx.BFGS(rtol=1e-6, atol=1e-6)
result = optx.minimise(nll, solver, diffable, args=(graphdef, static, data))

# Reconstruct fitted parameters
fitted = nnx.merge(graphdef, result.value, static)
print(f"mu    = {float(fitted.mu.get_value()):.3f}  (true: 125.0)")
print(f"sigma = {float(fitted.sigma.get_value()):.3f}  (true: 2.0)")
```

## Documentation

## Contributing

## License

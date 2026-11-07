#!/usr/bin/env python3
"""Independent dependency-rich science chains; CPU references and shared evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import helper_chain_cases as short  # noqa: E402

Stage = short.Stage
COMMON = """You are implementing one numerical stage inside an existing scientific pipeline.
The functions in the supplied context are already available, with the exact definitions
shown below. Reuse them where the stage contract requests them; their conventions are
part of the interface, not suggestions. The returned program must contain imports and
one complete solve function. Do not copy or redefine the supplied context functions or
return a partially evaluated expression. There is no interactive input, file access,
random sampling, plotting, training or hidden state in this task.

All real input arrays are finite float64 NumPy arrays; integer index arrays are explicitly
identified. Dimensions may vary between calls, so infer them from the inputs. Vectors
have shape (n,), not (n,1). Matrix products use conventional column-vector mathematics,
while batches are stored as rows only when explicitly stated. Return a new float64
ndarray of exactly the specified shape even when one dimension equals one. Never
squeeze dimensions, return a tuple, or mutate any input or supplied helper result in
place. A scalar mathematical quantity must still occupy its specified array position.
The inputs satisfy the stated positivity, rank and index conditions; you do not need
validation code or numerical regularization beyond that explicitly requested.

The stage is deterministic. Preserve cross terms and the ordering specified by the
contract. Do not replace a covariance by its diagonal, normalize an already normalized
quantity again, round intermediate values, clip signs, or add a stabilizing identity
unless the mathematics below calls for it. Use ordinary double precision. The checks
compare numerical values with a tolerance, not expression spelling or implementation
style. Valid output shape alone is insufficient: the result must implement the whole
stated transformation. For the downstream stage, use the available helper's returned
mathematical quantity rather than silently substituting a simpler approximation.
"""


@dataclass(frozen=True)
class Chain(short.Chain):
    context: str
    support: str

    def prompt(self, stage: int, helper: str | None = None) -> str:
        item = self.helper if stage == 0 else self.downstream
        available = (
            ""
            if stage == 0
            else "\nAvailable stage helper (do not redefine):\n" + str(helper)
        )
        return (
            COMMON
            + "\nScientific context and conventions:\n"
            + self.context
            + "\nProvided context functions:\n"
            + self.support
            + available
            + f"\nImplement solve({item.parameters}).\n"
            + item.requirement
        )

    def row(self, stage: int) -> dict[str, Any]:
        row = super().row(stage)
        row["id"] = f"long-{self.family}-{stage + 1}"
        row["depends_on"] = [] if stage == 0 else [f"long-{self.family}-1"]
        row["provided_context_sha256"] = short.digest(self.support.encode())
        row["answer_unit"] = (
            "complete solve function and imports; provided context and stage helper excluded"
        )
        return row


def catalog() -> list[Chain]:
    return [
        Chain(
            "weighted-constraint",
            "train",
            "science_long_helper",
            0,
            Stage(
                "weighted-constraint-helper",
                "a, b, variance, offset, penalty, strength",
                "Use whiten_observations and prior_terms. Return shape (p,p+1): the first p columns are "
                "N=a.T@diag(1/variance)@a + strength*penalty.T@penalty; final column is "
                "r=a.T@diag(1/variance)@(b-offset). The offset is subtracted before whitening. "
                "The penalty is applied once with strength, not its square. This stage assembles the "
                "unconstrained system only; do not solve it or append any diagnostic values.",
                "",
                "wa, wb = whiten_observations(a,b,variance,offset)\nn = wa.T@wa + prior_terms(penalty,strength)\nr = wa.T@wb\nreturn np.column_stack((n,r))",
                (
                    (
                        "precision-as-root",
                        "wa.T@wa",
                        "a.T@(a/np.sqrt(variance)[:,None])",
                    ),
                    (
                        "double-prior",
                        "prior_terms(penalty,strength)",
                        "2*prior_terms(penalty,strength)",
                    ),
                ),
            ),
            Stage(
                "weighted-constraint-composition",
                "a, b, variance, offset, penalty, strength, constraint, target, query",
                "Call helper(a,b,variance,offset,penalty,strength), then solve_spd. Let C=N^{-1}, x0=C@r, "
                "u=C@constraint, d=constraint@u. Enforce the exact equality constraint@x=target using "
                "x=x0+u*(target-constraint@x0)/d. Its constrained covariance is C-u[:,None]*u[None,:]/d. "
                "Return (m,2): column0=query@x; column1 is each query row's quadratic form in that "
                "constrained covariance. Do not add observation noise, offset or target to predictions. "
                "The equality is exact, not another ridge penalty.",
                "",
                "z = helper(a,b,variance,offset,penalty,strength)\nn, r = z[:,:-1], z[:,-1]\nc = solve_spd(n,np.eye(n.shape[0]))\nx = solve_spd(n,r)\nu = c@constraint\nd = constraint@u\nx = x+u*(target-constraint@x)/d\nc = c-np.outer(u,u)/d\nreturn np.column_stack((query@x,np.einsum('ij,jk,ik->i',query,c,query)))",
                (
                    (
                        "unconstrained-covariance",
                        "c = c-np.outer(u,u)/d",
                        "c = c.copy()",
                    ),
                    ("reversed-equality", "target-constraint@x", "constraint@x-target"),
                ),
            ),
            """A sensor model b=offset+a@x+noise has n observations and p unknown coefficients. The
known offset is length n. variance contains strictly positive noise VARIANCES, not standard
deviations or precisions. a has full column rank and shape (n,p), with n>=p; penalty is
(q,p), strength is a positive dimensionless multiplier, and all quantities use a consistent
fixed unit system. The quadratic objective is sum((a@x+offset-b)**2/variance) plus
strength*||penalty@x||**2. No intercept is implicit and neither observations nor columns
are to be centered. The normal matrix is SPD. In the downstream call, constraint is a
nonzero vector (p,), target is a scalar with the units of constraint@x, and query is (m,p).
The constrained covariance is the inverse normal matrix restricted to the equality plane;
it is not an unbiased residual variance estimate and has no division by n-p. A query
parallel to the constraint has zero constrained variance up to rounding. The supplied SPD
solver accepts vector or matrix right-hand sides and does not modify them. The whitening
function divides rows by standard deviations; applying variance again would change the
objective. The prior helper returns a precision contribution in coefficient coordinates.
The packed matrix convention always puts the RHS in the final column, including p=1.
The downstream result contains uncertainty in the coefficient solution, not random future
sensor noise. This distinction matters even if query reuses rows from the design matrix.""",
            """import numpy as np
from scipy.linalg import solve as scipy_solve

def whiten_observations(a,b,variance,offset):
    root = np.sqrt(variance)
    return a/root[:,None], (b-offset)/root

def prior_terms(penalty,strength):
    return strength*(penalty.T@penalty)

def solve_spd(matrix,rhs):
    return scipy_solve(matrix,rhs,assume_a='pos')
""",
        ),
        Chain(
            "conditional-covariance",
            "train",
            "science_long_helper",
            1,
            Stage(
                "conditional-covariance-helper",
                "mean, covariance, observed, value, target",
                "Use partition_gaussian and solve_spd. Return (r,r+1), first r columns the conditional "
                "target covariance T-B@O^{-1}@B.T, final column the conditional target mean "
                "mt+B@O^{-1}@(value-mo). target and observed order must be preserved exactly. "
                "Condition on all observed values jointly; no independent-coordinate approximation.",
                "",
                "mt, mo, t, b, o = partition_gaussian(mean,covariance,observed,target)\nm = mt+b@solve_spd(o,value-mo)\nc = t-b@solve_spd(o,b.T)\nreturn pack_moments(c,m)",
                (
                    (
                        "wrong-schur-sign",
                        "t-b@solve_spd(o,b.T)",
                        "t+b@solve_spd(o,b.T)",
                    ),
                    ("ignore-observed-mean", "value-mo", "value"),
                ),
            ),
            Stage(
                "conditional-covariance-composition",
                "mean, covariance, observed, value, target, query, bias, noise_variance",
                "Call helper and use pack_moments. Define z=query@X_target+bias+independent noise, "
                "after conditioning. Return (m,m+1): full covariance query@C@query.T + diag(noise_variance) "
                "in the first m columns and query@conditional_mean+bias in the final column. "
                "bias has no contribution to covariance; noise is added after projection, exactly once.",
                "",
                "z = helper(mean,covariance,observed,value,target)\nc, mu = z[:,:-1], z[:,-1]\nreturn pack_moments(query@c@query.T+np.diag(noise_variance),query@mu+bias)",
                (
                    (
                        "drop-cross-covariance",
                        "query@c@query.T",
                        "query@np.diag(np.diag(c))@query.T",
                    ),
                    (
                        "noise-as-deviation",
                        "np.diag(noise_variance)",
                        "np.diag(noise_variance**2)",
                    ),
                ),
            ),
            """A jointly Gaussian real vector has mean (n,) and SPD covariance (n,n). observed and
target are disjoint, nonempty arrays of unique valid integer indices. They need not cover
all n coordinates, need not be sorted, and their order is meaningful: value[j] belongs to
observed[j], and output target coordinate j belongs to target[j]. The model is conditioned
on exact observed values, without measurement noise in that conditioning operation. Any
coordinates in neither set are marginalized, not conditioned to zero. All covariance
entries use squared units of the corresponding variables, and off-diagonal entries can
be negative. The Schur complement is SPD for these nonempty disjoint subsets. Do not
sort indices, take absolute values, square covariance entries, or replace the observed
block by its diagonal. The partition helper returns copies in the requested orders and
uses NumPy outer indexing to select whole matrix blocks, never pairwise indexing.
For the downstream affine observation, query has shape (m,r), bias and noise_variance
have shape (m,), and all noise variances are strictly positive. New noise coordinates are
mutually independent and independent of the original Gaussian vector. query rows can
share target variables, so output coordinates generally remain correlated. Return the
entire covariance and mean, even if m=1; an (m,m+1) array has no separate scalar return.
Changing observed values changes the conditional mean but not its covariance. Adding a
constant to all original means and the corresponding observed values is not generally
an invariant unless the target shift and downstream bias are accounted for explicitly.
The supplied packing helper appends mean as one column without changing covariance.""",
            """import numpy as np
from scipy.linalg import solve as scipy_solve

def partition_gaussian(mean,covariance,observed,target):
    return (mean[target],mean[observed],covariance[np.ix_(target,target)],
            covariance[np.ix_(target,observed)],covariance[np.ix_(observed,observed)])

def solve_spd(matrix,rhs):
    return scipy_solve(matrix,rhs,assume_a='pos')

def pack_moments(covariance,mean):
    return np.column_stack((covariance,mean))
""",
        ),
        Chain(
            "affine-state-history",
            "train",
            "science_long_helper",
            0,
            Stage(
                "affine-state-history-helper",
                "mean, covariance, dt, decay, drive, noise_rate",
                "Use transition_parts and pack_moments for ONE step. Return (2d,2d+1), covariance "
                "F@covariance@F.T+Q followed by mean F@mean+u. Q is already scaled by dt. "
                "Position uses the previous velocity; drive affects the new velocity only.",
                "",
                "f, u, q = transition_parts(dt,decay,drive,noise_rate)\nreturn pack_moments(f@covariance@f.T+q,f@mean+u)",
                (
                    ("double-time-noise", "+q,", "+dt*q,"),
                    ("drive-to-position", "f@mean+u", "f@mean+np.roll(u,len(decay))"),
                ),
            ),
            Stage(
                "affine-state-history-composition",
                "mean, covariance, durations, decays, drives, noise_rates, observation, bias, measurement_covariance",
                "Apply helper once per interval in chronological order, carrying BOTH returned mean "
                "and covariance. Then call project_observation on the final moments. Return its "
                "(m,m+1) covariance-plus-mean array. Add measurement_covariance only at the final "
                "observation, not at each dynamics step; do not return the initial state or a history.",
                "",
                "mu, c = mean.copy(), covariance.copy()\nfor dt, decay, drive, noise in zip(durations,decays,drives,noise_rates):\n    z = helper(mu,c,float(dt),decay,drive,noise)\n    c, mu = z[:,:-1], z[:,-1]\nreturn project_observation(c,mu,observation,bias,measurement_covariance)",
                (
                    (
                        "reset-covariance",
                        "helper(mu,c,float(dt)",
                        "helper(mu,covariance,float(dt)",
                    ),
                    (
                        "reverse-time",
                        "zip(durations,decays,drives,noise_rates)",
                        "zip(durations[::-1],decays[::-1],drives[::-1],noise_rates[::-1])",
                    ),
                ),
            ),
            """The state vector is ordered [positions_0..positions_(d-1), velocities_0..velocities_(d-1)].
A discrete interval of duration dt seconds is defined exactly by p_next=p+dt*v and
v_next=decay*v+dt*drive, with elementwise multiplication in the latter expression.
decay is a dimensionless vector (d,) whose entries lie strictly between zero and one;
it is already the interval multiplier, not a per-second rate to exponentiate. drive is
(d,) in velocity units per second. This is the stipulated discrete update, not an ODE
integrator or a constant-acceleration position formula: do not add dt**2*drive/2 to
position. mean is (2d,), covariance is SPD (2d,2d), and noise_rate is SPD in full state
coordinates per second. The additive process covariance is dt*noise_rate, including
its position-velocity cross terms. Process noise is independent between intervals and
independent of the entering state. A deterministic drive does not add covariance.
The downstream durations array has length L>=2, decays and drives are (L,d), and
noise_rates is (L,2d,2d). Each row/block describes its own interval, and the order cannot
be changed because affine transitions generally do not commute. observation is (m,2d),
bias is (m,), and measurement_covariance is SPD (m,m), independent of process noise.
The observation helper projects the full final state distribution and adds this final
measurement covariance. It packs covariance in the first m columns and mean in the
last; no time axis is retained in the output. All supplied helpers allocate new results,
so retaining views of returned packed moments is permitted without mutating inputs.
Dimensions and L vary; no fixed two-dimensional state or fixed interval count is assumed.""",
            """import numpy as np

def transition_parts(dt,decay,drive,noise_rate):
    d = len(decay)
    f = np.block([[np.eye(d),dt*np.eye(d)],[np.zeros((d,d)),np.diag(decay)]])
    u = np.concatenate((np.zeros(d),dt*drive))
    return f,u,dt*noise_rate

def pack_moments(covariance,mean):
    return np.column_stack((covariance,mean))

def project_observation(covariance,mean,observation,bias,measurement_covariance):
    return pack_moments(observation@covariance@observation.T+measurement_covariance,
                        observation@mean+bias)
""",
        ),
        Chain(
            "affine-mixture-moments",
            "train",
            "science_long_helper",
            1,
            Stage(
                "affine-mixture-moments-helper",
                "log_prior, log_evidence",
                "Use combine_log_weights and normalize_log_rows. Return (n,k) posterior component "
                "probabilities. log_prior is shared over observations; each log_evidence row has its "
                "own normalization. Preserve zeros implied by underflow; do not normalize columns.",
                "",
                "z = combine_log_weights(log_prior,log_evidence)\nreturn normalize_log_rows(z)",
                (
                    (
                        "omit-prior",
                        "combine_log_weights(log_prior,log_evidence)",
                        "log_evidence.copy()",
                    ),
                    (
                        "column-normalize",
                        "normalize_log_rows(z)",
                        "normalize_log_rows(z.T).T",
                    ),
                ),
            ),
            Stage(
                "affine-mixture-moments-composition",
                "log_prior, log_evidence, means, covariances, transform, shift",
                "Call helper and affine_components. For each observation, combine transformed component "
                "means and covariances using the law of total covariance, including between-component "
                "mean variation. Return (n,m+m*m): first m entries of each row are the mixture mean; "
                "remaining m*m entries are the full covariance flattened in C (row-major) order.",
                "",
                "p = helper(log_prior,log_evidence)\nmu, cov = affine_components(means,covariances,transform,shift)\nanswer = []\nfor weights in p:\n    center = weights@mu\n    delta = mu-center\n    c = np.einsum('k,kij->ij',weights,cov)+np.einsum('k,ki,kj->ij',weights,delta,delta)\n    answer.append(np.concatenate((center,c.reshape(-1))))\nreturn np.asarray(answer,dtype=np.float64)",
                (
                    (
                        "omit-between-component",
                        "+np.einsum('k,ki,kj->ij',weights,delta,delta)",
                        "",
                    ),
                    ("raw-second-moment", "delta = mu-center", "delta = mu.copy()"),
                ),
            ),
            """A mixture model has k>=2 latent components and n independent observations. log_prior
has shape (k,) and contains finite logarithms of unnormalized positive prior weights;
log_evidence has shape (n,k) and contains finite log likelihoods. Common additive log
offsets may be large, so normalize through logsumexp rather than exponentiating raw
values. The same prior applies to every observation. The posterior for one observation
must not depend on another row. This stage does not estimate the prior or average
posteriors over observations. The provided row normalizer handles its final axis.
Component means have shape (k,d), covariances (k,d,d), and each covariance is SPD.
The affine transform maps column vectors x to transform@x+shift, with transform (m,d)
and shift (m,). The same transform acts on all components. The transformed mean is
stored as a row in the helper output, while its covariance is a full (m,m) matrix.
The shift changes means but not within-component covariance. Across components,
the law of total covariance adds the weighted covariance of their transformed means;
a weighted average of within-component covariances alone is incomplete. Conversely,
the raw second moment is not a central covariance. Output order is mean coordinates,
then covariance row0, row1, and so on; do not pack only its diagonal or triangle.
The output always has n rows even for n=1 and m+m*m columns even for m=1. Every
input observation gets its own mixture moments, although component parameters are
shared. No observation noise, sample-covariance correction, posterior resampling or
MAP-component selection is part of this deterministic calculation. Use the full soft
posterior probabilities rather than rounding them to a single selected component.""",
            """import numpy as np
from scipy.special import logsumexp

def combine_log_weights(log_prior,log_evidence):
    return log_evidence+log_prior[None,:]

def normalize_log_rows(z):
    return np.exp(z-logsumexp(z,axis=1,keepdims=True))

def affine_components(means,covariances,transform,shift):
    mu = means@transform.T+shift
    cov = np.einsum('ai,kij,bj->kab',transform,covariances,transform)
    return mu,cov
""",
        ),
        Chain(
            "mass-spectral-response",
            "transfer",
            "science_long_helper",
            0,
            Stage(
                "mass-spectral-response-helper",
                "stiffness, mass, time",
                "Use mass_modes and mode_decay. Return the (n,n) propagator P=V@diag(exp(-time*lambda))@V.T@mass. "
                "This maps an initial state vector to its state at time; V is mass-orthonormal, "
                "not Euclidean-orthonormal. Do not return eigenvectors or a symmetric surrogate.",
                "",
                "values, vectors = mass_modes(stiffness,mass)\nreturn (vectors*mode_decay(values,time)[None,:])@vectors.T@mass",
                (
                    ("omit-mass", "@vectors.T@mass", "@vectors.T"),
                    (
                        "double-decay-time",
                        "mode_decay(values,time)",
                        "mode_decay(values,2*time)",
                    ),
                ),
            ),
            Stage(
                "mass-spectral-response-composition",
                "stiffness, mass, durations, initial, observation",
                "Apply helper for every duration in order to initial. Return (m+1,1): first m rows "
                "are observation@final_state as a COLUMN; final row is 0.5*final_state.T@mass@final_state. "
                "No stiffness factor belongs in this specified mass energy. Use report_state.",
                "",
                "x = initial.copy()\nfor duration in durations:\n    x = helper(stiffness,mass,float(duration))@x\nreturn report_state(x,mass,observation)",
                (
                    (
                        "only-last-interval",
                        "x = helper(stiffness,mass,float(duration))@x",
                        "x = helper(stiffness,mass,float(duration))@initial",
                    ),
                    (
                        "wrong-energy-metric",
                        "report_state(x,mass,observation)",
                        "report_state(x,stiffness,observation)",
                    ),
                ),
            ),
            """A finite-dimensional diffusion system is mass@dx/dt=-stiffness@x. Both mass and
stiffness are SPD real (n,n) matrices; they need not commute and mass is generally
not diagonal. time and each duration are positive in seconds, while generalized
eigenvalues have units of inverse seconds. mass_modes returns eigenvalues lambda and
columns V satisfying stiffness@V=mass@V@diag(lambda), V.T@mass@V=I. The usual
Euclidean eigenvector transpose is therefore not the inverse modal transform; the
mass factor in V.T@mass is required when converting a state to modal coefficients.
Eigenvector signs and degenerate eigenspace bases are not prescribed, but the resulting
propagator is basis-independent. It is generally nonsymmetric in Euclidean coordinates,
even though it is self-adjoint under the mass inner product. Do not symmetrize it.
The physical state and observation use the original coordinate basis, not the modal
basis. The exponential multiplier is exp(-lambda*time) and not exp(-time/lambda).
The downstream durations vector contains L>=2 consecutive intervals with identical
mass and stiffness. initial is (n,) and observation (m,n). Carry the evolved state
between intervals rather than restarting it. The reported energy is explicitly the
mass norm squared divided by two, not the stiffness energy or Euclidean norm.
Return a two-dimensional column array of shape (m+1,1), including when m=1; a vector
of length m+1 is a different interface. The energy occupies the last row after the m
linear observations. There is no forcing, boundary term, normalization of the state,
noise, or numerical time integration approximation. The helpers use an exact dense
spectral exponential for the stipulated linear system, up to floating-point error.""",
            """import numpy as np
from scipy.linalg import eigh

def mass_modes(stiffness,mass):
    return eigh(stiffness,mass)

def mode_decay(values,time):
    return np.exp(-values*time)

def report_state(state,mass,observation):
    return np.concatenate((observation@state,[0.5*state@mass@state]))[:,None]
""",
        ),
        Chain(
            "rate-reward-history",
            "transfer",
            "science_long_helper",
            1,
            Stage(
                "rate-reward-history-helper",
                "rates, duration",
                "Use rate_generator and transition_matrix. Return (n,n) row transition probabilities "
                "exp(duration*Q). Q is a row generator. Ignore the input diagonal of rates as specified; "
                "off-diagonal rates are used once and are not row-normalized.",
                "",
                "q = rate_generator(rates)\nreturn transition_matrix(q,duration)",
                (
                    (
                        "transpose-generator",
                        "transition_matrix(q,duration)",
                        "transition_matrix(q.T,duration)",
                    ),
                    (
                        "double-duration",
                        "transition_matrix(q,duration)",
                        "transition_matrix(q,2*duration)",
                    ),
                ),
            ),
            Stage(
                "rate-reward-history-composition",
                "rates, durations, initial, reward",
                "Start with row distribution initial. For each interval, get P=helper(rates,dt), "
                "advance p_next=p@P, and accumulate dt*(p@reward+p_next@reward)/2. Return float64 "
                "shape (1,2): final expected reward then the accumulated TRAPEZOIDAL reward integral. "
                "Use reward_pair; this defined endpoint quadrature is not an exact continuous-time integral.",
                "",
                "p = initial.copy()\ntotal = 0.0\nfor dt in durations:\n    next_p = p@helper(rates,float(dt))\n    total += float(dt)*0.5*(p@reward+next_p@reward)\n    p = next_p\nreturn reward_pair(p,reward,total)",
                (
                    (
                        "right-endpoint-only",
                        "0.5*(p@reward+next_p@reward)",
                        "next_p@reward",
                    ),
                    ("missing-duration", "float(dt)*0.5", "0.5"),
                ),
            ),
            """A finite-state continuous-time Markov model is stored using row probability vectors.
rates is a finite nonnegative (n,n) array with strictly positive off-diagonal entries;
its diagonal is ignored. Off-diagonal entries have units of inverse seconds and are
transition intensities, not normalized probabilities or counts. The generator Q has
these entries off the diagonal and Q[i,i]=-sum(j!=i,rates[i,j]), so each row sums to zero.
The supplied generator copies the input before clearing its diagonal. Multiplying rates
by a constant speeds up time; it does not leave the transition probabilities unchanged
unless duration is divided by that constant. The transition exponential uses duration
in seconds. Do not normalize Q, clip negative diagonal entries, or transpose conventions.
initial is a nonnegative row probability vector (n,) summing to one. reward is (n,) in
reward units per second. durations is a positive vector of L>=2 consecutive intervals;
the same rates apply throughout, but unequal intervals are allowed. The final expected
reward is p_final@reward, in reward units per second. The second returned value is the
sum of the stated trapezoidal endpoint approximation on each interval, in reward units.
This approximation is deliberately defined as part of the algorithm; substituting an
exact integral, left-endpoint sum or final-value-times-total-time changes the requested
quantity. Carry p between intervals and include the entering distribution in every
trapezoid. Return exactly shape (1,2), not shape (2,), (2,1), a scalar, or a tuple.
The transition function returns a new dense matrix and does not modify its generator.
All n states participate in the expectation, including states with negative reward.
There is no discount factor, terminal bonus, absorbing-state exception or renormalization
step beyond the generator/exponential definitions. Tiny roundoff is handled by tolerance.""",
            """import numpy as np
from scipy.linalg import expm

def rate_generator(rates):
    q = rates.copy()
    np.fill_diagonal(q,0.0)
    np.fill_diagonal(q,-q.sum(axis=1))
    return q

def transition_matrix(generator,duration):
    return expm(duration*generator)

def reward_pair(distribution,reward,integral):
    return np.array([[distribution@reward,integral]],dtype=np.float64)
""",
        ),
    ]


def _spd(rng, n):
    a = rng.normal(size=(n, n))
    return a.T @ a + np.eye(n)


def samples(chain: Chain, seed: int, trials: int):
    rng = np.random.default_rng(
        seed + int(hashlib.sha256(chain.family.encode()).hexdigest()[:8], 16)
    )
    args: tuple[Any, ...] = ()
    hargs: tuple[Any, ...]
    previous = None
    for trial in range(trials):
        if trial % 2 == 0:
            if chain.family == "weighted-constraint":
                args = (
                    rng.normal(size=(7, 3)),
                    rng.normal(size=7),
                    rng.uniform(0.2, 3, 7),
                    rng.normal(size=7),
                    rng.normal(size=(4, 3)),
                    0.6,
                    rng.normal(size=3),
                    0.7,
                    rng.normal(size=(4, 3)),
                )
            elif chain.family == "conditional-covariance":
                args = (
                    rng.normal(size=6),
                    _spd(rng, 6),
                    np.array([4, 1]),
                    rng.normal(size=2),
                    np.array([5, 0, 3]),
                    rng.normal(size=(2, 3)),
                    rng.normal(size=2),
                    rng.uniform(0.1, 2, 2),
                )
            elif chain.family == "affine-state-history":
                args = (
                    rng.normal(size=4),
                    _spd(rng, 4),
                    np.array([0.3, 0.8, 0.5]),
                    rng.uniform(0.3, 0.9, (3, 2)),
                    rng.normal(size=(3, 2)),
                    np.stack([_spd(rng, 4) / 8 for _ in range(3)]),
                    rng.normal(size=(2, 4)),
                    rng.normal(size=2),
                    _spd(rng, 2) / 5,
                )
            elif chain.family == "affine-mixture-moments":
                args = (
                    rng.normal(size=4),
                    rng.normal(size=(3, 4)),
                    rng.normal(size=(4, 3)),
                    np.stack([_spd(rng, 3) for _ in range(4)]),
                    rng.normal(size=(2, 3)),
                    rng.normal(size=2),
                )
            elif chain.family == "mass-spectral-response":
                args = (
                    _spd(rng, 4),
                    _spd(rng, 4),
                    np.array([0.2, 0.6, 0.3]),
                    rng.normal(size=4),
                    rng.normal(size=(2, 4)),
                )
            else:
                args = (
                    rng.uniform(0.1, 0.6, (3, 3)),
                    np.array([0.2, 0.7, 0.4]),
                    np.array([0.2, 0.3, 0.5]),
                    rng.normal(size=3),
                )
        else:
            args = tuple(x.copy() if isinstance(x, np.ndarray) else x for x in args)
            if chain.family == "weighted-constraint":
                args = (
                    args[0],
                    args[1] * 2,
                    args[2],
                    args[3] * 2,
                    *args[4:7],
                    args[7] * 2,
                    args[8],
                )
            elif chain.family == "conditional-covariance":
                args = (args[0], args[1], args[2][::-1], args[3][::-1], *args[4:])
            elif chain.family == "affine-state-history":
                args = (
                    args[0] * 2,
                    args[1],
                    args[2],
                    args[3],
                    args[4] * 2,
                    *args[5:7],
                    args[7] * 2,
                    args[8],
                )
            elif chain.family == "affine-mixture-moments":
                args = (args[0] + 700, args[1] - 600, *args[2:])
            elif chain.family == "mass-spectral-response":
                args = (*args[:3], args[3] * 2, args[4])
            else:
                args = (args[0] * 2, args[1] / 2, args[2], args[3])
        if chain.family == "weighted-constraint":
            a, b, v, o, r, strength, c, t, q = args
            n = np.einsum("ni,nj,n->ij", a, a, 1 / v) + strength * r.T @ r
            rhs = np.einsum("ni,n,n->i", a, b - o, 1 / v)
            h = np.column_stack((n, rhs))
            hargs = args[:6]
            kkt = np.block([[n, c[:, None]], [c[None, :], np.zeros((1, 1))]])
            inv = np.linalg.inv(kkt)
            x = inv @ np.r_[rhs, t]
            y = np.column_stack((q @ x[:-1], [row @ inv[:-1, :-1] @ row for row in q]))
        elif chain.family == "conditional-covariance":
            mu, cov, oi, val, ti, q, bias, noise = args
            indices = np.r_[ti, oi]
            joint = cov[np.ix_(indices, indices)]
            precision = np.linalg.inv(joint)
            r = len(ti)
            c = np.linalg.inv(precision[:r, :r])
            m = mu[ti] - c @ precision[:r, r:] @ (val - mu[oi])
            h = np.column_stack((c, m))
            hargs = args[:5]
            y = np.column_stack((q @ c @ q.T + np.diag(noise), q @ m + bias))
        elif chain.family == "affine-state-history":
            mu, cov, durations, decays, drives, noises, obs, bias, measurement = args

            def step(m, c, dt, decay, drive, noise):
                d = len(decay)
                f = np.eye(2 * d)
                for i in range(d):
                    f[i, i + d] = dt
                    f[i + d, i + d] = decay[i]
                u = np.r_[np.zeros(d), dt * drive]
                return f @ m + u, f @ c @ f.T + dt * noise

            m, c = step(mu, cov, durations[0], decays[0], drives[0], noises[0])
            h = np.column_stack((c, m))
            hargs = (mu, cov, float(durations[0]), decays[0], drives[0], noises[0])
            m, c = mu.copy(), cov.copy()
            for dt, de, dr, no in zip(durations, decays, drives, noises):
                m, c = step(m, c, dt, de, dr, no)
            y = np.column_stack((obs @ c @ obs.T + measurement, obs @ m + bias))
        elif chain.family == "affine-mixture-moments":
            prior, evidence, means, covs, t, shift = args
            z = prior + evidence
            e = np.exp(z - z.max(axis=1, keepdims=True))
            h = e / e.sum(axis=1, keepdims=True)
            hargs = args[:2]
            transformed = np.array([t @ m + shift for m in means])
            transformed_cov = np.array([t @ c @ t.T for c in covs])
            rows = []
            for p in h:
                center = sum(p[k] * transformed[k] for k in range(len(p)))
                second = sum(
                    p[k]
                    * (transformed_cov[k] + np.outer(transformed[k], transformed[k]))
                    for k in range(len(p))
                )
                rows.append(np.r_[center, (second - np.outer(center, center)).ravel()])
            y = np.array(rows)
        elif chain.family == "mass-spectral-response":
            k, m, durations, x, obs = args
            chol = np.linalg.cholesky(m)
            li = np.linalg.inv(chol)
            lam, u = np.linalg.eigh(li @ k @ li.T)

            def prop(dt):
                return li.T @ ((u * np.exp(-dt * lam)) @ u.T) @ chol.T

            h = prop(durations[0])
            hargs = (k, m, float(durations[0]))
            final = prop(sum(durations)) @ x
            y = np.r_[obs @ final, 0.5 * final @ m @ final][:, None]
        else:
            rates, durations, p, reward = args
            q = rates.copy()
            np.fill_diagonal(q, 0)
            np.fill_diagonal(q, -q.sum(axis=1))
            rate = max(-np.diag(q))
            embedded = np.eye(len(q)) + q / rate

            def prop(dt):
                term = np.eye(len(q))
                result = term.copy()
                for n in range(1, 70):
                    term = term @ embedded * (rate * dt / n)
                    result += term
                return np.exp(-rate * dt) * result

            h = prop(durations[0])
            hargs = (rates, float(durations[0]))
            p = p.copy()
            total = 0.0
            for dt in durations:
                nxt = p @ prop(dt)
                total += dt * (p @ reward + nxt @ reward) / 2
                p = nxt
            y = np.array([[p @ reward, total]])
        if trial % 2:
            assert previous is not None
            expected = previous.copy()
            if chain.family == "weighted-constraint":
                expected[:, 0] *= 2
            elif chain.family == "affine-state-history":
                expected[:, -1] *= 2
            elif chain.family == "mass-spectral-response":
                expected[:-1] *= 2
                expected[-1] *= 4
            elif chain.family == "rate-reward-history":
                expected[:, 1] /= 2
            if not np.allclose(y, expected, rtol=1e-9, atol=1e-9):
                raise ValueError("metamorphic oracle mismatch")
        previous = np.asarray(y).copy()
        yield (
            hargs,
            args,
            np.asarray(h, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
        )


def score(chain, stage, code, run, seed, trials):
    cursor = iter(samples(chain, seed, trials))

    def sample(_name, _rng, _trial):
        ha, a, h, y = next(cursor)
        return (ha, h) if stage == 0 else (a, y)

    payload, expected = short.sandbox.prepare(
        chain.helper if stage == 0 else chain.downstream,
        chain.support + "\n" + code,
        seed,
        trials,
        sample_fn=sample,
    )
    return short.sandbox.compare(run(payload), expected)


def trusted_run(data):
    allowed = set()
    for c in catalog():
        oracle = short.as_helper(c.helper.code())
        for i in (None, *range(len(c.helper.foils))):
            allowed.add(c.support + "\n" + c.helper.code(i))
            allowed.add(
                c.support
                + "\n"
                + short.as_helper(c.helper.code(i))
                + c.downstream.code()
            )
        for i in (None, *range(len(c.downstream.foils))):
            allowed.add(c.support + "\n" + oracle + c.downstream.code(i))
    if data["code"] not in allowed:
        raise ValueError("source-owned reference programs only")
    r = subprocess.run(
        [sys.executable, "-I", "-c", short.sandbox.WORKER],
        input=json.dumps(data),
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    return json.loads(r.stdout)


def validate(seed=64123, trials=8):
    if not 4 <= trials <= 100 or trials % 2:
        raise ValueError("even4..100 trials required")
    records = []
    for c in catalog():
        correct = short.evaluate(
            c,
            c.helper.code(),
            c.downstream.code(),
            trusted_run,
            seed=seed,
            trials=trials,
            score_fn=score,
        )
        if not all(
            correct[k]["passed"]
            for k in (
                "helper",
                "downstream_given_oracle",
                "whole_chain_generated_helper",
            )
        ):
            raise ValueError((c.family, correct))
        foils = []
        for stage, item in enumerate((c.helper, c.downstream)):
            for i, (label, _, _) in enumerate(item.foils):
                code = item.code(i)
                if stage:
                    code = short.as_helper(c.helper.code()) + code
                result = score(c, stage, code, trusted_run, seed, trials)
                if (
                    result["passed"]
                    or result["status"] != "evaluated"
                    or any(
                        f["kind"] in {"runtime", "output_shape", "output_type"}
                        for f in result["failures"]
                    )
                ):
                    raise ValueError(("foil not numerical", c.family, label, result))
                foils.append({"stage": stage + 1, "foil_id": label, "outcome": result})
        records.append(
            {"family": c.family, "split": c.split, "correct": correct, "foils": foils}
        )
    return {
        "seed": seed,
        "trials_per_stage": trials,
        "paired_metamorphic_trials": trials // 2,
        "libraries": {n: importlib.metadata.version(n) for n in ("numpy", "scipy")},
        "chains": records,
    }


def write_bundle(output, *, base=None, base_sha256=None, seed=64123, trials=8):
    output = Path(output).resolve()
    source = Path(__file__).resolve()
    inputs = [
        source,
        Path(short.__file__).resolve(),
        Path(short.sandbox.__file__).resolve(),
        Path(short.generator.__file__).resolve(),
    ]
    if base:
        inputs.append(Path(base).resolve())
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if any(p.is_relative_to(output) or output.is_relative_to(p) for p in inputs):
        raise ValueError("output overlaps input")
    before = {str(p): short.digest(p.read_bytes()) for p in inputs}
    if base and before[str(Path(base).resolve())] != base_sha256:
        raise ValueError("base SHA mismatch")
    original_input = json.loads(Path(base).read_text()) if base else {}
    original = original_input.get("cases", original_input)
    combined = json.loads(json.dumps(original))
    extension = {}
    extras = []
    for c in catalog():
        for stage, item in enumerate((c.helper, c.downstream)):
            if c.split == "train":
                purpose = (
                    "science_long_helper" if stage == 0 else "science_long_composition"
                )
                extension.setdefault(purpose, [[], []])[c.fold].append(c.row(stage))
            extras.append(
                {
                    "family": c.family,
                    "stage": stage + 1,
                    "split": c.split,
                    "gradient_allowed": False,
                    "foils": [
                        {"foil_id": label, "code": item.code(i)}
                        for i, (label, _, _) in enumerate(item.foils)
                    ],
                }
            )
    if set(extension) & set(combined):
        raise ValueError("new purposes overlap base")
    ids = {row["id"] for folds in combined.values() for fold in folds for row in fold}
    if any(
        r["id"] in ids for folds in extension.values() for fold in folds for r in fold
    ):
        raise ValueError("duplicate case ID")
    combined.update(extension)
    validation = validate(seed, trials)
    metadata = {
        "schema_version": 1,
        "parent_binding": None,
        "training_families": 4,
        "training_stages": 8,
        "transfer_families": 2,
        "new_purposes": list(extension),
        "source_sha256": before,
        "base_sha256": base_sha256,
        "old_objects_preserved": all(combined[k] == v for k, v in original.items()),
        "answer_unit": "complete stage solve and imports; context helpers excluded",
        "collector_weighting": "one row per stage; two stages/family; first foil only for gradients",
        "extra_foils": "CPU-only semantic failure probes, no duplicate correct LL weight",
        "token_count_measured": False,
        "validation": validation,
    }
    if before != {str(p): short.digest(p.read_bytes()) for p in inputs}:
        raise ValueError("input changed")
    output.mkdir(parents=True, exist_ok=False)
    for name, value in [
        ("train-extension.json", extension),
        ("train-cases.json", combined),
        ("failure-mode-foils.json", extras),
        ("metadata.json", metadata),
    ]:
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return metadata


evaluate_requests = partial(short.evaluate_requests, score_fn=score)

if __name__ == "__main__":
    short.main(
        catalog_fn=catalog,
        bundle_fn=write_bundle,
        evaluator=evaluate_requests,
        extra_sources=(Path(__file__),),
    )

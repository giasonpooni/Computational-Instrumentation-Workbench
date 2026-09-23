"""Bounded synthetic experiment declaration and explicit coordinate maps."""
from copy import deepcopy
import math

import numpy as np

from .adapters.subprocess import _json
from .telemetry import canonical, _keys

KIND = "variational-free-energy"
SOURCE_SCHEMA = "ciw.variational-free-energy-source.v1"
POLICY = {
    "inference": "static_two_latent_linear_gaussian",
    "objective": "normalized_variational_free_energy_including_constants",
    "covariance_update": "natural_gradient_precision_relaxation",
    "observations": "retained_synthetic_values",
    "coverage": "synthetic_prior_predictive_ensemble",
    "train_heldout_noise": "conditionally_independent_given_latent",
    "physical_validation": "not_established",
    "plant_stability": "not_established",
    "state_admission": "not_performed",
}
STATE_NAMES = ["initial_lateral_error", "initial_heading_error"]
STATE_UNITS = ["m", "radian"]
SOURCE_LIMIT = 256 * 1024


def number(value, limit=1e6):
    if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > limit:
        raise ValueError("Require a bounded finite number, not a Boolean")
    return float(value)


def vector(value, limit=1e6):
    if type(value) is not list or len(value) != 2:
        raise ValueError("Require two ordered coordinates")
    return [number(x, limit) for x in value]


def matrix(value, *, positive=False):
    if type(value) is not list or len(value) != 2:
        raise ValueError("Require a two by two matrix")
    a = np.asarray([vector(row) for row in value])
    if positive:
        if not np.array_equal(a, a.T):
            raise ValueError("Covariance must be exactly symmetric; no repair")
        eig = np.linalg.eigvalsh(a)
        if eig[0] < 1e-12 or eig[-1] / eig[0] > 1e6:
            raise ValueError("Covariance violates the positive margin or condition budget")
    return a


def validate_source(raw):
    try:
        if type(raw) is not bytes or not 1 <= len(raw) <= SOURCE_LIMIT:
            raise ValueError("Free-energy source requires bounded exact JSON bytes")
        source = _json(raw)
        _keys(source, {"schema", "experiment_id", "configuration", "geometry", "coordinates", "sensors",
                       "assumed_model", "generator", "samples", "representative_index", "solver"})
        if source["schema"] != SOURCE_SCHEMA or canonical(source["configuration"]) != canonical(POLICY):
            raise ValueError("Unsupported free-energy source or authority policy")
        if type(source["experiment_id"]) is not str or not 1 <= len(source["experiment_id"]) <= 128:
            raise ValueError("Require a bounded experiment identity")
        geo = source["geometry"]
        _keys(geo, {"arclength", "gaussian_curvature", "length_unit", "relative_tolerance"})
        grid = geo["arclength"]
        if type(grid) is not list or not 2 <= len(grid) <= 81:
            raise ValueError("Require 2..81 arclength samples")
        grid = [number(x, 4) for x in grid]
        curvature = number(geo["gaussian_curvature"], 1)
        if grid[0] != 0 or any(b <= a or abs(curvature)*(b-a)**2 > .01+1e-14 for a,b in zip(grid,grid[1:])):
            raise ValueError("Refine a strictly increasing Jacobi grid starting at zero")
        if geo["length_unit"] != "m" or not 0 < number(geo["relative_tolerance"]) <= .01:
            raise ValueError("Declare metre arclength and a bounded linearization tolerance")
        coordinates = source["coordinates"]
        _keys(coordinates, {"names", "units", "scales", "frame"})
        if coordinates["names"] != STATE_NAMES or coordinates["units"] != STATE_UNITS or coordinates["frame"] != "transverse-to-gamma, parallel-transported":
            raise ValueError("Declare the initial lateral/heading coordinate basis")
        if any(not 1e-4 <= x <= 1 for x in vector(coordinates["scales"])):
            raise ValueError("State normalization scales must be explicit and positive")
        sensors = source["sensors"]
        if type(sensors) is not list or len(sensors) != 2:
            raise ValueError("Require two declared scalar measurement sources")
        for index, sensor in enumerate(sensors):
            _keys(sensor, {"id", "component", "unit", "scale", "training_index", "heldout_index"})
            if type(sensor["id"]) is not str or not 1 <= len(sensor["id"]) <= 64:
                raise ValueError("Require a bounded sensor name")
            if sensor["component"] != ["lateral", "heading"][index] or sensor["unit"] != STATE_UNITS[index]:
                raise ValueError("The two sources measure lateral displacement and heading respectively")
            if not 1e-4 <= number(sensor["scale"]) <= 1:
                raise ValueError("Observation normalization scale must be positive")
            for key in ("training_index", "heldout_index"):
                if type(sensor[key]) is not int or not 0 <= sensor[key] < len(grid):
                    raise ValueError("Measurement position is outside the Jacobi grid")
        if sensors[0]["id"] == sensors[1]["id"]:
            raise ValueError("Measurement source identities must differ")
        model = source["assumed_model"]
        _keys(model, {"prior_mean", "prior_covariance", "training_noise_covariance", "heldout_noise_covariance",
                      "training_bias", "heldout_bias"})
        for key in ("prior_mean", "training_bias", "heldout_bias"):
            vector(model[key], 1)
        for key in ("prior_covariance", "training_noise_covariance", "heldout_noise_covariance"):
            matrix(model[key], positive=True)
        gen = source["generator"]
        _keys(gen, {"kind", "seed", "numpy_version", "latent_distribution", "gaussian_curvature",
                    "training_noise_covariance", "heldout_noise_covariance", "training_bias", "heldout_bias"})
        if (gen["kind"] != "numpy_pcg64_retained_draws" or gen["numpy_version"] != "2.4.3" or
                gen["latent_distribution"] != "assumed_model_prior" or type(gen["seed"]) is not int or not 0 <= gen["seed"] < 2**32):
            raise ValueError("Require an explicit synthetic generator declaration")
        number(gen["gaussian_curvature"], 1)
        for key in ("training_noise_covariance", "heldout_noise_covariance"):
            matrix(gen[key], positive=True)
        for key in ("training_bias", "heldout_bias"):
            vector(gen[key], 1)
        samples = source["samples"]
        if type(samples) is not list or not 1 <= len(samples) <= 128:
            raise ValueError("Retain 1..128 actual synthetic realizations")
        for sample in samples:
            _keys(sample, {"truth", "training_noise", "heldout_noise", "training", "heldout"})
            for value in sample.values():
                vector(value, 10)
        # Bind the retained observations to the declared synthetic construction.
        # This checks algebra, not that the finite draws authenticate an IID law.
        truth_phi = analytic_transfer(grid, gen["gaussian_curvature"])
        for selection in ("training", "heldout"):
            g = np.asarray([truth_phi[sensor[selection+"_index"]][i] for i,sensor in enumerate(sensors)])
            for sample in samples:
                expected = g @ np.asarray(sample["truth"]) + np.asarray(gen[selection+"_bias"]) + np.asarray(sample[selection+"_noise"])
                if not np.allclose(sample[selection], expected, rtol=1e-13, atol=1e-14):
                    raise ValueError("Synthetic observations differ from declared truth, geometry, bias or retained noise")
        if type(source["representative_index"]) is not int or not 0 <= source["representative_index"] < len(samples):
            raise ValueError("Representative sample is outside the retained ensemble")
        solver = source["solver"]
        _keys(solver, {"initial_mean", "initial_covariance", "alpha", "beta", "max_iterations", "gradient_tolerance", "precision_tolerance"})
        vector(solver["initial_mean"], 100)
        matrix(solver["initial_covariance"], positive=True)
        if not 0 < number(solver["alpha"]) <= 10 or not 0 < number(solver["beta"]) < 1:
            raise ValueError("Declare positive mean step and strict precision relaxation in (0,1)")
        if type(solver["max_iterations"]) is not int or not 1 <= solver["max_iterations"] <= 256:
            raise ValueError("Require 1..256 explicit iterations")
        for key in ("gradient_tolerance", "precision_tolerance"):
            if not 1e-12 <= number(solver[key]) <= 1e-4:
                raise ValueError("Declare a convergence tolerance in [1e-12,1e-4]")
        return source
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError, RecursionError) as exc:
        raise ValueError("Malformed free-energy declaration") from exc


def analytic_transfer(arclength, curvature):
    """Closed-form constant-curvature reference used only for synthetic truth."""
    values = []
    for s in arclength:
        if curvature == 0:
            c, t = 1.0, float(s)
        elif curvature > 0:
            root = math.sqrt(curvature)
            c, t = math.cos(root*s), math.sin(root*s)/root
        else:
            root = math.sqrt(-curvature)
            c, t = math.cosh(root*s), math.sinh(root*s)/root
        values.append([[c,t],[-curvature*t,c]])
    return values


def csg_request(source):
    geometry = source["geometry"]
    return {"arclength": deepcopy(geometry["arclength"]), "gaussian_curvature": geometry["gaussian_curvature"],
            "units": {"length": "m", "angle": "radian"}, "relative_tolerance": geometry["relative_tolerance"]}


def problems(source, phi):
    """Map only training observations into inference; truth/heldout are diagnostics."""
    scale = np.asarray(source["coordinates"]["scales"])
    obs_scale = np.asarray([sensor["scale"] for sensor in source["sensors"]])
    model, sample = source["assumed_model"], source["samples"][source["representative_index"]]
    matrices = {}
    for selection in ("training", "heldout"):
        physical = np.asarray([phi[sensor[selection+"_index"]][i] for i,sensor in enumerate(source["sensors"])])
        matrices[selection] = physical * scale[None,:] / obs_scale[:,None]
    prior = np.asarray(model["prior_covariance"]) / np.outer(scale,scale)
    common = {"coordinate_system": "normalized_dimensionless"}
    problem = {**common, "prior_mean": (np.asarray(model["prior_mean"])/scale).tolist(),
        "prior_covariance": prior.tolist(), "observation_matrix": matrices["training"].tolist(),
        "observations": ((np.asarray(sample["training"])-model["training_bias"])/obs_scale).tolist(),
        "noise_covariance": (np.asarray(model["training_noise_covariance"])/np.outer(obs_scale,obs_scale)).tolist()}
    held = {**common, "observation_matrix": matrices["heldout"].tolist(),
        "observations": ((np.asarray(sample["heldout"])-model["heldout_bias"])/obs_scale).tolist(),
        "noise_covariance": (np.asarray(model["heldout_noise_covariance"])/np.outer(obs_scale,obs_scale)).tolist()}
    normalization = {"latent_scales": scale.tolist(), "observation_scales": obs_scale.tolist(),
        "log_observation_jacobian": float(np.log(obs_scale).sum()),
        "map": "G_normalized=diag(observation_scales)^-1 H Phi diag(latent_scales)",
        "solver_coordinates": "normalized_dimensionless", "density_measure": "declared_observation_coordinates"}
    return problem, held, normalization

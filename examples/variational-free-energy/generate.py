"""Regenerate six paired synthetic declarations, retaining every actual draw.

Run after installing this CIW revision with its NumPy 2.4.3 pin. The generator
uses the analytic constant-curvature transfer for synthetic truth. Its printed
diagnostics also use analytic assumed transfers; the native integration gate
separately exercises the pinned CSG/GSIE/PLSR providers.

The seed, representative index and 128 prior-predictive replicates are fixed in
advance. No examples are selected to attain a desired coverage count. Cases use
the same latent and independent standard-normal noise arrays: modeled versus
ignored correlation share identical observed samples; wrong curvature and the
unstable step share baseline samples. Sensor bias adds a fixed source offset.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from ciw.free_energy_math import evaluate_ensemble, information_system, predict_held_out, variational_fit
from ciw.free_energy_profile import POLICY, SOURCE_SCHEMA, STATE_NAMES, STATE_UNITS, analytic_transfer, problems, validate_source

SEED = 20260923
REPLICATES = 128
CASES = ("baseline", "correlated-noise", "ignored-correlation", "sensor-bias", "wrong-curvature", "unstable-step")


def _noise(correlation):
    off_diagonal = correlation * 0.002 * 0.003
    return [[0.002**2, off_diagonal], [off_diagonal, 0.003**2]]


def build_sources():
    if np.__version__ != "2.4.3":
        raise RuntimeError("Regeneration requires the declared NumPy 2.4.3 runtime")
    generator = np.random.Generator(np.random.PCG64(SEED))
    latent_standard = generator.standard_normal((REPLICATES, 2))
    training_standard = generator.standard_normal((REPLICATES, 2))
    heldout_standard = generator.standard_normal((REPLICATES, 2))
    prior = np.array([[0.015**2, 0.25*0.015*0.02], [0.25*0.015*0.02, 0.02**2]])
    truths = latent_standard @ np.linalg.cholesky(prior).T
    grid = [index / 40 for index in range(81)]
    sensors = [
        {"id":"lateral-source", "component":"lateral", "unit":"m", "scale":0.01,
         "training_index":20, "heldout_index":70},
        {"id":"heading-source", "component":"heading", "unit":"radian", "scale":0.01,
         "training_index":30, "heldout_index":80}]
    true_phi = analytic_transfer(grid, 0.25)
    true_matrices = {selection: np.asarray([true_phi[sensor[selection+"_index"]][index]
        for index,sensor in enumerate(sensors)]) for selection in ("training", "heldout")}
    sources = {}
    for case in CASES:
        generated_covariance = _noise(0.85 if case in {"correlated-noise", "ignored-correlation"} else 0)
        assumed_covariance = generated_covariance if case == "correlated-noise" else _noise(0)
        generated_bias = [0.006, -0.009] if case == "sensor-bias" else [0, 0]
        training_noise = training_standard @ np.linalg.cholesky(generated_covariance).T
        heldout_noise = heldout_standard @ np.linalg.cholesky(generated_covariance).T
        training = truths @ true_matrices["training"].T + generated_bias + training_noise
        heldout = truths @ true_matrices["heldout"].T + generated_bias + heldout_noise
        source = {"schema":SOURCE_SCHEMA, "experiment_id":"variational-free-energy-"+case,
            "configuration":deepcopy(POLICY),
            "geometry":{"arclength":grid[:], "gaussian_curvature":-0.75 if case == "wrong-curvature" else 0.25,
                "length_unit":"m", "relative_tolerance":0.001},
            "coordinates":{"names":STATE_NAMES[:], "units":STATE_UNITS[:], "scales":[0.01,0.01],
                "frame":"transverse-to-gamma, parallel-transported"},
            "sensors":deepcopy(sensors),
            "assumed_model":{"prior_mean":[0,0], "prior_covariance":prior.tolist(),
                "training_noise_covariance":deepcopy(assumed_covariance),
                "heldout_noise_covariance":deepcopy(assumed_covariance), "training_bias":[0,0], "heldout_bias":[0,0]},
            "generator":{"kind":"numpy_pcg64_retained_draws", "seed":SEED, "numpy_version":np.__version__,
                "latent_distribution":"assumed_model_prior", "gaussian_curvature":0.25,
                "training_noise_covariance":deepcopy(generated_covariance),
                "heldout_noise_covariance":deepcopy(generated_covariance),
                "training_bias":generated_bias[:], "heldout_bias":generated_bias[:]},
            "samples":[{"truth":truths[index].tolist(), "training_noise":training_noise[index].tolist(),
                "heldout_noise":heldout_noise[index].tolist(), "training":training[index].tolist(),
                "heldout":heldout[index].tolist()} for index in range(REPLICATES)],
            "representative_index":0,
            "solver":{"initial_mean":[1,-1], "initial_covariance":[[1,0],[0,1]], "alpha":0.01, "beta":0.2,
                "max_iterations":12 if case == "unstable-step" else 256,
                "gradient_tolerance":1e-8, "precision_tolerance":1e-8}}
        assumed_phi = analytic_transfer(grid, source["geometry"]["gaussian_curvature"])
        normalized, _, _ = problems(source, assumed_phi)
        eigenvalues = np.linalg.eigvalsh(information_system(normalized)["precision"])
        # Depends only on declared prior/design/noise, never on truths or outcomes.
        source["solver"]["alpha"] = float(2.4/eigenvalues[-1] if case == "unstable-step" else 2/(eigenvalues[0]+eigenvalues[-1]))
        encoded = (json.dumps(source,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode("utf-8")
        validate_source(encoded)
        sources[case] = source
    return sources


def diagnostic_summary(sources):
    output = {}
    for case,source in sources.items():
        phi = analytic_transfer(source["geometry"]["arclength"],source["geometry"]["gaussian_curvature"])
        normalized, held, _ = problems(source,phi)
        fitted = variational_fit(normalized,**source["solver"])
        model = source["assumed_model"]
        scales = np.asarray(source["coordinates"]["scales"])
        obs_scales = np.asarray([sensor["scale"] for sensor in source["sensors"]])
        truths = np.asarray([sample["truth"] for sample in source["samples"]])/scales
        training = (np.asarray([sample["training"] for sample in source["samples"]])-model["training_bias"])/obs_scales
        heldout = (np.asarray([sample["heldout"] for sample in source["samples"]])-model["heldout_bias"])/obs_scales
        ensemble = evaluate_ensemble(normalized,truths,training,held,heldout)
        prediction = predict_held_out(held,fitted["mean"],fitted["covariance"])
        output[case] = {"transfer_scope":"analytic_generation_check; native_gate_separate",
            "solver_status":fitted["status"], "iterations":fitted["iterations"], "alpha":source["solver"]["alpha"],
            "spectral_radius":fitted["stability"]["spectral_radius"],
            "initial_free_energy":fitted["trace"][0]["free_energy"],
            "final_free_energy":fitted["trace"][-1]["free_energy"],
            "final_kl":fitted["trace"][-1]["kl_to_reference"],
            "latent_joint_count":ensemble["latent_joint_count"], "latent_marginal_counts":ensemble["latent_marginal_counts"],
            "heldout_joint_count":ensemble["heldout_joint_count"], "heldout_marginal_counts":ensemble["heldout_marginal_counts"],
            "mean_heldout_log_predictive_density":float(np.mean([record["held_out"]["log_predictive_density"] for record in ensemble["records"]])),
            "representative_variational_heldout_joint_covered":prediction["joint_covered"],
            "replicates":REPLICATES}
    return output


def main():
    sources = build_sources()
    summary = diagnostic_summary(sources)
    for case in CASES:
        if case != "unstable-step" and summary[case]["solver_status"] != "converged":
            raise RuntimeError("Declared stable demonstration failed its convergence gate: "+case)
        if case == "unstable-step" and (summary[case]["solver_status"] == "converged" or summary[case]["spectral_radius"] <= 1):
            raise RuntimeError("Declared unstable control does not expose capped unstable iteration")
    directory = Path(__file__).resolve().parent
    for case,source in sources.items():
        encoded = (json.dumps(source,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode("utf-8")
        (directory/(case+".json")).write_bytes(encoded)
    print(json.dumps(summary,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

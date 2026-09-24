# What remains unmeasured

Scope: the clean-room reports of T001-T166 in this output directory. Hardware runs retained under lab/hardware/ are aggregated outside the queue by `ciw lab unmeasured --retained lab`; this ledger never reads them, so its counts cover this run only.

Hardware-measured findings in this run: 0.

## Next acquisitions

1. hardware:nvidia-gpu — On the RTX 2080 host: `ciw energy probe --gpu-index 0` must report status ok; then run T116, T117, T118, T119, T121, T124, T147 there into a fresh output directory following their protocols (10 open physical claims name this route's device; 3 computational comparisons (T117, T121, T147) run only on this host) and retain the run with `ciw lab hardware retain`.
2. hardware:rapl — On the capture host: an intel-rapl energy_uj counter under /sys/class/powercap must read as a number (the runner's hardware:rapl probe); `python -m ciw.lab.energy_gpu_telemetry rapl-capture` then records the capture; then run T115, T117, T120 there into a fresh output directory following their protocols (6 open physical claims name this route's device) and retain the run with `ciw lab hardware retain`.

Open physical claims with no instrument probe for them in their task (below) need a probe of their instrument, or a signed-capture trust anchor, before any acquisition can establish them.

## Open claims by what they need

### Acquisition on hardware:nvidia-gpu

- T116 [physical; failed probes hardware:nvidia-gpu]: GPU-domain gross energy per measured batch
- T118 [physical; failed probes hardware:nvidia-gpu]: RTX 2080 power draw during the measurement phase (NVML)
- T118 [physical; failed probes hardware:nvidia-gpu]: RTX 2080 temperature during the measurement phase (NVML)
- T118 [physical; failed probes hardware:nvidia-gpu]: RTX 2080 graphics clock during the measurement phase (NVML)
- T118 [physical; failed probes hardware:nvidia-gpu]: Host-bracketed batch solve duration (launch, sync and copy included)
- T118 [physical; failed probes hardware:nvidia-gpu]: RTX 2080 GPU utilization during the measurement phase (nvidia-smi rows inside the measurement window)
- T118 [physical; failed probes hardware:nvidia-gpu]: RTX 2080 power draw is steady over the measurement phase (coefficient of variation <= 0.10)
- T118 [physical; failed probes hardware:nvidia-gpu]: RTX 2080 temperature drifts by at most 5 C over the measurement phase
- T119 [physical]: Physical GPU energy per accepted numerical result
- T124 [physical]: The bound operator log names an NVML device present on this analyzing host (UUID, name, driver, NVML version and NVML library digest equal this host's; the capture is unauthenticated)

### Acquisition on hardware:rapl

- T115 [physical]: Gross CPU package energy (background-inclusive, idle not subtracted) per geodesic trajectory
- T115 [physical]: Idle-subtracted CPU package energy per geodesic trajectory (equal-length idle bracket after the workload)
- T115 [physical]: Gross CPU package energy (background-inclusive, idle not subtracted) per batch of the common Gaussian VI workload (NumPy float64 reference)
- T115 [physical]: Idle-subtracted CPU package energy per batch of the common Gaussian VI workload (NumPy float64 reference)
- T117 [physical; failed probes hardware:nvidia-gpu, tool:julia]: CPU package energy per batch of the common Gaussian VI workload: Rust port and NumPy reference (RAPL, gross and idle-subtracted)
- T120 [physical]: CPU package energy per batch of the common Gaussian VI workload in float32 and in float64 (NumPy reference, RAPL, gross and idle-subtracted)

### Runs on the nvidia-gpu host (computational comparison; no acquisition needed)

Computational comparisons whose code exists and that run only where the route's probe succeeds; a run on that host (listed under Next acquisitions) decides them, and they need no acquisition record:

- T117 [numerical; failed probes hardware:nvidia-gpu, tool:julia]: The gaussian_vi PTX kernel on the GPU reproduces the NumPy reference bitwise on every replica of the common Gaussian VI workload
- T121 [numerical; failed probes hardware:nvidia-gpu]: The gaussian_vi PTX kernel keeps the declared unfused order of its in-kernel reductions on the GPU: its outputs equal the NumPy reference bitwise
- T147 [numerical; failed probes hardware:nvidia-gpu]: CPU and GPU outputs of the common Gaussian VI workload agree under T148's policy for fixed-order reductions (bitwise) on GPU hardware

### Needs implementation first

- T117 [numerical; failed probes hardware:nvidia-gpu, tool:julia]: A Julia implementation of the common Gaussian VI workload agrees with the NumPy reference — implementation missing: no Julia implementation of the common workload exists, so this comparison cannot run on any host; julia is not on PATH here
- T118 [physical; failed probes hardware:nvidia-gpu]: RTX 2080 kernel-only duration of the Gaussian VI kernel — log.json brackets launch, synchronization and copy; kernel spans need an Nsight Systems report, which this section does not ingest (deferred research question: ingest `nsys stats --report cuda_gpu_kern_sum` output)
- T120 [physical]: GPU energy per batch of a float32 build of the common Gaussian VI workload — implementation missing: no float32 implementation of the gaussian_vi PTX kernel exists (ciw.energy_cuda is binary64 only, and `ciw energy record` captures only it), so a float32 GPU workload cannot run on any host; Deferred research question: extend the common workload's GPU path beyond the binary64 gaussian_vi PTX kernel of ciw.energy_cuda: a float32 rendering of the same kernel, and a device-wide reduction of its per-replica outputs (a fixed tree over the stored order per T148's REDUCTION_POLICY, plus one atomicAdd variant), each compared with the NumPy reference by compare_outputs on the RTX 2080 host and captured with `ciw energy record` for energy; until then those claims stay not_established on every host
- T121 [numerical; failed probes hardware:nvidia-gpu]: A device-wide reduction of the common workload's per-replica outputs on the RTX 2080 (a fixed tree per T148's policy, and atomicAdd) reproduces the emulated order spreads and sign flips — implementation missing: no PTX device reduction kernel exists (the gaussian_vi kernel reduces nothing across replicas), so this comparison cannot run on any host; Deferred research question: extend the common workload's GPU path beyond the binary64 gaussian_vi PTX kernel of ciw.energy_cuda: a float32 rendering of the same kernel, and a device-wide reduction of its per-replica outputs (a fixed tree over the stored order per T148's REDUCTION_POLICY, plus one atomicAdd variant), each compared with the NumPy reference by compare_outputs on the RTX 2080 host and captured with `ciw energy record` for energy; until then those claims stay not_established on every host
- T145 [computational_pipeline; failed probes tool:julia]: Julia environment pinned and exercised through the CIW to SCR boundary — This task has no Julia execution path: it records the pin procedure and probes for julia only, so provisioning Julia changes no finding until a Julia worker behind the SCR boundary and a dispatch path from this task exist
- T146 [computational_pipeline]: Byte-identical canonical JSON holds for Julia, C++ and GPU-host implementations — its task: Julia, C++ and GPU-host encoders were not run

### Needs a reference instrument no acquisition route provides

- T116 [sensor_performance; failed probes hardware:nvidia-gpu]: The NVML total-energy counter of the RTX 2080 has a characterized accuracy and resolution — NVML declares no accuracy or resolution for this counter and no external power meter was compared; the counter is used uncalibrated

### Claims about retained synthetic fixtures (no acquisition changes their origin)

- T124 [physical]: The fixtures' counter readings were produced by a physical GPU and NVML counter — the fixtures declare origin synthetic_fixture and carry placeholder library and executable digests

### No instrument probe for it in its task

- T005 [physical]: Nearby real trajectories on a physical curved surface separate according to this Jacobi law
- T009 [physical]: The ranking of lateral versus heading start errors computed here predicts which error dominates the endpoint error of real tool or vehicle paths on physical curved parts
- T013 [physical]: A physical cylinder or large-radius torus workpiece shows these separations
- T018 [sensor_performance]: Curvature signals resolvable here would be resolvable in measured sensor data
- T023 [sensor_performance]: A physical heading sensor closes the shortest loop within the computed heading tolerance
- T030 [physical]: Grid-planned path lengths predict distances travelled by a physical vehicle or tool
- T031 [calibration]: A metric calibrated from physical measurements is accurate enough to decide between near-tied routes
- T033 [physical]: The conformance suite certifies surfaces reconstructed from physical measurements
- T034 [physical]: Symbolic and dual-number derivative agreement certifies derivatives of surfaces reconstructed from physical measurements
- T035 [physical]: The optimal-step law derived here applies to derivatives of measured surface samples
- T037 [physical]: The singularity classification applies to scanned physical parts
- T038 [physical]: Straightest geodesics on a mesh reconstructed from a real scan reproduce the geodesics of the scanned physical surface
- T039 [physical]: The observed convergence orders transfer to meshes reconstructed from real scans
- T040 [physical]: Angle-defect curvature of a scanned mesh estimates the Gaussian curvature of the physical part
- T042 [physical]: The refusal catalogue covers every defect present in real scanned surface data
- T043 [calibration]: Isotropic Gaussian vertex noise of the tested sigma describes the error of a real scanner
- T044 [sensor_performance]: A real marker-distance sensor on a real scanned part has this geometry and sensor variance split
- T044 [physical]: The geometry sigma of 1e-3 is the accuracy of a real scanned surface
- T045 [sensor_performance]: The declared noise-model parameters describe real instruments of these modes
- T046 [physical]: The chord correction computed from nominal curvature holds for chords measured on a physical part
- T047 [physical]: Chords measured between physical markers on a cylindrical part follow the cos^4(alpha)/(24 R^2) coefficient
- T048 [sensor_performance]: A physical stereo rig with this geometry achieves the synthetic chord accuracy
- T049 [calibration]: The declared perturbation magnitudes bound the calibration error of a real stereo rig
- T050 [calibration]: A two-term radial plus tangential Brown-Conrady model describes a real lens to the required accuracy
- T050 [sensor_performance]: Real circular markers and their detector show only the modelled perspective centre bias
- T051 [sensor_performance]: Real image noise is Gaussian with sigma = 0.25 px and marker localization rounds to whole pixels
- T052 [physical]: A real encoder drive train behaves as a constant-width play operator with constant scale and bias
- T053 [sensor_performance]: Real gyroscopes have constant bias and white rate noise with the declared densities
- T054 [physical]: Real sensor clocks have the declared constant offset and white jitter
- T055 [sensor_performance]: Real links drop observations as Bernoulli or two-state burst processes with these rates
- T056 [physical]: Real tracker latencies and target speeds match the declared values
- T057 [sensor_performance]: A real tracker's measurement noise and target motion match the constant-velocity model
- T058 [calibration]: The declared frame and clock mappings equal the real extrinsic calibration and clock synchronization
- T060 [sensor_performance]: The bench's noise levels, rates and motion describe real camera, encoder, IMU or tracker hardware
- T061 [sensor_performance]: Real sensors' noise covariance equals the covariance declared for this bench
- T062 [sensor_performance]: Real camera and tracker noises share the common-mode covariance assumed here, or are independent
- T063 [calibration]: The declared 35 degree rotation, translation and body covariance describe a real sensor mounting or extrinsic calibration
- T064 [physical]: The declared [lateral, heading] covariance and these surfaces predict the path uncertainty of a real vehicle or tool on a real curved part
- T065 [sensor_performance]: A real sensor's internally filtered output can be fused downstream as white noise
- T066 [sensor_performance]: A real residual monitor normalized by datasheet sensor covariance is correctly calibrated
- T067 [sensor_performance]: A real gate at the 99% quantile rejects 1% of valid real readings
- T068 [sensor_performance]: Real outliers are rare, isolated and of fixed magnitude as in this contamination model
- T069 [sensor_performance]: Real sensor dropouts are independent of the state and of the noise, as assumed here
- T070 [calibration]: The recovered offset calibrates the clock of a real camera
- T071 [calibration]: A 2 degree rotation is the size of a real extrinsic calibration error
- T072 [calibration]: The validity interval [0, 60) reflects how long a real camera calibration stays valid
- T074 [physical]: The fused estimate equals the physical state of a real target within its covariance
- T075 [calibration]: The declared tracker latency, synchronization and room-to-cell mapping are those of a real tracker installation
- T077 [physical]: The retained log's device and kernel identities identify the producing GPU and code
- T078 [physical]: The retained energy logs are real GPU energy measurements
- T097 [physical]: The integer heat field describes physical heat diffusion in a material
- T100 [physical]: The relabelled energy log is a physical GPU energy measurement
- T100 [sensor_performance]: The synthetic energy fixture characterizes real NVML counter accuracy
- T105 [physical]: The declared stiffness box contains the stiffness of a real axis
- T112 [physical]: The disturbance bound w_bar = 0.5 holds for a physical plant
- T113 [sensor_performance]: The synthetic residual statistics describe a real encoder's performance
- T113 [sensor_performance]: The EKF standard error of theta covers the true parameter of a real axis at the stated rate
- T113 [calibration]: The calibration referenced in the host envelope is valid
- T114 [physical]: The placeholder inertia interval contains the real axis inertia
- T114 [calibration]: Encoder and current-sensor calibrations of the bench are valid
- T122 [physical]: The variational free energy of this model equals a thermodynamic free energy of a physical system
- T123 [physical]: A decrease of variational free energy corresponds to a decrease of physical energy consumed by the computation
- T125 [physical]: Replayed energy values are physically valid measurements
- T126 [physical]: Measured marker chords on the physical plate equal the predicted geodesic distances within instrument uncertainty
- T126 [calibration]: The declared instrument uncertainties (camera 0.02 mm, tracker 0.015 mm, CMM 0.002 mm) hold for the instruments that will be used
- T127 [physical]: Measured chords and surface distances on the physical tube match the predicted gaps
- T127 [calibration]: The physical tube radius and roundness lie within the declared +/- 0.1 mm
- T127 [calibration]: The unrolled-film gauge achieves the declared 0.05 mm on marker-to-marker surface distances
- T128 [physical]: Measured separations of offset routes on the physical coupon follow the Jacobi prediction and cross at the predicted focal point
- T128 [calibration]: The formed coupon matches the declared dome (height 10 mm, sigma 20 mm) within tolerance
- T128 [physical]: An unsteered 3 mm tape laid on the coupon follows a geodesic of the as-built surface (no in-plane bending, lift-off or slip)
- T128 [calibration]: The start jig realizes the relative start pose of the offset tape within the declared 0.05 mm and 0.5 mrad of its insert and the laying and 0.01 mm and 0.1 mrad per seating of the jig
- T129 [sensor_performance]: A real laser line scanner achieves the declared 0.01 mm point noise on the coupon surface (finish, incidence angle, speckle)
- T129 [physical]: The formed coupon passes the Gaussian model test and its fitted height and width lie within the declared forming tolerances
- T129 [calibration]: The declared scanner scale (20 ppm) and target registration uncertainties hold for the scan
- T130 [calibration]: The physical gauge sphere, step gauge and scale bar have their certified dimensions, and the lab frame chain has the declared covariances
- T131 [sensor_performance]: The real gage (instrument, fixture and operators) has %GRR below 10% on the coupon features
- T132 [physical]: Tows placed by a real AFP head follow the programmed course within the stack, with gaps and overlaps inside 0.5 mm
- T132 [physical]: The declared 635 mm minimum steering radius avoids tow wrinkling for the placed material
- T133 [physical]: Fibre does not slip on a real mandrel wherever abs(kappa_g / kappa_n) <= 0.2 (the friction coefficient is declared, not measured)
- T134 [physical]: The torch or gun on a real cell stays within the predicted lateral and standoff band
- T135 [sensor_performance]: The real scanner footprint is a 20 mm swath on this surface at the planned standoff
- T136 [calibration]: The robot, fixture and frame calibration achieves the required heading and lateral tolerances
- T137 [physical]: Physical paths near a predicted focus show the predicted loss of lateral-error ordering
- T138 [physical]: Measured separation on the coupon agrees with the prediction (E_n <= 1 at every station)
- T138 [calibration]: The start jig realizes the relative start pose of the offset tape within the declared 0.05 mm and 0.5 mrad of its insert and the laying and 0.01 mm and 0.1 mrad per seating of the jig
- T139 [physical]: A real measurement with raw bytes, calibration and frame metadata has been retained
- T140 [calibration]: The declared instrument uncertainties are the uncertainties of the instruments used
- T140 [calibration]: The start jig realizes the relative start pose of the offset tape within the declared 0.05 mm and 0.5 mrad of its insert and the laying and 0.01 mm and 0.1 mrad per seating of the jig
- T140 [physical]: The budget contains every significant physical error source (thermal, fixturing, tape bending along the route, target centring)
- T143 [sensor_performance]: Vendor camera SDK acquisition meets its timing on real cameras
- T149 [physical]: The frame format works on real FPGA links
- T150 [physical]: A real bitstream with this identity exists and is loaded on hardware
- T152 [physical]: Simulated loss, latency and staleness represent the real FPGA telemetry link

## Physical and authority claims not established

- T005 [physical]: Nearby real trajectories on a physical curved surface separate according to this Jacobi law
- T009 [physical]: The ranking of lateral versus heading start errors computed here predicts which error dominates the endpoint error of real tool or vehicle paths on physical curved parts
- T013 [physical]: A physical cylinder or large-radius torus workpiece shows these separations
- T017 [machine_safety]: These validity domains certify first-order path corrections as safe on real machines
- T018 [sensor_performance]: Curvature signals resolvable here would be resolvable in measured sensor data
- T021 [machine_safety]: The shortest route on a physical flat workpiece is the safest route to execute
- T023 [sensor_performance]: A physical heading sensor closes the shortest loop within the computed heading tolerance
- T024 [machine_safety]: Ranking routes by focus margin s_c - L selects a route that is safe to execute on a physical part
- T025 [machine_safety]: A Pareto-optimal route is safe to execute on a physical part
- T030 [physical]: Grid-planned path lengths predict distances travelled by a physical vehicle or tool
- T031 [calibration]: A metric calibrated from physical measurements is accurate enough to decide between near-tied routes
- T032 [machine_safety]: A route from this library is safe (or unsafe) to execute on a physical part or vehicle
- T033 [physical]: The conformance suite certifies surfaces reconstructed from physical measurements
- T034 [physical]: Symbolic and dual-number derivative agreement certifies derivatives of surfaces reconstructed from physical measurements
- T035 [physical]: The optimal-step law derived here applies to derivatives of measured surface samples
- T036 [industrial_readiness]: Chart-switching geodesic integration is ready for tool paths over physical parts
- T037 [physical]: The singularity classification applies to scanned physical parts
- T038 [physical]: Straightest geodesics on a mesh reconstructed from a real scan reproduce the geodesics of the scanned physical surface
- T039 [physical]: The observed convergence orders transfer to meshes reconstructed from real scans
- T040 [physical]: Angle-defect curvature of a scanned mesh estimates the Gaussian curvature of the physical part
- T041 [production_acceptance]: A minimum-angle or radius-ratio threshold certifies a scanned mesh for production metrology
- T042 [physical]: The refusal catalogue covers every defect present in real scanned surface data
- T043 [calibration]: Isotropic Gaussian vertex noise of the tested sigma describes the error of a real scanner
- T044 [sensor_performance]: A real marker-distance sensor on a real scanned part has this geometry and sensor variance split
- T044 [physical]: The geometry sigma of 1e-3 is the accuracy of a real scanned surface
- T045 [sensor_performance]: The declared noise-model parameters describe real instruments of these modes
- T046 [physical]: The chord correction computed from nominal curvature holds for chords measured on a physical part
- T047 [physical]: Chords measured between physical markers on a cylindrical part follow the cos^4(alpha)/(24 R^2) coefficient
- T048 [sensor_performance]: A physical stereo rig with this geometry achieves the synthetic chord accuracy
- T049 [calibration]: The declared perturbation magnitudes bound the calibration error of a real stereo rig
- T050 [calibration]: A two-term radial plus tangential Brown-Conrady model describes a real lens to the required accuracy
- T050 [sensor_performance]: Real circular markers and their detector show only the modelled perspective centre bias
- T051 [sensor_performance]: Real image noise is Gaussian with sigma = 0.25 px and marker localization rounds to whole pixels
- T052 [physical]: A real encoder drive train behaves as a constant-width play operator with constant scale and bias
- T053 [sensor_performance]: Real gyroscopes have constant bias and white rate noise with the declared densities
- T054 [physical]: Real sensor clocks have the declared constant offset and white jitter
- T055 [sensor_performance]: Real links drop observations as Bernoulli or two-state burst processes with these rates
- T056 [physical]: Real tracker latencies and target speeds match the declared values
- T057 [sensor_performance]: A real tracker's measurement noise and target motion match the constant-velocity model
- T058 [calibration]: The declared frame and clock mappings equal the real extrinsic calibration and clock synchronization
- T059 [actuator_authority]: Admission as workbench state confers authority to act on a machine
- T060 [sensor_performance]: The bench's noise levels, rates and motion describe real camera, encoder, IMU or tracker hardware
- T061 [sensor_performance]: Real sensors' noise covariance equals the covariance declared for this bench
- T062 [sensor_performance]: Real camera and tracker noises share the common-mode covariance assumed here, or are independent
- T063 [calibration]: The declared 35 degree rotation, translation and body covariance describe a real sensor mounting or extrinsic calibration
- T064 [physical]: The declared [lateral, heading] covariance and these surfaces predict the path uncertainty of a real vehicle or tool on a real curved part
- T065 [sensor_performance]: A real sensor's internally filtered output can be fused downstream as white noise
- T066 [sensor_performance]: A real residual monitor normalized by datasheet sensor covariance is correctly calibrated
- T067 [sensor_performance]: A real gate at the 99% quantile rejects 1% of valid real readings
- T068 [sensor_performance]: Real outliers are rare, isolated and of fixed magnitude as in this contamination model
- T069 [sensor_performance]: Real sensor dropouts are independent of the state and of the noise, as assumed here
- T070 [calibration]: The recovered offset calibrates the clock of a real camera
- T071 [calibration]: A 2 degree rotation is the size of a real extrinsic calibration error
- T072 [calibration]: The validity interval [0, 60) reflects how long a real camera calibration stays valid
- T073 [machine_safety]: A 1 m, 99% track-loss radius is a safe operating threshold
- T074 [physical]: The fused estimate equals the physical state of a real target within its covariance
- T075 [actuator_authority]: An admitted synthetic state may command actuators
- T075 [calibration]: The declared tracker latency, synchronization and room-to-cell mapping are those of a real tracker installation
- T076 [production_acceptance]: Synthetic fusion output is admissible as production state
- T077 [physical]: The retained log's device and kernel identities identify the producing GPU and code
- T078 [physical]: The retained energy logs are real GPU energy measurements
- T089 [production_acceptance]: A retained replay receipt or verification authorizes admission of the replayed result into canonical state
- T092 [production_acceptance]: A content-consistent reopened bundle is acceptable as a verified production result
- T096 [production_acceptance]: Passing provider-free conformance admits a candidate into canonical state
- T097 [physical]: The integer heat field describes physical heat diffusion in a material
- T099 [production_acceptance]: A successful locked build makes the engine acceptable for production use
- T100 [physical]: The relabelled energy log is a physical GPU energy measurement
- T100 [sensor_performance]: The synthetic energy fixture characterizes real NVML counter accuracy
- T105 [physical]: The declared stiffness box contains the stiffness of a real axis
- T106 [actuator_authority]: A CERTIFIED_WITH_MARGIN verdict (operationally_acceptable) authorizes actuation
- T112 [physical]: The disturbance bound w_bar = 0.5 holds for a physical plant
- T112 [machine_safety]: The ISS bound defines a safe operating envelope for a machine
- T113 [sensor_performance]: The synthetic residual statistics describe a real encoder's performance
- T113 [sensor_performance]: The EKF standard error of theta covers the true parameter of a real axis at the stated rate
- T113 [calibration]: The calibration referenced in the host envelope is valid
- T114 [machine_safety]: The servo-axis pilot is safe to operate
- T114 [actuator_authority]: The Lyapunov monitor may command, gate or release the axis
- T114 [production_acceptance]: The pilot configuration is acceptable for production use
- T114 [industrial_readiness]: The monitor is ready for industrial deployment
- T114 [physical]: The placeholder inertia interval contains the real axis inertia
- T114 [calibration]: Encoder and current-sensor calibrations of the bench are valid
- T115 [physical]: Gross CPU package energy (background-inclusive, idle not subtracted) per geodesic trajectory
- T115 [physical]: Idle-subtracted CPU package energy per geodesic trajectory (equal-length idle bracket after the workload)
- T115 [physical]: Gross CPU package energy (background-inclusive, idle not subtracted) per batch of the common Gaussian VI workload (NumPy float64 reference)
- T115 [physical]: Idle-subtracted CPU package energy per batch of the common Gaussian VI workload (NumPy float64 reference)
- T116 [physical]: GPU-domain gross energy per measured batch
- T116 [sensor_performance]: The NVML total-energy counter of the RTX 2080 has a characterized accuracy and resolution
- T117 [physical]: CPU package energy per batch of the common Gaussian VI workload: Rust port and NumPy reference (RAPL, gross and idle-subtracted)
- T118 [physical]: RTX 2080 power draw during the measurement phase (NVML)
- T118 [physical]: RTX 2080 temperature during the measurement phase (NVML)
- T118 [physical]: RTX 2080 graphics clock during the measurement phase (NVML)
- T118 [physical]: Host-bracketed batch solve duration (launch, sync and copy included)
- T118 [physical]: RTX 2080 GPU utilization during the measurement phase (nvidia-smi rows inside the measurement window)
- T118 [physical]: RTX 2080 power draw is steady over the measurement phase (coefficient of variation <= 0.10)
- T118 [physical]: RTX 2080 temperature drifts by at most 5 C over the measurement phase
- T118 [physical]: RTX 2080 kernel-only duration of the Gaussian VI kernel
- T119 [physical]: Physical GPU energy per accepted numerical result
- T120 [physical]: CPU package energy per batch of the common Gaussian VI workload in float32 and in float64 (NumPy reference, RAPL, gross and idle-subtracted)
- T120 [physical]: GPU energy per batch of a float32 build of the common Gaussian VI workload
- T122 [physical]: The variational free energy of this model equals a thermodynamic free energy of a physical system
- T123 [physical]: A decrease of variational free energy corresponds to a decrease of physical energy consumed by the computation
- T124 [physical]: The fixtures' counter readings were produced by a physical GPU and NVML counter
- T124 [physical]: The bound operator log names an NVML device present on this analyzing host (UUID, name, driver, NVML version and NVML library digest equal this host's; the capture is unauthenticated)
- T125 [physical]: Replayed energy values are physically valid measurements
- T126 [physical]: Measured marker chords on the physical plate equal the predicted geodesic distances within instrument uncertainty
- T126 [calibration]: The declared instrument uncertainties (camera 0.02 mm, tracker 0.015 mm, CMM 0.002 mm) hold for the instruments that will be used
- T127 [physical]: Measured chords and surface distances on the physical tube match the predicted gaps
- T127 [calibration]: The physical tube radius and roundness lie within the declared +/- 0.1 mm
- T127 [calibration]: The unrolled-film gauge achieves the declared 0.05 mm on marker-to-marker surface distances
- T128 [physical]: Measured separations of offset routes on the physical coupon follow the Jacobi prediction and cross at the predicted focal point
- T128 [calibration]: The formed coupon matches the declared dome (height 10 mm, sigma 20 mm) within tolerance
- T128 [physical]: An unsteered 3 mm tape laid on the coupon follows a geodesic of the as-built surface (no in-plane bending, lift-off or slip)
- T128 [calibration]: The start jig realizes the relative start pose of the offset tape within the declared 0.05 mm and 0.5 mrad of its insert and the laying and 0.01 mm and 0.1 mrad per seating of the jig
- T129 [sensor_performance]: A real laser line scanner achieves the declared 0.01 mm point noise on the coupon surface (finish, incidence angle, speckle)
- T129 [physical]: The formed coupon passes the Gaussian model test and its fitted height and width lie within the declared forming tolerances
- T129 [calibration]: The declared scanner scale (20 ppm) and target registration uncertainties hold for the scan
- T130 [calibration]: The physical gauge sphere, step gauge and scale bar have their certified dimensions, and the lab frame chain has the declared covariances
- T131 [sensor_performance]: The real gage (instrument, fixture and operators) has %GRR below 10% on the coupon features
- T131 [production_acceptance]: The measurement system is approved for production use
- T132 [physical]: Tows placed by a real AFP head follow the programmed course within the stack, with gaps and overlaps inside 0.5 mm
- T132 [physical]: The declared 635 mm minimum steering radius avoids tow wrinkling for the placed material
- T133 [physical]: Fibre does not slip on a real mandrel wherever abs(kappa_g / kappa_n) <= 0.2 (the friction coefficient is declared, not measured)
- T134 [physical]: The torch or gun on a real cell stays within the predicted lateral and standoff band
- T134 [machine_safety]: The trajectory is safe to execute on a welding or coating robot cell
- T135 [sensor_performance]: The real scanner footprint is a 20 mm swath on this surface at the planned standoff
- T136 [calibration]: The robot, fixture and frame calibration achieves the required heading and lateral tolerances
- T137 [physical]: Physical paths near a predicted focus show the predicted loss of lateral-error ordering
- T138 [physical]: Measured separation on the coupon agrees with the prediction (E_n <= 1 at every station)
- T138 [calibration]: The start jig realizes the relative start pose of the offset tape within the declared 0.05 mm and 0.5 mrad of its insert and the laying and 0.01 mm and 0.1 mrad per seating of the jig
- T139 [physical]: A real measurement with raw bytes, calibration and frame metadata has been retained
- T140 [calibration]: The declared instrument uncertainties are the uncertainties of the instruments used
- T140 [calibration]: The start jig realizes the relative start pose of the offset tape within the declared 0.05 mm and 0.5 mrad of its insert and the laying and 0.01 mm and 0.1 mrad per seating of the jig
- T140 [physical]: The budget contains every significant physical error source (thermal, fixturing, tape bending along the route, target centring)
- T141 [production_acceptance]: Production acceptance of the coupon, cylinder or plate process
- T141 [industrial_readiness]: The manufacturing protocols and models are ready for industrial use
- T141 [customer_demand]: Manufacturers need curvature-aware placement, winding, coating or inspection path checking
- T142 [industrial_readiness]: Rust ports of the ranked kernels are ready for industrial deployment
- T143 [industrial_readiness]: The listed interfaces are qualified for plant integration
- T143 [sensor_performance]: Vendor camera SDK acquisition meets its timing on real cameras
- T144 [machine_safety]: Keeping native code behind subprocess boundaries makes machine interfaces safe
- T147 [industrial_readiness]: GPU/CPU agreement establishes industrial readiness
- T149 [physical]: The frame format works on real FPGA links
- T149 [machine_safety]: A telemetry-only interface guarantees the FPGA cannot actuate the machine
- T150 [physical]: A real bitstream with this identity exists and is loaded on hardware
- T150 [production_acceptance]: The bitstream is approved for production deployment
- T151 [machine_safety]: The rollback procedure is safe to execute on a production machine
- T151 [production_acceptance]: Rollback records are accepted for production change control
- T152 [physical]: Simulated loss, latency and staleness represent the real FPGA telemetry link
- T153 [actuator_authority]: The lab holds actuator write authority
- T153 [machine_safety]: Disabled-by-default software writes make the machine safe
- T154 [actuator_authority]: Heading proposals are authorized for execution as actuator commands
- T154 [machine_safety]: Applying the proposed heading corrections on a machine is safe

## Computational boundary

What a computational experiment may establish, what it cannot establish alone, and the not-established claims filed in the domain that records it:

| May establish | Cannot establish alone | Domain | Not established |
| --- | --- | --- | --- |
| Analytic agreement | Physical truth | physical | 65 |
| Numerical convergence | Calibration validity | calibration | 23 |
| Synthetic sensor performance | Real sensor performance | sensor_performance | 26 |
| Replay determinism | Independent verification by another party | (outside the queue) | - |
| Schema/provenance integrity | Machine safety | machine_safety | 14 |
| GPU/CPU agreement | Industrial readiness | industrial_readiness | 6 |
| A plausible use case | Actual customer demand | customer_demand | 1 |
| A simulated control response | Safe actuator authority | actuator_authority | 6 |
| - | Production acceptance | production_acceptance | 11 |

Customer demand: 1 retained claims filed in the customer_demand domain, none established; a plausible use case is not demand.

## Cross-cutting open items

No queue task closes these; each needs a queue extension. Statements are grouped by phrases in each report's own assumptions, next step and unestablished claims (research_portfolio.OPEN_ITEMS).

### Key custody and signatures

Closes it: a signing key held outside the workspace and signatures over workspace records, receipts, operator captures and the release digest (a queue extension; content identities and seals are unkeyed hashes).

- T005 (next step): Deferred research question (hardware-gated): measure the separation law on real neighbouring trajectories (for example two tracked markers on great circles of a sphere and on a saddle, with a calibrated position sensor) and compare the measured separation with eps j(s) within the sensor's calibrated uncertainty. Route: T005 would read the tracker export as an operator capture (ctx.capture('trajectory-log'), bound with ciw lab run T005 --capture trajectory-log=PATH) and fit the separation from it as a computational finding; the physical finding needs, besides an acquisition record (device, raw_sha256 of the captured bytes, acquired_at, calibration), a probe of the tracker on the analysing host that succeeds in T005 (runner.CAPTURE_INSTRUMENTS has no entry for trajectory-log) or a signed-capture trust anchor, and the run is retained with ciw lab hardware retain under lab/hardware/<run-id>. Neither the capture reader nor a tracker probe exists, so the physical-domain finding stays not_established even when such data exist
- T043 (next step): Deferred research question: re-trace the geodesic per Monte Carlo sample past the strip-dependent corridor threshold, where the fixed-strip distance stops being the geodesic distance, and propagate a correlated, anisotropic vertex covariance and independently measured normals. A realistic scanner covariance needs acquired scan data (hardware-gated). Route: T043 would read the scan export as an operator capture (ctx.capture('scan-export'), bound with ciw lab run T043 --capture scan-export=PATH) and fit the covariance from it as a computational finding; a scanner claim needs, besides an acquisition record (device, raw_sha256 of the captured bytes, acquired_at, calibration), a probe of the scanner on the analysing host that succeeds in T043 (runner.CAPTURE_INSTRUMENTS has no entry for scan-export) or a signed-capture trust anchor, and the run is retained with ciw lab hardware retain under lab/hardware/<run-id>. Neither the capture reader nor a scanner probe exists, so T043's scanner claims stay not_established even when such data exist.
- T045 (next step): Deferred research question: a mesh-reconstruction conversion that fills geometry_m2 = sigma_vertex^2 |grad d|^2 from T043's linearized vertex-noise gain (valid only while the unfolded segment stays in its face corridor), checked against T044's nested Monte Carlo, so a reconstructed_surface_distance derived from a scanned mesh carries the same split as one derived from a parametric model. A physical check of the split needs paired chord and tape readings. The runner can retain their raw export as an operator capture (ciw lab run --capture ROLE=PATH), but no section-4 task reads a capture or parses a camera, tracker or tape export, so a capture parser for chord and tape readings is missing; and a capture is unauthenticated, so without an instrument probe or a signed-capture trust anchor for these roles its physical findings stay not_established. No hardware_measured observation exists.
- T077 (assumption): Deferred research question (key custody and signatures): which key signs CIW workspace records, replay receipts and execution records, who holds it, how is it provisioned, rotated and revoked, and how does a verifier obtain its public key? Every seal and identity on these paths is an unkeyed SHA-256, so 'Retained workspace records are authenticated' stays not_established until records are signed (for example Ed25519 over the canonical record bytes, with an RFC 3161 timestamp for created_at) and T077 and T080-T090 are re-run against the signed records to check that every surviving forgery is refused.
- T077 (finding): Retained workspace records are authenticated: a holder without a secret cannot produce a forged record that reopens
- T078 (assumption): Deferred research question (key custody and signatures): which key signs CIW workspace records, replay receipts and execution records, who holds it, how is it provisioned, rotated and revoked, and how does a verifier obtain its public key? Every seal and identity on these paths is an unkeyed SHA-256, so 'Retained workspace records are authenticated' stays not_established until records are signed (for example Ed25519 over the canonical record bytes, with an RFC 3161 timestamp for created_at) and T077 and T080-T090 are re-run against the signed records to check that every surviving forgery is refused.
- T080 (assumption): Deferred research question (key custody and signatures): which key signs CIW workspace records, replay receipts and execution records, who holds it, how is it provisioned, rotated and revoked, and how does a verifier obtain its public key? Every seal and identity on these paths is an unkeyed SHA-256, so 'Retained workspace records are authenticated' stays not_established until records are signed (for example Ed25519 over the canonical record bytes, with an RFC 3161 timestamp for created_at) and T077 and T080-T090 are re-run against the signed records to check that every surviving forgery is refused.
- T080 (finding): Retained workspace records are authenticated: a holder without a secret cannot produce a forged record that reopens
- T081 (assumption): Deferred research question (key custody and signatures): which key signs CIW workspace records, replay receipts and execution records, who holds it, how is it provisioned, rotated and revoked, and how does a verifier obtain its public key? Every seal and identity on these paths is an unkeyed SHA-256, so 'Retained workspace records are authenticated' stays not_established until records are signed (for example Ed25519 over the canonical record bytes, with an RFC 3161 timestamp for created_at) and T077 and T080-T090 are re-run against the signed records to check that every surviving forgery is refused.
- T081 (finding): Retained workspace records are authenticated: a holder without a secret cannot produce a forged record that reopens
- T082 (assumption): Deferred research question (key custody and signatures): which key signs CIW workspace records, replay receipts and execution records, who holds it, how is it provisioned, rotated and revoked, and how does a verifier obtain its public key? Every seal and identity on these paths is an unkeyed SHA-256, so 'Retained workspace records are authenticated' stays not_established until records are signed (for example Ed25519 over the canonical record bytes, with an RFC 3161 timestamp for created_at) and T077 and T080-T090 are re-run against the signed records to check that every surviving forgery is refused.
- T082 (next step): Deferred research question (CIW change): keyed or externally timestamped execution records (for example an RFC 3161 timestamp over the execution record digest) so that created_at carries evidential weight, and a registry of occurrences across workspaces so that two workspaces reusing one occurrence identity are detected.
- T082 (finding): Retained workspace records are authenticated: a holder without a secret cannot produce a forged record that reopens
- T083 (assumption): Deferred research question (key custody and signatures): which key signs CIW workspace records, replay receipts and execution records, who holds it, how is it provisioned, rotated and revoked, and how does a verifier obtain its public key? Every seal and identity on these paths is an unkeyed SHA-256, so 'Retained workspace records are authenticated' stays not_established until records are signed (for example Ed25519 over the canonical record bytes, with an RFC 3161 timestamp for created_at) and T077 and T080-T090 are re-run against the signed records to check that every surviving forgery is refused.
- T083 (next step): Deferred research question (CIW change): bind replay receipts into the replay bundle identity or a catalog-level seal so that an accidental deletion is detected, and sign receipts at replay time with a host-held key under a defined key custody so that the retained fabricated sibling receipt (B0b) and a transplanted receipt are refused.
- T083 (finding): Retained workspace records are authenticated: a holder without a secret cannot produce a forged record that reopens
- T084 (assumption): Deferred research question (key custody and signatures): which key signs CIW workspace records, replay receipts and execution records, who holds it, how is it provisioned, rotated and revoked, and how does a verifier obtain its public key? Every seal and identity on these paths is an unkeyed SHA-256, so 'Retained workspace records are authenticated' stays not_established until records are signed (for example Ed25519 over the canonical record bytes, with an RFC 3161 timestamp for created_at) and T077 and T080-T090 are re-run against the signed records to check that every surviving forgery is refused.
- T084 (next step): Deferred research question (CIW change): bind each replay receipt to the source execution's own identity as well as its bytes, or sign it under a defined key custody, and re-run these mutations to check that receipt-source.sibling-execution is refused while every killed mutant keeps its pinned message.
- T084 (finding): Retained workspace records are authenticated: a holder without a secret cannot produce a forged record that reopens
- T085 (assumption): Deferred research question (key custody and signatures): which key signs CIW workspace records, replay receipts and execution records, who holds it, how is it provisioned, rotated and revoked, and how does a verifier obtain its public key? Every seal and identity on these paths is an unkeyed SHA-256, so 'Retained workspace records are authenticated' stays not_established until records are signed (for example Ed25519 over the canonical record bytes, with an RFC 3161 timestamp for created_at) and T077 and T080-T090 are re-run against the signed records to check that every surviving forgery is refused.
- T085 (finding): Retained workspace records are authenticated: a holder without a secret cannot produce a forged record that reopens
- T086 (assumption): Deferred research question (key custody and signatures): which key signs CIW workspace records, replay receipts and execution records, who holds it, how is it provisioned, rotated and revoked, and how does a verifier obtain its public key? Every seal and identity on these paths is an unkeyed SHA-256, so 'Retained workspace records are authenticated' stays not_established until records are signed (for example Ed25519 over the canonical record bytes, with an RFC 3161 timestamp for created_at) and T077 and T080-T090 are re-run against the signed records to check that every surviving forgery is refused.
- T086 (finding): Retained workspace records are authenticated: a holder without a secret cannot produce a forged record that reopens
- T087 (assumption): Deferred research question (key custody and signatures): which key signs CIW workspace records, replay receipts and execution records, who holds it, how is it provisioned, rotated and revoked, and how does a verifier obtain its public key? Every seal and identity on these paths is an unkeyed SHA-256, so 'Retained workspace records are authenticated' stays not_established until records are signed (for example Ed25519 over the canonical record bytes, with an RFC 3161 timestamp for created_at) and T077 and T080-T090 are re-run against the signed records to check that every surviving forgery is refused.
- T087 (finding): Retained workspace records are authenticated: a holder without a secret cannot produce a forged record that reopens
- T088 (assumption): Deferred research question (key custody and signatures): which key signs CIW workspace records, replay receipts and execution records, who holds it, how is it provisioned, rotated and revoked, and how does a verifier obtain its public key? Every seal and identity on these paths is an unkeyed SHA-256, so 'Retained workspace records are authenticated' stays not_established until records are signed (for example Ed25519 over the canonical record bytes, with an RFC 3161 timestamp for created_at) and T077 and T080-T090 are re-run against the signed records to check that every surviving forgery is refused.
- T088 (next step): Deferred research question: run exchange inspection through inspect_exchange with the pinned State Estimation Evaluation Testbed validator (the set provider, CIW_SET_REPO), which the pure exchange._identity rows here do not reach, and check whether a recomputed artifact claiming independent: true is still accepted as content_recomputed_not_authenticated.
- T088 (finding): Retained workspace records are authenticated: a holder without a secret cannot produce a forged record that reopens
- T089 (assumption): Deferred research question (key custody and signatures): which key signs CIW workspace records, replay receipts and execution records, who holds it, how is it provisioned, rotated and revoked, and how does a verifier obtain its public key? Every seal and identity on these paths is an unkeyed SHA-256, so 'Retained workspace records are authenticated' stays not_established until records are signed (for example Ed25519 over the canonical record bytes, with an RFC 3161 timestamp for created_at) and T077 and T080-T090 are re-run against the signed records to check that every surviving forgery is refused.
- T089 (next step): Deferred research question: define a signed admission authorization record (signer, key custody, scope of the admitted result) that CIW would require before any state_admission other than not_performed, and re-run to check that oscillator-admission.injected is refused (admission itself stays an authority decision, a production_acceptance claim never established here).
- T089 (finding): Retained workspace records are authenticated: a holder without a secret cannot produce a forged record that reopens
- T090 (assumption): Deferred research question (key custody and signatures): which key signs CIW workspace records, replay receipts and execution records, who holds it, how is it provisioned, rotated and revoked, and how does a verifier obtain its public key? Every seal and identity on these paths is an unkeyed SHA-256, so 'Retained workspace records are authenticated' stays not_established until records are signed (for example Ed25519 over the canonical record bytes, with an RFC 3161 timestamp for created_at) and T077 and T080-T090 are re-run against the signed records to check that every surviving forgery is refused.
- T090 (finding): Retained workspace records are authenticated: a holder without a secret cannot produce a forged record that reopens
- T098 (next step): Deferred research question: authenticate what the digests only identify, by verifying signed upstream commits or tags of each bound checkout against its maintainers' published keys and recording digests of the Rust and Python toolchains, since a matching HEAD, tree and lock digest is not authentication; and settle why CIW pins two SCR revisions (a59aba2 for declared-workload and proved-heat, 5f04097 for the exchange workflow) and several SET revisions, converging them or recording each workflow's reason.
- T098 (finding): A matching HEAD, tree and lock digest authenticates the upstream repository, toolchain and built engine
- T100 (assumption): The Basis column shows the components and identities a finding declares; they are as recorded, not authenticated (a generator name or provider repository is text in the basis).
- T100 (next step): Deferred research question (CIW change): key the workspace seals, or have the provider sign its runtime identity at execution under a defined key custody, so that a bundle copying CIW's public pins (the retained copied-pin counterexample) is refused or classified not_established; compare a retained runtime identity with CIW's pins on reopen (Session.from_workspace), as ciw lab classify does; record in CIW's pin tables the source tree of every pinned revision that has none (the kinds ciw.lab.bridge.pins_without_tree lists, such as telemetry and calibrated-observable), so that the invented-tree check reaches them; and authenticate energy-log origin at acquisition, outside the workbench, so that a resealed relabel under a fresh occurrence is detected.
- T113 (next step): Deferred research question (hardware-gated): bind the residual adapter to acquired encoder data and check the coverage of its three-standard-error interval there, which synthetic EKF data cannot establish (a sensor_performance claim). Route: T113 would read the encoder log as an operator capture (ctx.capture('encoder-log'), bound with ciw lab run T113 --capture encoder-log=PATH) and run the residual adapter on it as a computational finding; its sensor_performance and calibration findings need, besides an acquisition record (device, raw_sha256 of the captured bytes, acquired_at, calibration), a probe of the encoder on the analysing host that succeeds in T113 (runner.CAPTURE_INSTRUMENTS has no entry for encoder-log) or a signed-capture trust anchor, and the run is retained with ciw lab hardware retain under lab/hardware/<run-id>. Neither the capture reader nor a probe of the encoder exists, so they stay not_established even when such data exist.
- T114 (next step): Deferred research question (hardware-gated): identify J, friction and delay on the bench, replace the placeholder interval, re-run this check with an affine over-approximation suitable for PLSR's box semantics, then run T113's adapter on acquired encoder data. Route: T114 would read the servo bench log as an operator capture (ctx.capture('servo-bench-log'), bound with ciw lab run T114 --capture servo-bench-log=PATH) and identify the J, friction and delay interval from it as a computational finding; its physical and calibration findings need, besides an acquisition record (device, raw_sha256 of the captured bytes, acquired_at, calibration), a probe of the bench instrument on the analysing host that succeeds in T114 (runner.CAPTURE_INSTRUMENTS has no entry for servo-bench-log) or a signed-capture trust anchor, and the run is retained with ciw lab hardware retain under lab/hardware/<run-id>. Neither the capture reader nor a probe of the bench instrument exists, so they stay not_established even when such data exist.
- T116 (assumption): An operator log is an unauthenticated record; the gate binds it to this host's NVML identity but cannot prove the capture genuine
- T119 (assumption): An operator log is an unauthenticated record; the gate binds it to this host's NVML identity but cannot prove the capture genuine
- T124 (assumption): Sealing is integrity, not authenticity: no signature binds a log to the device
- T124 (assumption): A real device's identity is bound only by the acquisition gate on the GPU host: the gate binds a log to that host's NVML identity, it does not authenticate the capture
- T124 (assumption): Deferred research question: a signed capture that binds a log's readings to the device (no signature scheme exists)
- T124 (next step): Retain the first RTX 2080 log with its device and runtime identity through the acquisition gate on the RTX 2080 host (protocol in docs/lab/ENERGY_GPU.md): `mkdir -p runs/rtx2080-<date>`; `ciw energy probe --gpu-index 0` must return a reading with status ok; start `TZ=UTC nvidia-smi --query-gpu=timestamp,uuid,name,utilization.gpu,utilization.memory,temperature.gpu,power.draw,clocks.sm,clocks.mem,pstate --format=csv,nounits -lms 100 -f runs/rtx2080-<date>/smi.csv` in the background; `ciw energy record --problem examples/energy-accuracy/problem.json --output-dir runs/rtx2080-<date>/capture --duration 10 --replicas 4096 --warmup-batches 2 --idle-duration 2 --gpu-index 0`; stop the sidecar; `ciw energy replay runs/rtx2080-<date>/capture/log.json`; then `CIW_LAB_NVIDIA_SMI_UTC_OFFSET=+00:00 ciw lab run T116 T117 T118 T119 T120 T121 T124 T147 --capture energy-log=runs/rtx2080-<date>/capture/log.json --capture nvidia-smi-csv=runs/rtx2080-<date>/smi.csv --output-dir results/rtx2080-<date>` on the same host and `ciw lab hardware retain results/rtx2080-<date> --retained lab --run-id rtx2080-<date> --host "<RTX 2080 host>"`. Deferred research question: a signed capture that binds a log's readings to the device (no signature scheme exists; the gate binds a log only to the host's NVML identity)
- T124 (finding): The bound operator log names an NVML device present on this analyzing host (UUID, name, driver, NVML version and NVML library digest equal this host's; the capture is unauthenticated)
- T126 (next step): Acquire MFG-FLAT-PLATE-01 and bind its camera export to T138 (ciw lab run T138 --capture photogrammetry=<targets.csv>, format ciw.lab-mfg-target-capture.v1), then retain the run with ciw lab hardware retain; T138's pair comparison of the captured chords is computational, and a physical label needs a metrology instrument probe or a signed-capture trust anchor (deferred research question). Open: bound the plate flatness effect by measuring it rather than by the declared tolerance.
- T127 (next step): Acquire MFG-CYLINDER-01 and bind its exports to T138 (ciw lab run T138 --capture photogrammetry=<targets.csv> --capture film=<film.csv>), then retain the run with ciw lab hardware retain; T138's comparison of the captured gaps and chords is computational, and a physical label needs a metrology instrument probe or a signed-capture trust anchor (deferred research question). Open: model the seam weld and out-of-roundness, which the cylinder prediction omits.
- T128 (next step): Acquire MFG-SCAN-01 and MFG-COUPON-01 and bind the station-target and start-pose exports to T138 (ciw lab run T138 --capture photogrammetry=<targets.csv> --capture cmm=<start-pose.csv>), then retain the run with ciw lab hardware retain; the comparison is computational until a metrology instrument probe or a signed-capture trust anchor exists (deferred research question). Open: predict the separation on a surface fitted to the scan when the Gaussian model test rejects the as-built dome, which is not implemented.
- T138 (assumption): Operator captures are unauthenticated: their origin header and u column are declarations, and no metrology instrument probe exists on any analysing host. Deferred research question: a signed-capture trust anchor (an instrument-held key that signs each export, verified by the workbench) or a hardware:metrology probe of an instrument attached to the analysing host; until one exists, a bound capture yields computational comparisons only and the physical claims stay not_established.
- T138 (next step): Deferred research question: define a signed-capture trust anchor (instrument-held signing keys and their verification) or a hardware:metrology probe, without which no capture can support a physical label. Meanwhile acquire MFG-COUPON-01 and bind its exports (ciw lab run T138 --capture photogrammetry=<targets.csv> --capture cmm=<start-pose.csv>), then retain the run with ciw lab hardware retain.
- T138 (finding): Normalized errors of the bound metrology captures against the prediction of their protocol, computed from their unauthenticated bytes
- T139 (assumption): The validators cannot tell whether raw bytes came from an instrument; the runner also requires a hardware probe in the task that cites them, and no metrology instrument probe or signed-capture trust anchor exists (deferred research question, T138), so a retained record's acquisition fields support no physical label.
- T139 (next step): Deferred research question: a signed-capture trust anchor (instrument-held keys that sign each export) or a hardware:metrology probe, without which a retained record supports no physical label. Meanwhile retain the first real acquisition of a control protocol: run ciw lab run T139 --capture retention=<record.json> with its raw files bound under their roles (--capture photogrammetry=<targets.csv> --capture cmm=<start-pose.csv> ...), the record listing each file with its SHA-256, instrument serial, calibration certificate and validity window, frame chain with covariances and clock; then retain the run with ciw lab hardware retain.
- T139 (finding): A bound retention record validates against the raw bytes bound with it and yields acquisition fields (computational: the bytes are unauthenticated)
- T153 (assumption): Authorization record format and signature scheme are placeholders
- T153 (next step): Deferred research question: enforce that any future write transport is routed through ActuatorWritePolicy (the scan flags a new transport, routing is not enforced), with a real signature scheme for authorization records; enforcement itself belongs in hardware interlocks outside this package
- T160 (next step): Write the parts the lab cannot write (the thesis in the authors' words, the related-work discussion, the interpretation of the counterexamples and the conclusions) and submit the draft for external review; then the unfinished tasks' own next steps (T115: Measure package energy per geodesic trajectory and per common-workload batch on a Linux host with readable intel-rapl counters (docs/lab/ENERGY_GPU.md) …; T116: Measure gross device energy per batch of the common Gaussian VI workload on the RTX 2080 host (protocol in docs/lab/ENERGY_GPU.md), hardware_measured through …; T117: Compare the PTX kernel's outputs with the NumPy reference bit for bit on the RTX 2080 host (protocol in docs/lab/ENERGY_GPU.md): `mkdir -p …; T118: Record power, temperature, clock and utilization of the common workload on the RTX 2080 host (protocol in docs/lab/ENERGY_GPU.md): `mkdir -p …; T119: Measure energy per accepted replica solve of the common Gaussian VI workload on the RTX 2080 host (protocol in docs/lab/ENERGY_GPU.md), through the same …; T120: Measure float32 and float64 package energy per batch of the common workload on a Linux host with readable intel-rapl counters (docs/lab/ENERGY_GPU.md): `python …; T121: Check on the RTX 2080 host (protocol in docs/lab/ENERGY_GPU.md) that the PTX kernel keeps its in-kernel reductions unfused: `mkdir -p runs/rtx2080-<date>` …; T124: Retain the first RTX 2080 log with its device and runtime identity through the acquisition gate on the RTX 2080 host (protocol in docs/lab/ENERGY_GPU.md) …; T138: Deferred research question: define a signed-capture trust anchor (instrument-held signing keys and their verification) or a hardware:metrology probe, without …; T139: Deferred research question: a signed-capture trust anchor (instrument-held keys that sign each export) or a hardware:metrology probe, without which a retained …); 57 physical, calibration or sensor claims of completed tasks (T045, T046, T047, T048, T049, T050, T051, T052, T053, T054, T055, T056, T057, T058, T060, T061, T062, T063, T064, T065, T066, T067, T068, T069, T070, T071, T072, T074, T075, T122, T123, T125, T126, T127, T128, T129, T130, T131, T132, T133, T134, T135, T136, T137, T140) stay not established, and each task's own next step is listed under Outstanding work; 9 authority-domain claims stay not established on any evidence.
- T165 (next step): Sign the release digest with a project key once key custody is defined (a cross-cutting open item of T166 and T167, owned by no queue task); a whole-run release record, including T165-T168, is computed outside the queue with research_portfolio.release_record over all retained reports.
- T165 (finding): The release digest is signed by a project key

### Telemetry provider provisioning

Closes it: provider checkouts bound to the telemetry and declared-workload workflows (the pins of src/ciw/telemetry-runtimes.json), provisioned for the clean-room run like the other providers.

- T077 (assumption): Partial: ESM candidate_id and candidate execution identities (they need an operator-bound ESM adapter and a telemetry or calibrated bundle) and pinned-provider runtime identities (ciw.subprocess-runtime.v1; they need a provider checkout bound to a declared-workload or telemetry workflow) are planned matrix rows that the offline path cannot exercise; their predictions are read from the code, not observed.
- T077 (assumption): The ESM rows use a synthetic telemetry-shaped record (schema, digest, three step occurrences), not a telemetry session validated by the telemetry workflow, which needs provider checkouts.
- T077 (assumption): Provider-backed workflows (telemetry, declared workloads, proved heat) were not exercised offline; their identity rows are inferred only where they share the energy-accuracy code path.
- T077 (assumption): Deferred research question (telemetry provider stack): provision the telemetry stack pinned in src/ciw/telemetry-runtimes.json (ppda, stfe, gsie, set, cbsr; scripts/check_lab.py provisions only ppda and set of these, and scripts/reproduce_lab.py TEST_VARIABLES has no variable for stfe, gsie or cbsr), bind it to a telemetry or declared-workload workflow, and observe the ESM candidate, candidate execution and ciw.subprocess-runtime.v1 identity rows and telemetry-validated records that are now read from code or replaced by a synthetic telemetry-shaped record.
- T086 (assumption): Pure-validator rows run candidate_evidence.validate_response on a synthetic telemetry-shaped record (schema, digest, three step occurrences) that no telemetry workflow validated, without an ESM process, and exchange._identity without the pinned exchange validator.
- T086 (assumption): Deferred research question (telemetry provider stack): provision the telemetry stack pinned in src/ciw/telemetry-runtimes.json (ppda, stfe, gsie, set, cbsr; scripts/check_lab.py provisions only ppda and set of these, and scripts/reproduce_lab.py TEST_VARIABLES has no variable for stfe, gsie or cbsr), bind it to a telemetry or declared-workload workflow, and observe the ESM candidate, candidate execution and ciw.subprocess-runtime.v1 identity rows and telemetry-validated records that are now read from code or replaced by a synthetic telemetry-shaped record.
- T088 (assumption): Pure-validator rows run candidate_evidence.validate_response on a synthetic telemetry-shaped record (schema, digest, three step occurrences) that no telemetry workflow validated, without an ESM process, and exchange._identity without the pinned exchange validator.
- T088 (assumption): Deferred research question (telemetry provider stack): provision the telemetry stack pinned in src/ciw/telemetry-runtimes.json (ppda, stfe, gsie, set, cbsr; scripts/check_lab.py provisions only ppda and set of these, and scripts/reproduce_lab.py TEST_VARIABLES has no variable for stfe, gsie or cbsr), bind it to a telemetry or declared-workload workflow, and observe the ESM candidate, candidate execution and ciw.subprocess-runtime.v1 identity rows and telemetry-validated records that are now read from code or replaced by a synthetic telemetry-shaped record.
- T089 (assumption): Pure-validator rows run candidate_evidence.validate_response on a synthetic telemetry-shaped record (schema, digest, three step occurrences) that no telemetry workflow validated, without an ESM process.
- T089 (assumption): Deferred research question (telemetry provider stack): provision the telemetry stack pinned in src/ciw/telemetry-runtimes.json (ppda, stfe, gsie, set, cbsr; scripts/check_lab.py provisions only ppda and set of these, and scripts/reproduce_lab.py TEST_VARIABLES has no variable for stfe, gsie or cbsr), bind it to a telemetry or declared-workload workflow, and observe the ESM candidate, candidate execution and ciw.subprocess-runtime.v1 identity rows and telemetry-validated records that are now read from code or replaced by a synthetic telemetry-shaped record.
- T090 (assumption): Pinned-provider subprocess runtime identities (ciw.subprocess-runtime.v1: pinned revision, module and source_root, host-measured source_tree and python_sha256; src/ciw/declared_workload.py) were not mutated, because no provider checkout was bound to a declared-workload or telemetry workflow; only the CIW-internal oscillator provider identity and CIW's own energy analysis identity were.
- T090 (assumption): Deferred research question (telemetry provider stack): provision the telemetry stack pinned in src/ciw/telemetry-runtimes.json (ppda, stfe, gsie, set, cbsr; scripts/check_lab.py provisions only ppda and set of these, and scripts/reproduce_lab.py TEST_VARIABLES has no variable for stfe, gsie or cbsr), bind it to a telemetry or declared-workload workflow, and observe the ESM candidate, candidate execution and ciw.subprocess-runtime.v1 identity rows and telemetry-validated records that are now read from code or replaced by a synthetic telemetry-shaped record.
- T090 (next step): Deferred research question: mutate the pinned-provider subprocess runtime identities (ciw.subprocess-runtime.v1) once a provider checkout is bound to a declared-workload or telemetry workflow, and label a retained runtime that differs from the current analysis identity as historical and unverified on reopen (a CIW change), so that energy-runtime.all-bundles and energy-runtime.python-version no longer reopen as current.
- T092 (assumption): No ESM candidate adapter refusal ('No operator-bound ESM candidate adapter') is reached: that path needs a retained telemetry or calibrated-observable bundle, which needs bound providers. The ESM case here is refused for its bundle kind.
- T092 (next step): Deferred research question: observe, rather than infer, replay refusal for the provider kinds other than numerical-heat by retaining a bundle of each from its bound provider (the telemetry stack pinned in src/ciw/telemetry-runtimes.json for telemetry bundles, and the calibrated-observable stack pinned in src/ciw/calibrated-observable-runtimes.json for calibrated-observable bundles, both of which reach the 'No operator-bound ESM candidate adapter' refusal) and replaying it in a reopened unbound session; and add example sources for acquired-calibrated-window, bim-quantity, identified-stability and residual-monitor so that their execution refusal is observed too.
- T097 (assumption): The PPDA telemetry stack (ppda, stfe, gsie, set, cbsr at src/ciw/telemetry-runtimes.json pins) is a separate integration.
- T097 (next step): Deferred research question: provision the telemetry provider stack (ppda, stfe, gsie, set and cbsr at the src/ciw/telemetry-runtimes.json pins; scripts/check_lab.py provisions only ppda and set of these, at other pins) and drive CIW's telemetry workflow end to end against it, as the exchange roundtrip here does for PPDA, SCR and SET; and attest the SCR engine a bound replay uses (CIW records a bound engine as operator_asserted_not_attested), for example by requiring a locked build of the bound checkout in the same run before a replay is accepted.

### Cross-platform reproduction

Closes it: a non-gating clean-room run on Windows and on another BLAS build or architecture, compared with the retained reports by ciw lab verify, whose differences answer the platform assumptions.

- T014 (assumption): Deferred research question (cross-platform reproduction): run T014 on Windows x86-64 and macOS arm64 besides the retained Linux x86-64 run and check that the two bitwise claims (dyadic truncation and continuation reproduces direct integration; decimal truncation lengths change the step in the last bit) hold bit for bit there and that every other T014 finding value (reversal orders, adaptive return and restart ratios) matches the retained one within its regression tolerance; adaptive step sequences may differ in the last bit between platforms, and no second platform has been compared
- T081 (assumption): Deferred research question (cross-platform reproduction): run T081 on Windows x86-64 and macOS arm64 besides the retained Linux x86-64 run and check that every float-valued numerical_result_id of the energy-accuracy bundles (originals, sibling, replays, replay after reopen, separate session) equals the retained energy_numerical_result_id exactly (kept in T081's numerical-identity.json and in its identity finding's value, which ciw lab verify compares exactly); a mismatch would mean the canonical float serialization or the analysis arithmetic depends on the platform, and no second platform has been compared.
- T081 (next step): Deferred research question (CIW change): give oscillator operation results a content-level numerical_result_id so their replays can be linked, recompute statistics (or at least the moment inequalities) on reopen, and repeat the stability test in a separate process on a second host, since the separate session here shares the process and code.
- T094 (next step): Deferred research question: reopen the golden workspaces on a second platform (Windows x86-64, macOS arm64 or another NumPy/LAPACK build) to learn whether the bit-for-bit energy recomputation refuses the energy golden there, as the platform fingerprint in golden.json anticipates; and retain golden workspaces for provider kinds other than numerical-heat, produced by their pinned providers, so that validator drift in those kinds is caught at reopen.
- T101 (assumption): Deferred research question (cross-platform reproduction): run T101 and T102 against the pinned PLSR on Windows x86-64, on macOS arm64 and on Linux x86-64 with a LAPACK other than the retained OpenBLAS build (for example MKL), and check that the PLSR codes below the normal range (the subnormal witness's code included; resolution-floor.json) and the outside-window code flips and scaled-witness code (scaling-invariance.json) match the retained ones case for case, and that every finding value matches within its regression tolerance; these codes are retained as artifacts only because LAPACK/BLAS builds may decide them differently, and no second build has been compared.
- T102 (assumption): Deferred research question (cross-platform reproduction): run T101 and T102 against the pinned PLSR on Windows x86-64, on macOS arm64 and on Linux x86-64 with a LAPACK other than the retained OpenBLAS build (for example MKL), and check that the PLSR codes below the normal range (the subnormal witness's code included; resolution-floor.json) and the outside-window code flips and scaled-witness code (scaling-invariance.json) match the retained ones case for case, and that every finding value matches within its regression tolerance; these codes are retained as artifacts only because LAPACK/BLAS builds may decide them differently, and no second build has been compared.
- T117 (assumption): Cross-platform bitwise identity of the sphere kernel is not claimed
- T125 (next step): Deferred research question: replay a retained RTX 2080 log (lab/hardware/<run-id>/artifacts/T116/operator-log.json, once a GPU run is retained) through a Session on a second host and compare its numerical_result_id; whether the ids agree across platforms is open, because they hash binary64 analysis values and ciw.core.identities was not migrated to ciw.canonical-json.v1 (T146 kept the ASCII variant)
- T148 (assumption): Fixed-tree reproducibility was shown between two implementations on one platform; on other platforms it rests on IEEE-754 round-to-nearest addition and on the regression gate, which compares the retained first-order sum bits exactly
- T158 (assumption): Byte identity is established on one platform; Windows and other BLAS builds are not compared.
- T158 (next step): Have figure tasks declare timing figures in their reports, and re-execute every figure task on a second platform (Windows CI) outside the section's time budget.
- T164 (next step): Run scripts/check_lab.py under Python 3.12 on Windows and on a second Linux host and retain both gate records.

## Blocked, deferred and partial tasks

- T038 (partial): Trace declared geodesics; compare with exact developments; re-derive lengths by a second ciw implementation (strip layout from edge lengths); compare Dijkstra with a dense Floyd-Warshall and, when installed, scipy.sparse.csgraph; test every Steiner-graph edge for a common face; sandwich traced …
  - Unrun or unresolved: Partial delivery: the solver is an initial-value tracer (straightest geodesics from a point and heading) plus approximate distances. No exact two-point polyhedral geodesic (MMP, ICH or iterative edge flipping) is implemented, so no shortest path between two given points is solved exactly and the graph and heat distances are not compared with an exact polyhedral distance.
  - Unrun or unresolved: Vertex hits are refused rather than continued by the Polthier-Schmies angle-bisection rule.
  - Unrun or unresolved: Traced geodesics are shortest paths only when no shorter corridor exists; not proved here.
- T077 (partial): Build an oscillator session and energy-accuracy bundles offline, replay, save, reopen, replay again, replay a replay, and repeat in a separate session (same process and code); retain byte and resealed content variants; check every predicted property per identity, with CIW's own validators judging …
  - Unrun or unresolved: Partial: ESM candidate_id and candidate execution identities (they need an operator-bound ESM adapter and a telemetry or calibrated bundle) and pinned-provider runtime identities (ciw.subprocess-runtime.v1; they need a provider checkout bound to a declared-workload or telemetry workflow) are planned matrix rows that the offline path cannot exercise; their predictions are read from the code, not observed.
  - Unrun or unresolved: The ESM rows use a synthetic telemetry-shaped record (schema, digest, three step occurrences), not a telemetry session validated by the telemetry workflow, which needs provider checkouts.
  - Unrun or unresolved: Provider-backed workflows (telemetry, declared workloads, proved heat) were not exercised offline; their identity rows are inferred only where they share the energy-accuracy code path.
  - Unrun or unresolved: Reopen stability is observed as the value CIW saves again after reopening, which is the reopened session's own serialization of its restored state.
  - Unrun or unresolved: Content identities establish consistency, not authorship.
  - Unrun or unresolved: Deferred research question (telemetry provider stack): provision the telemetry stack pinned in src/ciw/telemetry-runtimes.json (ppda, stfe, gsie, set, cbsr; scripts/check_lab.py provisions only ppda and set of these, and scripts/reproduce_lab.py TEST_VARIABLES has no variable for stfe, gsie or cbsr), bind it to a telemetry or declared-workload workflow, and observe the ESM candidate, candidate execution and ciw.subprocess-runtime.v1 identity rows and telemetry-validated records that are now read from code or replaced by a synthetic telemetry-shaped record.
  - Unrun or unresolved: Deferred research question (key custody and signatures): which key signs CIW workspace records, replay receipts and execution records, who holds it, how is it provisioned, rotated and revoked, and how does a verifier obtain its public key? Every seal and identity on these paths is an unkeyed SHA-256, so 'Retained workspace records are authenticated' stays not_established until records are signed (for example Ed25519 over the canonical record bytes, with an RFC 3161 timestamp for created_at) and T077 and T080-T090 are re-run against the signed records to check that every surviving forgery is refused.
- T099 (partial): Bind --provider scr=<checkout> with cargo on PATH; command: CARGO_TARGET_DIR=<tmp> cargo build --release --locked --offline --manifest-path <scr>/crates/Cargo.toml -p execution-cli. The SP1 proved-heat build is recorded as blocked with its requirements and never attempted.
  - Unrun or unresolved: The binary digest depends on the Rust toolchain recorded in provider_runtime_identity; CI pins rustc 1.94.0 for the proved-heat gate, and other toolchains may produce different digests.
  - Unrun or unresolved: The SP1 proved-heat build needs crates.io and GitHub release downloads, the SP1 checkout, the Succinct compiler archive, protoc and >= 7 GiB RAM / 20 GiB disk (.github/workflows/proved-heat.yml); host probes are in sp1-requirements.json.
- T115 (partial): Integrate each geodesic with RK4 (N = 256) and adaptive DP5(4); count evaluations; count the common workload's operations; when a rapl-log capture from `python -m ciw.lab.energy_gpu_telemetry rapl-capture` is bound (a bracket of repeated geodesic batches and one of repeated common-workload batches …
  - Unrun or unresolved: Evaluation and operation counts are work proxies, not energy measurements
  - Unrun or unresolved: Package energy is not attributed to this process; idle subtraction assumes the idle interval after the workload represents the background during it
  - Unrun or unresolved: A capture is an operator record: identity binding is checked, authenticity is not
  - Unrun or unresolved: CPU energy per common-workload batch (here) and GPU energy per batch (T116) are for the same computation (the RAPL gate and T116's NVML gate both require the common workload's kernel, problem, prepared inputs, K and replicas) but different counters, scopes and batch boundaries: package versus whole device; the NumPy bracket excludes the prepared inputs (computed once before bracketing) and includes their per-batch broadcast to the replica columns, the GPU batch includes launch, synchronization, copy and the worker's output check; no finding combines them
  - Unrun or unresolved: no rapl-log capture was bound (--capture rapl-log=PATH or CIW_LAB_RAPL_LOG); the lab runner acquires no energy measurement: make one with `python -m ciw.lab.energy_gpu_telemetry rapl-capture` on a Linux host with readable intel-rapl counters (needs hardware:rapl)
- T116 (blocked): Blocked: unavailable requirement(s) hardware:nvidia-gpu. Planned: On the RTX 2080 host: (0) `mkdir -p runs/rtx2080-<date>`; (1) `ciw energy probe --gpu-index 0` must return a reading with status ok; (2) start `TZ=UTC nvidia-smi …
  - Unrun or unresolved: Blocked here: no NVIDIA GPU or NVML
  - Unrun or unresolved: NVML documents the total-energy counter for Volta-or-newer fully supported devices; GeForce support is not documented, so `ciw energy probe` must confirm it on this RTX 2080
  - Unrun or unresolved: An operator log is an unauthenticated record; the gate binds it to this host's NVML identity but cannot prove the capture genuine
- T117 (partial): Integrate the same initial states with the generic Python path, the closed-form Python path and the compiled Rust kernel; compare endpoints, counted evaluations and exact endpoints; send the Rust kernel a malformed and a nonfinite input. Run the common Gaussian VI workload with the NumPy reference …
  - Unrun or unresolved: No Julia implementation of the common Gaussian VI workload exists in the repository (T145 plans a Julia worker behind the SCR boundary; none exists), so that comparison cannot run on any host, whatever the tool:julia probe reports; the GPU comparison runs wherever an NVIDIA GPU answers the probe
  - Unrun or unresolved: Deferred research question: a Julia port of the common Gaussian VI workload with the kernel's operation order (no fused multiply-add), run through the Julia worker T145 plans behind the SCR boundary and compared bitwise with the NumPy reference
  - Unrun or unresolved: Deferred research question: extend the common workload's GPU path beyond the binary64 gaussian_vi PTX kernel of ciw.energy_cuda: a float32 rendering of the same kernel, and a device-wide reduction of its per-replica outputs (a fixed tree over the stored order per T148's REDUCTION_POLICY, plus one atomicAdd variant), each compared with the NumPy reference by compare_outputs on the RTX 2080 host and captured with `ciw energy record` for energy; until then those claims stay not_established on every host
  - Unrun or unresolved: Cross-platform bitwise identity of the sphere kernel is not claimed
  - Unrun or unresolved: The Rust and NumPy RAPL brackets share one boundary except process overhead: both exclude the prepared inputs (computed once before bracketing, telemetry.NUMPY_BOUNDARY and RUST_BOUNDARY); the Rust bracket includes one process start and one JSON exchange for all its batches, and the NumPy bracket vectorizes each batch across replicas; rust_over_numpy_idle_subtracted compares these two bracket contents, not the arithmetic alone
  - Unrun or unresolved: no NVIDIA GPU answered the hardware:nvidia-gpu probe in this task; the PTX kernel of the common workload exists (ciw.energy_cuda gaussian_vi), so this comparison runs on a host where the probe succeeds (needs hardware:nvidia-gpu)
  - Unrun or unresolved: no rapl-log capture was bound (--capture rapl-log=PATH or CIW_LAB_RAPL_LOG); the lab runner acquires no energy measurement: make one with `python -m ciw.lab.energy_gpu_telemetry rapl-capture` on a Linux host with readable intel-rapl counters (needs hardware:rapl)
- T118 (blocked): Blocked: unavailable requirement(s) hardware:nvidia-gpu. Planned: On the RTX 2080 host: (0) `mkdir -p runs/rtx2080-<date>`; (1) start `TZ=UTC nvidia-smi --query-gpu=timestamp,uuid,name,utilization.gpu,utilization.memory,temperature.gpu,power.draw,clocks.sm,clocks.mem,pstate --format=csv,nounits …
  - Unrun or unresolved: Blocked here: no NVIDIA GPU
  - Unrun or unresolved: Batch windows in log.json include launch, synchronization and copy; they are not kernel durations
  - Unrun or unresolved: The steady-state limits (CV 0.10, 5 C) are declared protocol criteria, not derived
  - Unrun or unresolved: Deferred research question: ingest `nsys stats --report cuda_gpu_kern_sum` output (its raw bytes retained as an artifact, bound through the acquisition gate to the same device UUID and capture session) so T118 can report RTX 2080 kernel-only duration; until then that claim stays not_established on every host
- T119 (partial): Analyze all four fixtures; recompute the baseline metric from raw readings and raw outputs; compare with the naive gross/executed ratio, with the per-distinct-result denominator and with a whole-run boundary. When CIW_LAB_ENERGY_LOG names an operator log, recompute E_acc from its raw readings and …
  - Unrun or unresolved: Idle subtraction and first-attainment accounting are intentionally not applied
  - Unrun or unresolved: The recomputation shares its origin (ciw) with energy_records, so it is a cross-implementation check, not independent verification
  - Unrun or unresolved: An operator log is an unauthenticated record; the gate binds it to this host's NVML identity but cannot prove the capture genuine
  - Unrun or unresolved: no operator NVML log was supplied (--capture energy-log=PATH or CIW_LAB_ENERGY_LOG); the repository fixtures are synthetic, so no physical energy per accepted result was measured
- T120 (partial): Vectorized RK4 in float32 and float64 across the step grid; smallest grid N per accuracy target; operation counts derived from the kernel source and checked by running one step on counting views. The common Gaussian VI workload in float32 and float64 for 0..256 iterations, KL at every iteration …
  - Unrun or unresolved: Operation counts do not capture memory traffic, vector width, transcendental cost or GPU float32 throughput; only a capture relates precision to energy
  - Unrun or unresolved: The NumPy reference runs each precision vectorized across replicas; CPU energy per batch reflects that implementation, not the GPU kernel
  - Unrun or unresolved: Both precision brackets share one boundary (telemetry.NUMPY_BOUNDARY): the prepared inputs are computed once before bracketing and excluded; each batch includes the rounding of the 15 inputs to its precision and their broadcast to the replica columns
  - Unrun or unresolved: implementation missing: no float32 implementation of the gaussian_vi PTX kernel exists (ciw.energy_cuda is binary64 only, and `ciw energy record` captures only it), so a float32 GPU workload cannot run on any host
  - Unrun or unresolved: Deferred research question: extend the common workload's GPU path beyond the binary64 gaussian_vi PTX kernel of ciw.energy_cuda: a float32 rendering of the same kernel, and a device-wide reduction of its per-replica outputs (a fixed tree over the stored order per T148's REDUCTION_POLICY, plus one atomicAdd variant), each compared with the NumPy reference by compare_outputs on the RTX 2080 host and captured with `ciw energy record` for energy; until then those claims stay not_established on every host
  - Unrun or unresolved: no rapl-log capture was bound (--capture rapl-log=PATH or CIW_LAB_RAPL_LOG); the lab runner acquires no energy measurement: make one with `python -m ciw.lab.energy_gpu_telemetry rapl-capture` on a Linux host with readable intel-rapl counters (needs hardware:rapl)
- T121 (partial): Emulate the orders in float32 and float64; search seeds for a sign flip among the sequential and two tree orders; test a pass/fail tolerance across atomic orders; test the guarded sign decision; evaluate a comparison-select maximum on signed zeros in both operand orders and retain numpy.max's …
  - Unrun or unresolved: Device-wide GPU reduction orders and warp-shuffle trees were not observed: no device reduction kernel exists; the kernel's in-kernel contraction is emulated here and observed only where the GPU runs
  - Unrun or unresolved: The pass/fail tolerance 0.01 was chosen inside the observed atomic spread; it shows that such flips exist, not how often they occur
  - Unrun or unresolved: The a priori guard is sound but so conservative that it decides none of the cancellation signs
  - Unrun or unresolved: T148's policy for fixed-order reductions: pairwise: fixed binary tree split at n//2 over the stored order (bitwise reproducible only for the same order and length); error <= gamma_{ceil(log2 n)} sum|x|
  - Unrun or unresolved: Deferred research question: extend the common workload's GPU path beyond the binary64 gaussian_vi PTX kernel of ciw.energy_cuda: a float32 rendering of the same kernel, and a device-wide reduction of its per-replica outputs (a fixed tree over the stored order per T148's REDUCTION_POLICY, plus one atomicAdd variant), each compared with the NumPy reference by compare_outputs on the RTX 2080 host and captured with `ciw energy record` for energy; until then those claims stay not_established on every host
  - Unrun or unresolved: no NVIDIA GPU answered the hardware:nvidia-gpu probe in this task; the PTX kernel of the common workload exists (ciw.energy_cuda gaussian_vi), so this comparison runs on a host where the probe succeeds (needs hardware:nvidia-gpu)
- T124 (partial): Retain the four fixtures' exact bytes as artifacts; validate them; apply 9 mutations (with or without resealing); validate a resealed doubling, an origin relabelling and a UUID mismatch. When an operator log is bound, retain its exact bytes, inventory its identity fields and bind it through T116's …
  - Unrun or unresolved: Sealing is integrity, not authenticity: no signature binds a log to the device
  - Unrun or unresolved: Hardware provenance of every fixture is not established
  - Unrun or unresolved: The fixtures' identity fields are present but mostly placeholders, including a compute capability given to the placeholder device
  - Unrun or unresolved: A real device's identity is bound only by the acquisition gate on the GPU host: the gate binds a log to that host's NVML identity, it does not authenticate the capture
  - Unrun or unresolved: The acquisition gate establishes an identity binding, not the origin of the readings: a resealed synthetic log whose identity strings were edited to name this host's device passes it (see the resealing and relabelling counterexamples), so a bound log names a device present here but its counter readings are not shown to come from that device
  - Unrun or unresolved: Deferred research question: a signed capture that binds a log's readings to the device (no signature scheme exists)
  - Unrun or unresolved: Not performed: retaining raw telemetry and device/runtime identity of a real device (no operator NVML log passed the acquisition gate in this run); only the synthetic fixtures were retained
  - Unrun or unresolved: no operator NVML log was bound (--capture energy-log=PATH or CIW_LAB_ENERGY_LOG)
- T138 (partial): Compute the prediction and its uncertainty components (open loop, conditioned on the start pose, and also on the as-built scan); evaluate a correct model against a tape realized 0.104 mm (2 sigma of the declared open-loop start error) off its nominal offset with and without the start-pose term, and …
  - Unrun or unresolved: The physical comparison has not been performed; its outcome is unknown.
  - Unrun or unresolved: The tapes are assumed to follow geodesics after their measured start; in-plane tape bending is not budgeted.
  - Unrun or unresolved: Operator captures are unauthenticated: their origin header and u column are declarations, and no metrology instrument probe exists on any analysing host. Deferred research question: a signed-capture trust anchor (an instrument-held key that signs each export, verified by the workbench) or a hardware:metrology probe of an instrument attached to the analysing host; until one exists, a bound capture yields computational comparisons only and the physical claims stay not_established.
  - Unrun or unresolved: The plate and cylinder cmm captures (tape start poses) are read and retained but do not yet condition those predictions.
- T139 (partial): Partial: the retention mechanism was exercised on a synthetic schema fixture (validate it, mutate it 11 ways, keep a rank-deficient covariance that an absolute threshold would refuse and refuse a negative one at the same scale, check identity invariance and the fixture/measurement boundary). No …
  - Unrun or unresolved: No acquisition was bound, so no raw measurement, calibration record or frame metadata has been retained: the task stays partial until a real acquisition of a control protocol is retained with its record. Synthetic or absent material never completes a retention task.
  - Unrun or unresolved: Media-type-specific readers (images, point clouds) are not defined; digests cover bytes, not content semantics.
  - Unrun or unresolved: The validators cannot tell whether raw bytes came from an instrument; the runner also requires a hardware probe in the task that cites them, and no metrology instrument probe or signed-capture trust anchor exists (deferred research question, T138), so a retained record's acquisition fields support no physical label.
- T145 (partial): Probe for julia; record the pin procedure; derive torus geometry with SymPy and compare with the core.
  - Unrun or unresolved: No Julia environment, manifest or worker exists; the pin procedure is unexecuted
  - Unrun or unresolved: The task has no Julia execution path; julia on PATH is recorded, never used
  - Unrun or unresolved: Optimization and exploratory roles are not demonstrated
- T147 (partial): Compute the dot products in four orders/precisions, compare with the harness under each policy, drop each partial product in turn from the float64 and float32 candidates and judge every faulty candidate with the harness, then compare detection with the operational guarantee, the threshold band and …
  - Unrun or unresolved: The batched dot products have no GPU path and need none: the GPU comparison uses the common Gaussian VI workload, whose PTX kernel exists; a device-wide reduction kernel does not
  - Unrun or unresolved: Deferred research question: extend the common workload's GPU path beyond the binary64 gaussian_vi PTX kernel of ciw.energy_cuda: a float32 rendering of the same kernel, and a device-wide reduction of its per-replica outputs (a fixed tree over the stored order per T148's REDUCTION_POLICY, plus one atomicAdd variant), each compared with the NumPy reference by compare_outputs on the RTX 2080 host and captured with `ciw energy record` for energy; until then those claims stay not_established on every host
  - Unrun or unresolved: GPU reductions may use FMA and tree shapes not modelled by the 32-lane order
  - Unrun or unresolved: Only single dropped products were injected; other fault classes (duplicated terms, wrong operands) have their own detection limits
  - Unrun or unresolved: no NVIDIA GPU answered the hardware:nvidia-gpu probe in this task; the PTX kernel of the common workload exists (ciw.energy_cuda gaussian_vi), so this comparison runs on a host where the probe succeeds (needs hardware:nvidia-gpu)
- T158 (partial): Hash and parse every retained SVG; re-execute the 41 declared inexpensive figure tasks plus every figure task that retains wall-clock timings in a scratch directory, and compare each regenerated figure's SHA-256 (and, for a difference, its bytes) with the retained one.
  - Unrun or unresolved: 41 figure tasks (77 figures) were not re-executed within the section's time budget: T001, T002, T003, T004, T005, T006, T007, T008, T009, T010, T011, T012, T014, T015, T016, T017, T018, T025, T036, T039, T040, T041, T043, T044, T046, T047, T051, T064, T067, T068, T097, T101, T107, T109, T126, T128, T133, T134, T135, T136, T137
  - Unrun or unresolved: Wall-clock timing figures are recognized by a retained JSON artifact of their task that mentions wall-clock or elapsed time; a timing figure without such a note counts as a mismatch.
  - Unrun or unresolved: Byte identity is established on one platform; Windows and other BLAS builds are not compared.
- T160 (partial): Generate the draft from retained reports of sections observation, sensor-fusion, manufacturing, energy-gpu; parse its finding tables back against the source findings and its label and boundary tables against their definitions; look up each required assumption in its section.
  - Unrun or unresolved: Missing from this draft, which the lab cannot write: the thesis in the authors' words (the generated thesis restates counts and counterexamples of the retained findings only), a related-work discussion (the reference list is the T156 textbook ledger's name matches), an interpretation of the counterexamples against the literature, conclusions, figure selection and captions, and external peer review. The draft stays partial until they exist.
  - Unrun or unresolved: For a task with independently_verified rows every assumption that mentions independence, same origin or ciw-written code is carried into the Qualifications, statistical independence of noise and events excepted (research_portfolio.STATISTICAL_INDEPENDENCE); for other tasks they are selected by phrase (research_portfolio.INDEPENDENCE_LIMITS), so one worded otherwise is not carried.
  - Unrun or unresolved: The selected finding of a task is its first established one, not a judgment of importance.
- T161 (partial): Generate the draft from retained reports of sections geodesic-jacobi, flat-torus-topology, surfaces-discrete; parse its finding tables back against the source findings and its label and boundary tables against their definitions; look up each required assumption in its section.
  - Unrun or unresolved: Missing from this draft, which the lab cannot write: the thesis in the authors' words (the generated thesis restates counts and counterexamples of the retained findings only), a related-work discussion (the reference list is the T156 textbook ledger's name matches), an interpretation of the counterexamples against the literature, conclusions, figure selection and captions, and external peer review. The draft stays partial until they exist.
  - Unrun or unresolved: For a task with independently_verified rows every assumption that mentions independence, same origin or ciw-written code is carried into the Qualifications, statistical independence of noise and events excepted (research_portfolio.STATISTICAL_INDEPENDENCE); for other tasks they are selected by phrase (research_portfolio.INDEPENDENCE_LIMITS), so one worded otherwise is not carried.
  - Unrun or unresolved: The selected finding of a task is its first established one, not a judgment of importance.
- T162 (partial): Generate the draft from retained reports of sections exchange-provenance, implementation-targets, lyapunov; parse its finding tables back against the source findings and its label and boundary tables against their definitions; look up each required assumption in its section.
  - Unrun or unresolved: Missing from this draft, which the lab cannot write: the thesis in the authors' words (the generated thesis restates counts and counterexamples of the retained findings only), a related-work discussion (the reference list is the T156 textbook ledger's name matches), an interpretation of the counterexamples against the literature, conclusions, figure selection and captions, and external peer review. The draft stays partial until they exist.
  - Unrun or unresolved: For a task with independently_verified rows every assumption that mentions independence, same origin or ciw-written code is carried into the Qualifications, statistical independence of noise and events excepted (research_portfolio.STATISTICAL_INDEPENDENCE); for other tasks they are selected by phrase (research_portfolio.INDEPENDENCE_LIMITS), so one worded otherwise is not carried.
  - Unrun or unresolved: The selected finding of a task is its first established one, not a judgment of importance.
  - Unrun or unresolved: Rules 6-9, 11 and 12 are stated from the specification; they are enforced by the validator, the report builder, the runner and the workspace classifier and tested in tests/test_lab_core.py and tests/test_lab_bridge.py, not backed by a T155 finding.

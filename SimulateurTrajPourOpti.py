# SimulateurTrajPourOpti.py  (fixed)
import gc

import numpy as np
import wgpu.backends.auto  # init a backend (Vulkan/DX12/Metal)
import wgpu

COEFFICIENT_FROTTEMENT = 0.25
G = 9.81
MAX_TARGETS = 5
DEFAULT_MAX_CANDIDATES = 200_000
WORKGROUP_SIZE = 128  # better occupancy for most GPUs

# WGSL: uses squared distances inside the loop (sqrt only at the end)
WGSL_SHADER = f"""
struct Consts {{
    dt: f32,
    g: f32,
    k: f32,
    position_net: f32,
    y0: f32,
    n_steps_f: f32,
    num_targets_f: f32,
    num_candidates_f: f32,
}};

@group(0) @binding(0) var<uniform> consts: Consts;
@group(0) @binding(1) var<storage, read> params: array<f32>;       // [v0, angle_deg] * N
@group(0) @binding(2) var<storage, read> target_px: array<f32>;    // len <= MAX_TARGETS
@group(0) @binding(3) var<storage, read> target_py: array<f32>;    // len <= MAX_TARGETS
@group(0) @binding(4) var<storage, read> target_w: array<f32>;     // len <= MAX_TARGETS
@group(0) @binding(5) var<storage, read_write> results: array<f32>; // len N

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {{
    let i = GlobalInvocationID.x;
    let n_steps = u32(consts.n_steps_f);
    let num_targets = u32(consts.num_targets_f);
    let num_candidates = u32(consts.num_candidates_f);
    if (i >= num_candidates) {{
        return;
    }}

    let v0 = params[2u * i + 0u];
    let ang_deg = params[2u * i + 1u];
    let rad = ang_deg * 0.017453292519943295; // pi/180

    // Preload target data into registers (avoid repeated storage reads)
    var l_tpx: array<f32, {MAX_TARGETS}>;
    var l_tpy: array<f32, {MAX_TARGETS}>;
    var l_tw:  array<f32, {MAX_TARGETS}>;
    for (var t: u32 = 0u; t < {MAX_TARGETS}u; t = t + 1u) {{
        if (t < num_targets) {{
            l_tpx[t] = target_px[t];
            l_tpy[t] = target_py[t];
            l_tw[t]  = target_w[t];
        }} else {{
            l_tpx[t] = 0.0;
            l_tpy[t] = 0.0;
            l_tw[t]  = 0.0;
        }}
    }}

    var vx = cos(rad) * v0;
    var vy = sin(rad) * v0;
    var px = consts.position_net;
    var py = consts.y0;

    var prev_px = px;
    var prev_py = py;

    var y_net = 0.0;
    var net_found = false;

    // track minimum squared distances for each target
    var min_d2: array<f32, {MAX_TARGETS}>;
    for (var t: u32 = 0u; t < {MAX_TARGETS}u; t = t + 1u) {{
        min_d2[t] = 3.402823e38;
    }}

    for (var s: u32 = 0u; s < n_steps; s = s + 1u) {{
        let speed = sqrt(vx * vx + vy * vy);
        vx = vx - consts.k * vx * speed * consts.dt;
        vy = vy - (consts.g + consts.k * vy * speed) * consts.dt;

        let nx = px + vx * consts.dt;
        let ny = py + vy * consts.dt;

        // min squared distances
        for (var t: u32 = 0u; t < num_targets; t = t + 1u) {{
            let dx = nx - l_tpx[t];
            let dy = ny - l_tpy[t];
            let d2 = dx*dx + dy*dy;
            if (d2 < min_d2[t]) {{
                min_d2[t] = d2;
            }}
        }}

        // net height (interpolate at x=0 crossing)
        if (!net_found && (prev_px <= 0.0) && (nx > 0.0)) {{
            y_net = prev_py + (-prev_px) * (ny - prev_py) / (nx - prev_px);
            net_found = true;
        }}

        prev_px = nx;
        prev_py = ny;
        px = nx;
        py = ny;

        if (ny < 0.0) {{
            break;
        }}
    }}

    var total = 0.0;
    for (var t: u32 = 0u; t < num_targets; t = t + 1u) {{
        total = total + l_tw[t] * sqrt(min_d2[t]);
    }}
    if (y_net < 1.55) {{
        total = total + (1.55 - y_net) * 1000.0;
    }}

    results[i] = total;
}}
"""


def pick_wgpu_device():
    # try common wgpu-py entry points; raise if none found
    try:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        if adapter is None:
            raise RuntimeError("Aucun GPU compatible WebGPU trouvé (Vulkan/DX12/Metal).")
        return adapter.request_device_sync()
    except Exception:
        # fallback to top-level helper
        adapter = wgpu.request_adapter(request_gpu=True)
        if adapter is None:
            raise RuntimeError("Aucun GPU compatible WebGPU trouvé (fallback).")
        return adapter.request_device()


class SimulateurTrajPourOptiGPU:
    """
    GPU evaluator (WebGPU). Reuses a single pipeline and buffers across calls.
    - Always allocates target buffers for MAX_TARGETS; we upload new targets in-place via update_targets().
    - Uniforms: dt, g, k, position_net, y0, n_steps, num_targets, num_candidates.
    - __call__ accepts (2,), (N,2) or (2,N) and returns float or ndarray(N,).
    """
    def __init__(self, points_cibles, poids, position_net, dt, t_max=8.0, device=None,
                 max_candidates=DEFAULT_MAX_CANDIDATES):

        if len(points_cibles) != len(poids):
            raise ValueError("points_cibles et poids doivent avoir la même longueur.")
        if len(points_cibles) > MAX_TARGETS:
            raise ValueError(f"Nombre de cibles {len(points_cibles)} > MAX_TARGETS={MAX_TARGETS}.")

        self.device = device or pick_wgpu_device()
        self.queue = self.device.queue
        self.max_N = int(max_candidates)
        self.dt = float(dt)
        self.n_steps = int(t_max / dt) + 1

        # Constantes physiques
        self.g = float(G)
        self.k = float(COEFFICIENT_FROTTEMENT)
        self.y0 = 1.55

        # Shader / pipeline
        self.shader = self.device.create_shader_module(code=WGSL_SHADER)
        self.bgl = self.device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.uniform, "has_dynamic_offset": False, "min_binding_size": 32}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage, "has_dynamic_offset": False, "min_binding_size": 0}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage, "has_dynamic_offset": False, "min_binding_size": 0}},
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage, "has_dynamic_offset": False, "min_binding_size": 0}},
            {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage, "has_dynamic_offset": False, "min_binding_size": 0}},
            {"binding": 5, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.storage, "has_dynamic_offset": False, "min_binding_size": 0}},
        ])
        self.pipeline = self.device.create_compute_pipeline(
            layout=self.device.create_pipeline_layout(bind_group_layouts=[self.bgl]),
            compute={"module": self.shader, "entry_point": "main"},
        )

        # Buffers dynamiques pré-alloués (params/results)
        params_bytes = 2 * self.max_N * 4  # float32 bytes
        results_bytes = self.max_N * 4

        self._params_size = params_bytes
        self._results_size = results_bytes

        self.buf_params = self.device.create_buffer(
            size=self._params_size,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self.buf_results = self.device.create_buffer(
            size=self._results_size,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
        )

        # Target buffers: fixed capacity = MAX_TARGETS (rewritten via update_targets)
        cap_bytes = MAX_TARGETS * 4
        self._targets_bytes = cap_bytes
        self.buf_tpx = self.device.create_buffer(size=cap_bytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
        self.buf_tpy = self.device.create_buffer(size=cap_bytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
        self.buf_tw = self.device.create_buffer(size=cap_bytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)

        # Uniforms (updated partially per-call)
        self.num_targets = min(len(points_cibles), MAX_TARGETS)
        self.position_net = float(position_net)

        self._uniform_vec = np.array([
            self.dt, self.g, self.k, self.position_net,
            self.y0, float(self.n_steps), float(self.num_targets), 0.0
        ], dtype=np.float32)
        uniform_bytes = self._uniform_vec.tobytes()
        self._uniform_size = len(uniform_bytes)

        # create uniform buffer with initial data
        self.buf_uniform = self.device.create_buffer_with_data(
            data=uniform_bytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        # Bind group (created once). Use explicit sizes for each entry so driver is happy.
        self.bind_group = self.device.create_bind_group(
            layout=self.bgl,
            entries=[
                {"binding": 0, "resource": {"buffer": self.buf_uniform, "offset": 0, "size": self._uniform_size}},
                {"binding": 1, "resource": {"buffer": self.buf_params, "offset": 0, "size": self._params_size}},
                {"binding": 2, "resource": {"buffer": self.buf_tpx, "offset": 0, "size": self._targets_bytes}},
                {"binding": 3, "resource": {"buffer": self.buf_tpy, "offset": 0, "size": self._targets_bytes}},
                {"binding": 4, "resource": {"buffer": self.buf_tw,  "offset": 0, "size": self._targets_bytes}},
                {"binding": 5, "resource": {"buffer": self.buf_results, "offset": 0, "size": self._results_size}},
            ],
        )

        # Initial upload of targets
        self.update_targets(points_cibles, poids, position_net)

    def _write_uniform_f32(self, index, value):
        # Update one float in the uniform buffer at given index
        self._uniform_vec[index] = float(value)
        # write_buffer expects (buffer, offset, data)
        self.queue.write_buffer(self.buf_uniform, index * 4, np.array([self._uniform_vec[index]], dtype=np.float32).tobytes())

    def update_targets(self, points_cibles, poids, position_net):
        """
        Update target arrays and dependent uniforms (num_targets, position_net).
        """
        if len(points_cibles) != len(poids):
            raise ValueError("points_cibles et poids doivent avoir la même longueur.")
        if len(points_cibles) > MAX_TARGETS:
            raise ValueError(f"Nombre de cibles {len(points_cibles)} > MAX_TARGETS={MAX_TARGETS}.")

        num_t = len(points_cibles)
        tpx = np.zeros(MAX_TARGETS, dtype=np.float32)
        tpy = np.zeros(MAX_TARGETS, dtype=np.float32)
        tw = np.zeros(MAX_TARGETS, dtype=np.float32)
        for idx, (x, y) in enumerate(points_cibles):
            tpx[idx] = float(x)
            tpy[idx] = float(y)
            tw[idx] = float(poids[idx])

        # Upload the full arrays (small: MAX_TARGETS<=5)
        self.queue.write_buffer(self.buf_tpx, 0, tpx.tobytes())
        self.queue.write_buffer(self.buf_tpy, 0, tpy.tobytes())
        self.queue.write_buffer(self.buf_tw, 0, tw.tobytes())

        self.num_targets = num_t
        self.position_net = float(position_net)
        self._write_uniform_f32(3, self.position_net)       # position_net (index 3)
        self._write_uniform_f32(6, float(self.num_targets)) # num_targets (index 6)

    def __call__(self, X):
        """
        X: shape (2,) or (N, 2) or (2, N). Returns float or ndarray (N,).
        Each candidate: [v0, angle_vertical_deg].
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        elif X.ndim == 2 and X.shape[0] == 2 and X.shape[1] != 2:
            X = X.T

        if X.shape[1] != 2:
            raise ValueError("Each candidate must be [v0, angle_vertical_deg].")

        N = X.shape[0]
        if N > self.max_N:
            raise ValueError(f"N={N} > max_candidates={self.max_N}.")

        # Upload params (pad the params buffer size if necessary)
        params_bytes = X.tobytes()
        self.queue.write_buffer(self.buf_params, 0, params_bytes)

        # Update num_candidates in uniform (float slot index 7)
        self._write_uniform_f32(7, float(N))

        # Dispatch
        encoder = self.device.create_command_encoder()
        pass_enc = encoder.begin_compute_pass()
        pass_enc.set_pipeline(self.pipeline)
        # Most wgpu-py set_bind_group expects (index, bind_group)
        pass_enc.set_bind_group(0, self.bind_group)
        group_x = (N + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE
        pass_enc.dispatch_workgroups(group_x, 1, 1)
        pass_enc.end()
        self.queue.submit([encoder.finish()])

        # Read back first N results (synchronous read)
        # Ensure we read at least N*4 bytes
        out_bytes = self.queue.read_buffer(self.buf_results, 0, N * 4)
        res = np.frombuffer(out_bytes, dtype=np.float32).copy()  # copy to own memory
        return res if N > 1 else float(res[0])

    def destroy(self):
        """Détruit explicitement tous les objets GPU pour libérer la VRAM."""
        print("Destruction des ressources wgpu...")
        self.buf_uniform.destroy()
        self.buf_params.destroy()
        self.buf_results.destroy()
        self.buf_tpx.destroy()
        self.buf_tpy.destroy()
        self.buf_tw.destroy()


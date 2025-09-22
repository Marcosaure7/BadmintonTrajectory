# simulateur_traj_gpu.py
import math
import numpy as np
import wgpu.backends.auto  # active un backend WebGPU (Vulkan/DX12/Metal)
import wgpu

COEFFICIENT_FROTTEMENT = 0.25
G = 9.81
MAX_TARGETS = 64  # augmente si tu as plus de points cibles

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
@group(0) @binding(1) var<storage, read> params: array<f32>;      // [v0, angle_deg] * N
@group(0) @binding(2) var<storage, read> target_px: array<f32>;   // len M
@group(0) @binding(3) var<storage, read> target_py: array<f32>;   // len M
@group(0) @binding(4) var<storage, read> target_w: array<f32>;    // len M
@group(0) @binding(5) var<storage, read_write> results: array<f32>; // len N

@compute @workgroup_size(64)
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

    var vx = cos(rad) * v0;
    var vy = sin(rad) * v0;
    var px = consts.position_net;
    var py = consts.y0;

    var prev_px = px;
    var prev_py = py;

    var y_net = 0.0;
    var net_found = false;

    var min_dists: array<f32, {MAX_TARGETS}>;
    for (var t: u32 = 0u; t < {MAX_TARGETS}u; t = t + 1u) {{
        min_dists[t] = 1.0e20;
    }}

    for (var s: u32 = 0u; s < n_steps; s = s + 1u) {{
        let speed = sqrt(vx * vx + vy * vy);
        vx = vx - consts.k * vx * speed * consts.dt;
        vy = vy - (consts.g + consts.k * vy * speed) * consts.dt;

        let nx = px + vx * consts.dt;
        let ny = py + vy * consts.dt;

        // distances aux cibles (réduction min)
        for (var t: u32 = 0u; t < num_targets; t = t + 1u) {{
            let dx = nx - target_px[t];
            let dy = ny - target_py[t];
            let d = sqrt(dx*dx + dy*dy);
            if (d < min_dists[t]) {{
                min_dists[t] = d;
            }}
        }}

        // hauteur au filet (interpolation au passage x=0)
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
        total = total + target_w[t] * min_dists[t];
    }}
    if (y_net < 1.55) {{
        total = total + (1.55 - y_net) * 1000.0;
    }}

    results[i] = total;
}}
"""

def pick_wgpu_device():
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
        raise RuntimeError("Aucun GPU compatible WebGPU trouvé (Vulkan/DX12/Metal).")
    return adapter.request_device_sync()

class SimulateurTrajPourOptiGPU:
    """
    Évalue l'objectif (distance pondérée + pénalité filet) sur GPU via WebGPU.
    - Supporte lot (shape (N,2)) ou scalaire (shape (2,)).
    - dt et t_max configurables.
    """
    def __init__(self, points_cibles, poids, position_net, dt, t_max=8.0, device=None):
        if len(points_cibles) != len(poids):
            raise ValueError("points_cibles et poids doivent avoir la même longueur.")
        if len(points_cibles) > MAX_TARGETS:
            raise ValueError(f"Nombre de cibles {len(points_cibles)} > MAX_TARGETS={MAX_TARGETS}.")

        self.device = device or pick_wgpu_device()
        self.queue = self.device.queue

        self.dt = float(dt)
        self.n_steps = int(t_max / dt) + 1

        # Constantes
        self.g = float(G)
        self.k = float(COEFFICIENT_FROTTEMENT)
        self.position_net = float(position_net)
        self.y0 = 1.55
        self.num_targets = len(points_cibles)

        # Buffers cibles
        tpx = np.asarray([float(px) for (px, _) in points_cibles], dtype=np.float32)
        tpy = np.asarray([float(py) for (_, py) in points_cibles], dtype=np.float32)
        tw = np.asarray([float(w) for w in poids], dtype=np.float32)

        self.buf_tpx = self.device.create_buffer_with_data(
            data=tpx.tobytes(), usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        )
        self.buf_tpy = self.device.create_buffer_with_data(
            data=tpy.tobytes(), usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        )
        self.buf_tw = self.device.create_buffer_with_data(
            data=tw.tobytes(), usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        )

        # Shader / pipeline
        self.shader = self.device.create_shader_module(code=WGSL_SHADER)
        self.bgl = self.device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.uniform, "has_dynamic_offset": False, "min_binding_size": 0}},
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

        # Uniforms
        self._uniform_vec = np.array([
            self.dt, self.g, self.k, self.position_net,
            self.y0, float(self.n_steps), float(self.num_targets), 0.0  # dernier champ = num_candidates (MAJ à l'appel)
        ], dtype=np.float32)
        self.buf_uniform = self.device.create_buffer_with_data(
            data=self._uniform_vec.tobytes(),
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        # Buffers dynamiques (taillés à la volée)
        self.buf_params = None
        self.buf_results = None

    def _ensure_buffers(self, N):
        params_bytes = int(2 * N * 4)
        results_bytes = int(N * 4)
        if (self.buf_params is None) or (self.buf_params.size < params_bytes):
            self.buf_params = self.device.create_buffer(
                size=params_bytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
            )
        if (self.buf_results is None) or (self.buf_results.size < results_bytes):
            self.buf_results = self.device.create_buffer(
                size=results_bytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
            )

    def _bind_group(self):
        return self.device.create_bind_group(
            layout=self.bgl,
            entries=[
                {"binding": 0, "resource": {"buffer": self.buf_uniform, "offset": 0, "size": self.buf_uniform.size}},
                {"binding": 1, "resource": {"buffer": self.buf_params, "offset": 0, "size": self.buf_params.size}},
                {"binding": 2, "resource": {"buffer": self.buf_tpx, "offset": 0, "size": self.buf_tpx.size}},
                {"binding": 3, "resource": {"buffer": self.buf_tpy, "offset": 0, "size": self.buf_tpy.size}},
                {"binding": 4, "resource": {"buffer": self.buf_tw, "offset": 0, "size": self.buf_tw.size}},
                {"binding": 5, "resource": {"buffer": self.buf_results, "offset": 0, "size": self.buf_results.size}},
            ],
        )

    def __call__(self, X):
        """
        X: shape (2,) or (N, 2) or (2, N) for vectorized calls. Returns float or ndarray (N,).
        Each candidate is [v0, angle_vertical_deg].
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]  # Convert (2,) to (1, 2)
        elif X.ndim == 2 and X.shape[0] == 2 and X.shape[1] != 2:
            X = X.T  # Convert (2, N) to (N, 2) for vectorized calls

        if X.shape[1] != 2:
            raise ValueError("Each candidate must be [v0, angle_vertical_deg].")

        N = X.shape[0]
        self._ensure_buffers(N)

        # Params to GPU
        self.queue.write_buffer(self.buf_params, 0, X.tobytes())

        # Uniform: Update number of candidates
        self._uniform_vec[-1] = float(N)
        self.queue.write_buffer(self.buf_uniform, 0, self._uniform_vec.tobytes())

        # Dispatch
        encoder = self.device.create_command_encoder()
        pass_enc = encoder.begin_compute_pass()
        pass_enc.set_pipeline(self.pipeline)
        pass_enc.set_bind_group(0, self._bind_group(), [], 0, 0xFFFFFFFF)
        group_x = (N + 63) // 64
        pass_enc.dispatch_workgroups(group_x, 1, 1)
        pass_enc.end()
        self.queue.submit([encoder.finish()])

        # Read results
        out_bytes = self.queue.read_buffer(self.buf_results, 0, N * 4)
        res = np.frombuffer(out_bytes, dtype=np.float32).copy()
        return res if X.shape[0] > 1 else float(res[0])
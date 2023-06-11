import torch
import pypose as pp
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
import warnings
from nn_utile import AUVRNNDeltaV, AUVTraj
import time
import torch.onnx
import onnxruntime


from onnx import load_model, save_model
from onnxruntime import InferenceSession

from network_utils import load_onnx_model, create_onnx_bound_model

from utile import to_euler
from utile import plot_traj
import matplotlib.pyplot as plt

gpu_ok = False
if torch.cuda.is_available():
    device_cap = torch.cuda.get_device_capability()
    if device_cap in ((7, 0), (8, 0), (9, 0)):
        gpu_ok = True

if not gpu_ok:
    warnings.warn(
        "GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower "
        "than expected."
    )

def get_device(gpu=False, unit=0):
    use_cuda = False
    if gpu:
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Asked for GPU but torch couldn't find a Cuda capable device")
    return torch.device(f"cuda:{unit}" if use_cuda else "cpu")
    # return torch.device("mps")

def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    restul = fn()
    end.record()
    torch.cuda.synchronize()
    return restul, start.elapsed_time(end) / 1000

def gen_data(b, device):
    state = torch.zeros(size=(b, 1, 13), device=device)
    state[..., 6] = 1.
    seq = torch.zeros(size=(b, 50, 6), device=device)
    return state, seq

def evaluate(mod, state, seq):
    return mod(state, seq)

evaluate_opt = torch.compile(evaluate)

N_ITERS = 10
BATCH_SIZE = 2000

device = get_device(True)
state, seq = gen_data(BATCH_SIZE, device)

# Warm-up
model = AUVTraj().to(device)
model.load_state_dict(torch.load("model.pth"))

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

onnx_model_filename = "./model.onnx"

#####################  Normal ONNX run #######################
ort_session = load_onnx_model(onnx_model_filename, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# bind_session, io_binding = create_onnx_bound_model(onnx_model_filename, state, seq, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(state), ort_session.get_inputs()[1].name: to_numpy(seq)}

pred_trajs, pred_vels, pred_dvs = evaluate(model, state, seq)

pred2_trajs, pred2_vels, pred2_dvs = ort_session.run(None, ort_inputs)



# pred_trajs, pred_vels, pred_dvs = evaluate(model, state, seq)
# pred2_trajs, pred2_vels, pred2_dvs = evaluate_opt(model, state, seq)

# #print("pred_traj", pred_trajs.shape)
# error_traj += loss(pred_trajs, traj)
# error_vel += loss(pred_vels, vel)
# error_dv += loss(pred_dvs, dv)

pred_trajs = pred_trajs.detach().cpu()
pred_vels = pred_vels.detach().cpu()
pred_dvs = pred_dvs.detach().cpu()

print(pred_trajs.shape)
print(pred2_trajs.shape)

# pred2_trajs = pred2_trajs.detach().cpu()
# pred2_vels = pred2_vels.detach().cpu()
# pred2_dvs = pred2_dvs.detach().cpu()
tau = 10
# traj_euler = trajs_plot[0]

pred_traj_euler = to_euler(pred_trajs[0].data)
pred_traj_euler2 = to_euler(pred2_trajs[0])
s_col = {"x": 0, "y": 1, "z": 2, "roll": 3, "pitch": 4, "yaw": 5}
plot_traj({"pred": pred_traj_euler, "pred2": pred_traj_euler2}, s_col, tau, True, title="State")
plt.show()









# pred2_trajs, pred2_vels, pred2_dvs = evaluate_opt(model, state, seq)


# print("eager:", timed(lambda: evaluate(model, state, seq))[1])
# print("compile:", timed(lambda: evaluate_opt(model, state, seq))[1])
# print("onnx:", timed(lambda: ort_session.run(None, ort_inputs))[1])
# print("onnx bind:", timed(lambda: bind_session.run_with_iobinding(io_binding))[1])


# eager_times = []
# compile_times = []
# onnx_times = []
# onnx_bind_times = []
# for i in range(N_ITERS):
#     _, eager_time = timed(lambda: evaluate(model, state, seq))
#     eager_times.append(eager_time)
#     print(f"eager eval time {i}: {eager_time}")

# print("~" * 10)

# compile_times = []
# for i in range(N_ITERS):
#     _, compile_time = timed(lambda: evaluate_opt(model, state, seq))
#     compile_times.append(compile_time)
#     print(f"compile eval time {i}: {compile_time}")
# print("~" * 10)

# onnx_times = []
# for i in range(N_ITERS):
#     _, onnx_time = timed(lambda:ort_session.run(None, ort_inputs))
#     onnx_times.append(onnx_time)
#     print(f"onnx eval time {i}: {onnx_time}")
# print("~" * 10)


# onnx_bind_times = []
# for i in range(N_ITERS):
#     _, onnx_time = timed(lambda:bind_session.run_with_iobinding(io_binding))
#     onnx_bind_times.append(onnx_time)
#     print(f"onnx eval time {i}: {onnx_time}")
# print("~" * 10)


# eager_med = np.median(eager_times)
# compile_med = np.median(compile_times)
# onnx_med = np.median(onnx_times)
# onnx_bind_med = np.median(onnx_bind_times)
# speedup = eager_med / compile_med
# speedup_onnx = eager_med / onnx_med
# speedup_onnx_bind = eager_med / onnx_bind_med
# print(f"(eval) eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup}x")
# print(f"(eval) eager median: {eager_med}, onnx median: {onnx_med}, speedup: {speedup_onnx}x")
# print(f"(eval) eager median: {eager_med}, onnx bind median: {onnx_bind_med}, speedup: {speedup_onnx_bind}x")
# print("~" * 10)

# prof_file = ort_session.end_profiling()

# print(prof_file)

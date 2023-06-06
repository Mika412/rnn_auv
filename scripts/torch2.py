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

# evaluate_opt = torch.compile(evaluate)

N_ITERS = 10
BATCH_SIZE = 2000

device_id = 4

device = get_device(True, device_id)
state, seq = gen_data(BATCH_SIZE, device)

# Warm-up
model = AUVTraj().to(device)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

onnx_model_filename = "./model_sim.onnx"

#####################  Normal ONNX run #######################
opts = onnxruntime.SessionOptions()

# opts.intra_op_num_threads = 2
# opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
# opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
# opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
# opts.enable_profiling = True
# opts.log_severity_level = 0
# opts.log_verbosity_level= 0

ort_session = onnxruntime.InferenceSession(onnx_model_filename, opts, providers=[('CUDAExecutionProvider', {'device_id': device_id}), 'CPUExecutionProvider'])
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(state), ort_session.get_inputs()[1].name: to_numpy(seq)}


#####################  Normal ONNX Bind #######################
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
bind_session = onnxruntime.InferenceSession(onnx_model_filename, opts, providers=[('CUDAExecutionProvider', {'device_id': device_id}), 'CPUExecutionProvider'])


io_binding = bind_session.io_binding()

io_binding.bind_input(
    name=ort_session.get_inputs()[0].name,
    device_type='cuda',
    device_id=device_id,
    element_type=np.float32,
    buffer_ptr=state.data_ptr(),
    shape=state.shape)

io_binding.bind_input(
    name=ort_session.get_inputs()[1].name,
    device_type='cuda',
    device_id=device_id,
    element_type=np.float32,
    buffer_ptr=seq.data_ptr(),
    shape=seq.shape)

io_binding.bind_output(
    name=ort_session.get_outputs()[0].name,
    device_type='cuda',
    device_id=device_id)

print("eager:", timed(lambda: evaluate(model, state, seq))[1])
# print("compile:", timed(lambda: evaluate_opt(model, state, seq))[1])
print("onnx:", timed(lambda: ort_session.run(None, ort_inputs))[1])
print("onnx bind:", timed(lambda: ort_session.run(None, ort_inputs))[1])


eager_times = []
compile_times = []
onnx_times = []
onnx_bind_times = []
for i in range(N_ITERS):
    _, eager_time = timed(lambda: evaluate(model, state, seq))
    eager_times.append(eager_time)
    print(f"eager eval time {i}: {eager_time}")

print("~" * 10)

# compile_times = []
# for i in range(N_ITERS):
#     _, compile_time = timed(lambda: evaluate_opt(model, state, seq))
#     compile_times.append(compile_time)
#     print(f"compile eval time {i}: {compile_time}")
# print("~" * 10)

onnx_times = []
for i in range(N_ITERS):
    _, onnx_time = timed(lambda:ort_session.run(None, ort_inputs))
    onnx_times.append(onnx_time)
    print(f"onnx eval time {i}: {onnx_time}")
print("~" * 10)


onnx_bind_times = []
for i in range(N_ITERS):
    _, onnx_time = timed(lambda:bind_session.run_with_iobinding(io_binding))
    onnx_bind_times.append(onnx_time)
    print(f"onnx eval time {i}: {onnx_time}")
print("~" * 10)


eager_med = np.median(eager_times)
# compile_med = np.median(compile_times)
onnx_med = np.median(onnx_times)
onnx_bind_med = np.median(onnx_bind_times)
# speedup = eager_med / compile_med
speedup_onnx = eager_med / onnx_med
speedup_onnx_bind = eager_med / onnx_bind_med
# print(f"(eval) eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup}x")
print(f"(eval) eager median: {eager_med}, onnx median: {onnx_med}, speedup: {speedup_onnx}x")
print(f"(eval) eager median: {eager_med}, onnx bind median: {onnx_bind_med}, speedup: {speedup_onnx_bind}x")
print("~" * 10)

prof_file = ort_session.end_profiling()

print(prof_file)

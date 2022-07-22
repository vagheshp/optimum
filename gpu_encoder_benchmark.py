from distutils.version import LooseVersion
from typing import Optional

import pandas as pd
import torch
from torch._C import _from_dlpack
from torch.utils.dlpack import to_dlpack
from transformers import AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

from onnxruntime.capi import _pybind_state as C  # needs onnxruntime-training
from onnxruntime.capi.onnxruntime_inference_collection import OrtValue
from optimum.onnxruntime import ORTModelForSequenceClassification

from time import perf_counter

import numpy as np


def _ortvalue_from_torch_tensor(torch_tensor):
    return C.OrtValue.from_dlpack(to_dlpack(torch_tensor), False)


def _ortvalues_to_torch_tensor(ortvalues):
    res = ortvalues.to_dlpacks(_from_dlpack)
    return res

_ortvalue_from_torch_tensor(torch.ones(1, 1))


class IOBindingModel(ORTModelForSequenceClassification):
    def __init__(self, model=None, config=None, **kwargs):
        super().__init__(model, config, **kwargs)
        # create {name:idx} dict for model outputs
        self.model_outputs = {output_key.name: idx for idx, output_key in enumerate(self.model.get_outputs())}
        self.model_inputs = {output_key.name: idx for idx, output_key in enumerate(self.model.get_inputs())}
        self.model_input_names = list(self.model_inputs.keys())
        self.model_output_names = list(self.model_outputs.keys())
        self.run_options = C.RunOptions()
        # leads to Segmentation fault (core dumped)
        # self.io_binding = self.model.io_binding()


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        self.io_binding = self.model.io_binding()

        # add input io binding
        self.io_binding.bind_ortvalue_input("input_ids", OrtValue(_ortvalue_from_torch_tensor(input_ids)))
        self.io_binding.bind_ortvalue_input("attention_mask", OrtValue(_ortvalue_from_torch_tensor(attention_mask)))
        if token_type_ids is not None:
            self.io_binding.bind_ortvalue_input(
                "token_type_ids", OrtValue(_ortvalue_from_torch_tensor(token_type_ids))
            )

        # add output io binding
        for name in self.model_output_names:
            self.io_binding.bind_output(name, self.device.type, device_id=self.device.index)

        # run inference with binding
        self.model.run_with_iobinding(self.io_binding, self.run_options)
        # Copy output contents to CPU (if on another device). No-op if already on the CPU.
        # Y = io_binding.copy_outputs_to_cpu()[0]
        raw_outputs = self.io_binding._iobinding.get_outputs()
        outputs = _ortvalues_to_torch_tensor(raw_outputs)

        # clear
        # self.io_binding.clear_binding_inputs()
        # self.io_binding.clear_binding_outputs()
        return SequenceClassifierOutput(logits=outputs[0])



def benchmark(seq_len, model, tokenizer, device, iterations=200):
    # prepare date
    seq_len = "l " * (seq_len - 2)
    payload = tokenizer(seq_len, return_tensors="pt")
    payload = {key: val.to(device) for key, val in payload.items()}
    latencies = []
    # warm up
    for _ in range(10):
        _ = model(**payload)
    # Timed run
    for _ in range(iterations):
        start_time = perf_counter()
        _ = model(**payload)
        latency = perf_counter() - start_time
        latencies.append(latency)
    # Compute run statistics
    time_avg_ms = 1000 * np.mean(latencies)
    time_p95_ms = 1000 * np.percentile(latencies, 95)
    return {"seq_len": payload["input_ids"].shape[1], "time_avg_ms": time_avg_ms, "time_p95_ms": time_p95_ms}


device = torch.device("cuda:0")
io_model = IOBindingModel.from_pretrained("optimum/distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = AutoTokenizer.from_pretrained("optimum/distilbert-base-uncased-finetuned-sst-2-english")

io_model.to(device)

payload = "I hate you"
d = tokenizer(payload, return_tensors="pt")
d["input_ids"]= d["input_ids"].to(device)
d["attention_mask"]= d["attention_mask"].to(device)
print(io_model(**d))


model = ORTModelForSequenceClassification.from_pretrained("optimum/distilbert-base-uncased-finetuned-sst-2-english")
model.to(device)
print(model(**d))


seq_lengths = [8, 16, 32, 64, 128, 256, 512]
res = []
for seq_len in seq_lengths:
    print("seq_len: ", seq_len)
    io = benchmark(seq_len, io_model, tokenizer, device, iterations=500)
    res.append({**io, "model": "io"})

    vanilla = benchmark(seq_len, model, tokenizer, device, iterations=500)
    res.append({**vanilla, "model": "vanilla"})


df = pd.DataFrame(res)

chart_df = pd.merge(
    df[df.model == "io"][["seq_len", "time_p95_ms"]],
    df[df.model == "vanilla"][["seq_len", "time_p95_ms"]],
    on="seq_len",
)

chart_df = chart_df.rename(columns={"time_p95_ms_x": "io_p95", "time_p95_ms_y": "vanilla_p95"})
chart_df['io_improvement'] = f"{round((chart_df['vanilla_p95'] - chart_df['io_p95']) / chart_df['vanilla_p95'] * 100,2)}%"

plt = chart_df.plot(x="seq_len", y=["io_p95", "vanilla_p95"], kind="line")
plt.figure.savefig("gpu_encoder_res.png", dpi=900)

print(chart_df.head(10))
chart_df.to_csv("gpu_encoder_res.csv")

from transformers import BarkModel, AutoProcessor, set_seed
import torch
from optimum.bettertransformer import BetterTransformer
from torch.cuda import Event
from torch.cuda import max_memory_allocated
from torch.cuda import empty_cache
from torch.cuda import reset_peak_memory_stats
import scipy
import os
import numpy as np
from scipy.io import wavfile


def output_to_file(name, data, sampling_rate):
    # If the data is float16, convert it to float32
    if data.dtype == np.float16:
        data = data.astype(np.float32)

    # Ensure the data is scaled between [-1, 1] for float types
    if data.dtype in [np.float32, np.float64]:
        data = data / np.max(np.abs(data))

    # Write the data to a WAV file
    wavfile.write(name, sampling_rate, data)


# Load the model and its processor
model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)
processor = AutoProcessor.from_pretrained("suno/bark-small", torch_dtype=torch.float16)

print("Model has been loaded...")

while True:

    text = input("\n\n> ")

    inputs = processor(text=text, voice_preset="v2/en_speaker_6").to(device)

    # Utility function to measure latency and memory usage
    def measure_latency_and_memory_use(model, inputs, nb_loops, eos_token_id):
        start_event = Event(enable_timing=True)
        end_event = Event(enable_timing=True)
        reset_peak_memory_stats(device)
        empty_cache()
        torch.cuda.synchronize()
        start_event.record()
        for _ in range(nb_loops):
            set_seed(0)
            # high quality
            # output = model.generate(**inputs, fine_temperature=0.4, coarse_temperature=0.8, semantic_temperature = 0.8)
            # less quality
            # output = model.generate(**inputs, fine_temperature=0.4, coarse_temperature=0.8, do_sample=True)
            output = model.generate(
                **inputs,
                fine_temperature=0.4,
                coarse_temperature=0.8,
                semantic_temperature=0.8
            )
        end_event.record()
        torch.cuda.synchronize()
        max_memory = max_memory_allocated(device)
        elapsed_time = start_event.elapsed_time(end_event) * 1.0e-3
        print("Execution time:", elapsed_time / nb_loops, "seconds")
        print("Max memory footprint:", max_memory * 1e-9, "GB")
        return output

    # Listen to the baseline output
    sampling_rate = model.generation_config.sample_rate
    eos_token_id = model.config.eos_token_id

    # Baseline measurement
    with torch.inference_mode():
        speech_output = measure_latency_and_memory_use(model, inputs, 1, eos_token_id)

    output_to_file("file1.wav", speech_output[0].cpu().numpy(), sampling_rate)

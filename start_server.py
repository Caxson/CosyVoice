#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess, sys

cmd = [
    sys.executable, "tts_server.py",
    "--model_dir", "pretrained_models/Fun-CosyVoice3-0.5B",
    "--port", "6006",
    "--prompt_wav", "zhaoxin5.wav",
    "--prompt_text", "You are a helpful assistant.<|endofprompt|>唉，戴姐，唉，那个我哥五百元的会员奖励确实没发呢",
    "--spk_id", "zhaoxin",
    "--load_trt",
    "--load_vllm",
]

with open("/tmp/tts_server.log", "w") as log:
    subprocess.Popen(cmd, stdout=log, stderr=log)

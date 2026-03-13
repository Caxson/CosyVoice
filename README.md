# CosyVoice - 语音合成系统

基于 FunAudioLLM/CosyVoice 的优化版本，提供流式 TTS HTTP 服务。

## 🚀 最近优化与修复

本仓库基于官方 CosyVoice 进行了以下优化和bug修复：

### ✨ 新功能
- **一键启动服务脚本** (`start_server.py`, `tts_server.py`)
  - 流式 TTS HTTP 服务，默认端口 6006
  - 支持 zero_shot / cross_lingual / instruct2 / sft 模式
  - 支持 PCM 流式输出和 MP3 非流式输出
  - 音色克隆持久化存储，支持音色管理（克隆/列表/删除）
  - 相同参考音频自动缓存，避免重复提取特征
  - 实时打印首帧时延和逐帧时延统计
  - 支持 TensorRT 和 vLLM 加速

### ⚡ 性能优化
- **流式首包延迟优化**：显著降低流式推理的首帧响应延迟
- **状态管理修复**：修复 `token_hop_len` 状态变异问题，避免多次推理时的状态污染
- **浮点精度改进**：使用 float64 避免精度误差，弃用 CPU 计算以提升性能
- **添加依赖支持**：增加 pydub 和 lameenc，支持更好的音频编码

### 🐛 Bug修复
- **训练稳定性修复**：解决 CausalMaskedDiffWithDiT 训练中的张量形状不匹配和索引越界问题
- **vLLM兼容性**：修复 vllm yaml 版本兼容性问题
- **RAS修复**：修复 Repetition Aware Sampling 相关问题

## 📦 快速开始

### 环境安装

```bash
# 克隆仓库
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
git submodule update --init --recursive

# 创建 Conda 环境
conda create -n cosyvoice -y python=3.10
conda activate cosyvoice
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# 如果遇到 sox 兼容性问题
# Ubuntu
sudo apt-get install sox libsox-dev
# CentOS
sudo yum install sox sox-devel
```

### 模型下载

```python
# 使用 modelscope SDK 下载模型（国内推荐）
from modelscope import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
snapshot_download('iic/CosyVoice-300M', local_dir='pretrained_models/CosyVoice-300M')
snapshot_download('iic/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT')
snapshot_download('iic/CosyVoice-300M-Instruct', local_dir='pretrained_models/CosyVoice-300M-Instruct')
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')

# 海外用户使用 HuggingFace
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
# ... 其他模型类似
```

## 🎯 一键启动TTS服务

### 方式一：使用启动脚本（推荐）

编辑 `start_server.py` 配置您的参数，然后运行：

```bash
python3 start_server.py
```

脚本会在后台启动服务，日志输出到 `/tmp/tts_server.log`

### 方式二：直接启动

```bash
python3 tts_server.py \
  --model_dir pretrained_models/Fun-CosyVoice3-0.5B \
  --port 6006 \
  --prompt_wav your_reference.wav \
  --prompt_text "You are a helpful assistant.<|endofprompt|>这里是参考音频的文本内容" \
  --spk_id your_voice_id \
  --load_trt \
  --load_vllm
```

**参数说明：**
- `--model_dir`: 模型路径（本地路径或 modelscope repo id）
- `--port`: 服务端口（默认 6006）
- `--host`: 监听地址（默认 0.0.0.0）
- `--fp16`: 启用 FP16 推理（可选）
- `--load_trt`: 启用 TensorRT 加速 flow decoder（推荐）
- `--load_vllm`: 启用 vLLM 加速 LLM 推理（推荐）
- `--prompt_wav`: 默认参考音频路径（配置后启用 /tts 简化接口）
- `--prompt_text`: 默认参考音频对应的文本
- `--spk_id`: 默认音色 ID

### API 接口

服务启动后，可通过以下接口调用：

#### 1. `/tts` - 简化TTS接口（需配置默认音色）
```bash
curl -X POST "http://localhost:6006/tts" \
  -F "tts_text=你好，这是一段测试文本" \
  -F "format=mp3" \
  --output output.mp3
```

#### 2. `/inference_zero_shot` - 零样本克隆
```bash
curl -X POST "http://localhost:6006/inference_zero_shot" \
  -F "tts_text=你好，这是一段测试文本" \
  -F "prompt_text=参考音频的文本" \
  -F "prompt_wav=@reference.wav" \
  -F "format=pcm" \
  --output output.pcm
```

#### 3. `/voice/clone` - 音色克隆（持久化）
```bash
curl -X POST "http://localhost:6006/voice/clone" \
  -F "prompt_wav=@reference.wav" \
  -F "prompt_text=参考音频的文本" \
  -F "voice_id=my_voice"
```

#### 4. `/voice/list` - 列出所有克隆音色
```bash
curl "http://localhost:6006/voice/list"
```

#### 5. `/voice/delete` - 删除音色
```bash
curl -X POST "http://localhost:6006/voice/delete" \
  -H "Content-Type: application/json" \
  -d '{"voice_id":"my_voice"}'
```

#### 6. 其他接口
- `/inference_cross_lingual` - 跨语言合成
- `/inference_instruct2` - 指令合成（CosyVoice2/3）
- `/inference_sft` - 内置音色合成
- `/health` - 健康检查
- `/spk_cache` - 查看缓存状态

**格式说明：**
- `format=pcm`：流式输出（低延迟，实时打印帧时延）
- `format=mp3`：非流式输出（完整编码后返回，打印合成/编码/总耗时）

## 💡 基本使用

### Python 示例
```bash
python example.py
```

### vLLM 加速（推荐）
CosyVoice2/3 支持 **vLLM 0.11.x+ (V1 engine)** 和 **vLLM 0.9.0 (legacy)**

```bash
conda create -n cosyvoice_vllm --clone cosyvoice
conda activate cosyvoice_vllm
# 安装 vLLM
pip install vllm==v0.11.0 transformers==4.57.1 numpy==1.26.4 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
python vllm_example.py
```

### Web UI
```bash
# 修改模型路径以使用不同的模型
python3 webui.py --port 50000 --model_dir pretrained_models/CosyVoice-300M
```

## 🔧 高级用法

### TensorRT-LLM 加速部署
使用 TensorRT-LLM 可获得 4x 加速：

```bash
cd runtime/triton_trtllm
docker compose up -d
```

详情参考：[runtime/triton_trtllm](https://github.com/FunAudioLLM/CosyVoice/tree/main/runtime/triton_trtllm)

### Docker 部署
```bash
cd runtime/python
docker build -t cosyvoice:v1.0 .

# gRPC 方式
docker run -d --runtime=nvidia -p 50000:50000 cosyvoice:v1.0 \
  /bin/bash -c "cd /opt/CosyVoice/CosyVoice/runtime/python/grpc && \
  python3 server.py --port 50000 --max_conc 4 --model_dir iic/CosyVoice-300M && sleep infinity"

# FastAPI 方式
docker run -d --runtime=nvidia -p 50000:50000 cosyvoice:v1.0 \
  /bin/bash -c "cd /opt/CosyVoice/CosyVoice/runtime/python/fastapi && \
  python3 server.py --port 50000 --model_dir iic/CosyVoice-300M && sleep infinity"
```

## 📊 模型性能

**Fun-CosyVoice 3.0** 是基于大语言模型（LLM）的先进文本转语音（TTS）系统，在内容一致性、说话人相似度和韵律自然度方面超越前代产品（CosyVoice 2.0），专为野外多语言零样本语音合成而设计。

### 核心特性
- **语言覆盖**：支持 9 种常用语言（中文、英语、日语、韩语、德语、西班牙语、法语、意大利语、俄语），18+ 种中文方言/口音
- **内容一致性与自然度**：在内容一致性、说话人相似度和韵律自然度方面达到最先进水平
- **发音修复**：支持中文拼音和英语 CMU 音素的发音修复，提供更多可控性，适合生产使用
- **文本归一化**：支持数字、特殊符号和各种文本格式的朗读，无需传统前端模块
- **双向流式**：支持文本输入流和音频输出流，延迟低至 150ms 同时保持高质量音频输出
- **指令支持**：支持语言、方言、情感、速度、音量等多种指令

详细评测数据请参考 [官方文档](https://github.com/FunAudioLLM/CosyVoice)。

## 📚 相关链接

- **Fun-CosyVoice 3.0**: [Demos](https://funaudiollm.github.io/cosyvoice3/) | [Paper](https://arxiv.org/pdf/2505.17589) | [Modelscope](https://www.modelscope.cn/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) | [Huggingface](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
- **CosyVoice 2.0**: [Demos](https://funaudiollm.github.io/cosyvoice2/) | [Paper](https://arxiv.org/pdf/2412.10117) | [Modelscope](https://www.modelscope.cn/models/iic/CosyVoice2-0.5B)
- **CosyVoice 1.0**: [Demos](https://fun-audio-llm.github.io) | [Paper](https://funaudiollm.github.io/pdf/CosyVoice_v1.pdf) | [Modelscope](https://www.modelscope.cn/models/iic/CosyVoice-300M)

## 💬 讨论与交流

- [Github Issues](https://github.com/FunAudioLLM/CosyVoice/issues)
- 钉钉官方交流群（见官方仓库）

## 📄 致谢

本项目借鉴了以下开源项目的代码：
1. [FunASR](https://github.com/modelscope/FunASR)
2. [FunCodec](https://github.com/modelscope/FunCodec)
3. [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS)
4. [AcademiCodec](https://github.com/yangdongchao/AcademiCodec)
5. [WeNet](https://github.com/wenet-e2e/wenet)

## 📝 引用

```bibtex
@article{du2024cosyvoice,
  title={Cosyvoice: A scalable multilingual zero-shot text-to-speech synthesizer based on supervised semantic tokens},
  author={Du, Zhihao and Chen, Qian and Zhang, Shiliang and Hu, Kai and Lu, Heng and Yang, Yexin and Hu, Hangrui and Zheng, Siqi and Gu, Yue and Ma, Ziyang and others},
  journal={arXiv preprint arXiv:2407.05407},
  year={2024}
}

@article{du2024cosyvoice2,
  title={Cosyvoice 2: Scalable streaming speech synthesis with large language models},
  author={Du, Zhihao and Wang, Yuxuan and Chen, Qian and Shi, Xian and Lv, Xiang and Zhao, Tianyu and Gao, Zhifu and Yang, Yexin and Gao, Changfeng and Wang, Hui and others},
  journal={arXiv preprint arXiv:2412.10117},
  year={2024}
}

@article{du2025cosyvoice3,
  title={CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training},
  author={Du, Zhihao and Gao, Changfeng and Wang, Yuxuan and Yu, Fan and Zhao, Tianyu and Wang, Hao and Lv, Xiang and Wang, Hui and Shi, Xian and An, Keyu and others},
  journal={arXiv preprint arXiv:2505.17589},
  year={2025}
}
```

## ⚠️ 免责声明

以上内容仅供学术目的，旨在展示技术能力。部分示例来自互联网。如有任何内容侵犯您的权利，请联系我们删除。

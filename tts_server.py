#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CosyVoice 流式 TTS HTTP 服务
- 端口: 6006 (可通过 --port 修改)
- 支持模式: zero_shot / cross_lingual / instruct2 / sft
- 输出格式: format=pcm 流式（默认）/ format=mp3 非流式（收齐后编码）
- 实时打印首帧时延与逐帧时延；MP3 时打印合成耗时、编码耗时、总耗时（首帧等价）
- 相同参考音频自动缓存，避免重复提取特征
"""
import os
import sys
import time
import json
import hashlib
import logging
import argparse
import tempfile
import threading
import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'third_party/Matcha-TTS'))

from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from cosyvoice.cli.cosyvoice import AutoModel, CosyVoice3


def _register_vllm_model():
    from vllm import ModelRegistry
    from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
    ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

# ── 日志配置 ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('tts_server')
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('pydub').setLevel(logging.WARNING)        # 屏蔽 pydub ffmpeg 调试输出

# ── FastAPI 初始化 ─────────────────────────────────────────────────────────────
app = FastAPI(title='CosyVoice Streaming TTS', version='1.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# ── 全局变量 ──────────────────────────────────────────────────────────────────
cosyvoice_model = None

# 启动时预注册的默认音色（--prompt_wav 指定时生效）
default_spk_id = None

# 说话人特征缓存：{md5(audio+prompt_text): spk_id}
# 相同参考音频+prompt文本只提取一次特征，后续直接复用
_spk_cache: dict = {}
_spk_cache_lock = threading.Lock()

# 克隆音色持久化目录（启动时初始化）
_voice_store_dir: str = ''
_voice_meta_path: str = ''
_voice_meta: dict = {}          # {voice_id: {prompt_text, created_at, ...}}
_voice_meta_lock = threading.Lock()


# ── CosyVoice3 prompt_text 修正 ───────────────────────────────────────────────
def _prepare_prompt_text(prompt_text: str) -> str:
    """
    CosyVoice3 的 LLM 要求 prompt_text 中必须包含 <|endofprompt|> 分隔符（token 151646）。
    若用户传入的是纯转录文本，则自动补充标准前缀。
    CosyVoice / CosyVoice2 无此要求，不作处理。
    """
    if isinstance(cosyvoice_model, CosyVoice3) and '<|endofprompt|>' not in prompt_text:
        return 'You are a helpful assistant.<|endofprompt|>' + prompt_text
    return prompt_text


# ── 音色持久化工具 ────────────────────────────────────────────────────────────
def _init_voice_store(base_dir: str):
    """初始化音色存储目录，加载已有音色到模型中"""
    global _voice_store_dir, _voice_meta_path, _voice_meta
    _voice_store_dir = os.path.join(base_dir, 'voice_store')
    os.makedirs(_voice_store_dir, exist_ok=True)
    _voice_meta_path = os.path.join(_voice_store_dir, 'meta.json')

    if os.path.exists(_voice_meta_path):
        with open(_voice_meta_path, 'r', encoding='utf-8') as f:
            _voice_meta = json.load(f)
    else:
        _voice_meta = {}

    loaded = 0
    repaired = 0
    for voice_id, info in list(_voice_meta.items()):
        pt_path = os.path.join(_voice_store_dir, f'{voice_id}.pt')
        if not os.path.exists(pt_path):
            logger.warning(f'[voice_store] 音色 {voice_id} 的 .pt 文件缺失，跳过')
            continue
        spk_data = torch.load(pt_path, map_location='cpu', weights_only=False)

        if isinstance(cosyvoice_model, CosyVoice3) and 'prompt_text' in spk_data:
            if 151646 not in spk_data['prompt_text']:
                raw_text = info.get('prompt_text', '')
                fixed_text = _prepare_prompt_text(raw_text)
                tk = cosyvoice_model.frontend.tokenizer
                tokens = tk.encode(fixed_text, allowed_special=cosyvoice_model.frontend.allowed_special)
                spk_data['prompt_text'] = torch.tensor([tokens], dtype=torch.int32)
                spk_data['prompt_text_len'] = torch.tensor([len(tokens)], dtype=torch.int32)
                torch.save(spk_data, pt_path)
                repaired += 1
                logger.info(f'[voice_store] 已修复音色 {voice_id} 的 prompt_text（补充 <|endofprompt|>）')

        cosyvoice_model.frontend.spk2info[voice_id] = spk_data
        loaded += 1
        logger.info(f'[voice_store] 已加载音色: {voice_id}')

    logger.info(f'[voice_store] 共加载 {loaded} 个克隆音色，修复 {repaired} 个（存储目录: {_voice_store_dir}）')


def _save_voice(voice_id: str, prompt_text: str):
    """将当前模型中的音色特征持久化到磁盘"""
    spk_data = cosyvoice_model.frontend.spk2info.get(voice_id)
    if spk_data is None:
        raise ValueError(f'音色 {voice_id} 不在模型中')

    pt_path = os.path.join(_voice_store_dir, f'{voice_id}.pt')
    torch.save(spk_data, pt_path)

    with _voice_meta_lock:
        _voice_meta[voice_id] = {
            'voice_id': voice_id,
            'prompt_text': prompt_text,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(_voice_meta_path, 'w', encoding='utf-8') as f:
            json.dump(_voice_meta, f, ensure_ascii=False, indent=2)

    logger.info(f'[voice_store] 音色已保存: {voice_id} → {pt_path}')


def _delete_voice(voice_id: str):
    """从磁盘和模型中删除音色"""
    pt_path = os.path.join(_voice_store_dir, f'{voice_id}.pt')
    if os.path.exists(pt_path):
        os.unlink(pt_path)

    with _voice_meta_lock:
        _voice_meta.pop(voice_id, None)
        with open(_voice_meta_path, 'w', encoding='utf-8') as f:
            json.dump(_voice_meta, f, ensure_ascii=False, indent=2)

    cosyvoice_model.frontend.spk2info.pop(voice_id, None)
    logger.info(f'[voice_store] 音色已删除: {voice_id}')


# ── 说话人缓存工具 ────────────────────────────────────────────────────────────
def _get_cache_key(audio_bytes: bytes, prompt_text: str) -> str:
    return hashlib.md5(audio_bytes + prompt_text.encode('utf-8')).hexdigest()


def _get_or_register_spk(audio_bytes: bytes, prompt_text: str, tmp_path: str) -> str:
    """
    查找或注册说话人特征缓存，返回 spk_id。
    - 命中缓存：直接返回，跳过音频预处理（节省 1~2s）
    - 未命中：调用 add_zero_shot_spk 提取特征并写入缓存
    """
    cache_key = _get_cache_key(audio_bytes, prompt_text)

    with _spk_cache_lock:
        if cache_key in _spk_cache:
            spk_id = _spk_cache[cache_key]
            logger.info(f'[spk_cache] 命中缓存 → spk_id={spk_id}，跳过音频预处理')
            return spk_id

    # 首次注册：提取特征（耗时操作，在锁外执行避免阻塞其他缓存查询）
    spk_id = f'spk_{cache_key[:12]}'
    t0 = time.perf_counter()
    cosyvoice_model.add_zero_shot_spk(_prepare_prompt_text(prompt_text), tmp_path, spk_id)
    cost_ms = (time.perf_counter() - t0) * 1000
    logger.info(f'[spk_cache] 新音色注册完成 → spk_id={spk_id}，特征提取耗时 {cost_ms:.0f} ms')

    with _spk_cache_lock:
        _spk_cache[cache_key] = spk_id

    return spk_id


# ── PCM 流生成器 ──────────────────────────────────────────────────────────────
def _pcm_generator(model_output, sample_rate: int, request_time: float,
                   mode: str, tmp_path: str = None):
    """
    包装模型输出，转换为 PCM int16 字节流，并在服务端打印时延统计。
    首帧时延  = 请求到达 → 第一帧 yield 的耗时
    后续帧时延 = 上一帧 yield → 当前帧 yield 的耗时
    tmp_path  = 流结束后需要清理的临时文件（仅首次注册时存在）
    """
    frame_idx = 0
    prev_time = request_time
    try:
        for output in model_output:
            tts_speech = output['tts_speech']          # shape: [1, N]
            audio_bytes = (tts_speech.numpy() * 32768).astype(np.int16).tobytes()

            now = time.perf_counter()
            interval_ms = (now - prev_time) * 1000
            speech_len_ms = tts_speech.shape[1] / sample_rate * 1000

            if frame_idx == 0:
                total_ms = (now - request_time) * 1000
                logger.info(
                    f'[{mode}] 首帧到达 | '
                    f'请求→首帧: {total_ms:>7.1f} ms | '
                    f'帧时长: {speech_len_ms:>6.0f} ms | '
                    f'RTF: {interval_ms / speech_len_ms:.3f}'
                )
            else:
                logger.info(
                    f'[{mode}] 第 {frame_idx + 1:>2d} 帧   | '
                    f'帧间隔:      {interval_ms:>7.1f} ms | '
                    f'帧时长: {speech_len_ms:>6.0f} ms | '
                    f'RTF: {interval_ms / speech_len_ms:.3f}'
                )

            frame_idx += 1
            prev_time = time.perf_counter()
            yield audio_bytes

    finally:
        total_elapsed_ms = (time.perf_counter() - request_time) * 1000
        logger.info(f'[{mode}] 合成完成 | 共 {frame_idx} 帧 | 总耗时: {total_elapsed_ms:.1f} ms')
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── 流式 MP3 生成器：每帧 PCM 立即编码成 MP3 chunk 再 yield，首帧延迟 ≈ 流式 PCM ─────────────
def _mp3_stream_generator(model_output, sample_rate: int, request_time: float,
                           mode: str, tmp_path: str = None, bitrate: int = 128):
    """
    逐帧将 PCM 编码为 MP3 并流式 yield，首帧延迟与流式 PCM 相当（只多几毫秒编码）。
    使用 lameenc 做增量编码，无需等全部帧合成完。
    """
    try:
        import lameenc
    except ImportError:
        raise HTTPException(status_code=500, detail='流式 MP3 需要安装 lameenc: pip install lameenc')

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(1)
    encoder.set_quality(7)          # 2=best 7=fast，优先速度

    frame_idx = 0
    prev_time = request_time
    try:
        for output in model_output:
            tts_speech = output['tts_speech']
            pcm_int16 = (tts_speech.numpy() * 32768).astype(np.int16)

            t_encode_start = time.perf_counter()
            mp3_chunk = encoder.encode(pcm_int16.tobytes())
            encode_ms = (time.perf_counter() - t_encode_start) * 1000

            now = time.perf_counter()
            interval_ms = (now - prev_time) * 1000
            speech_len_ms = tts_speech.shape[1] / sample_rate * 1000

            if frame_idx == 0:
                total_ms = (now - request_time) * 1000
                logger.info(
                    f'[{mode}/mp3s] 首帧到达 | '
                    f'请求→首帧: {total_ms:>7.1f} ms | '
                    f'编码: {encode_ms:.1f} ms | '
                    f'帧时长: {speech_len_ms:>6.0f} ms'
                )
            else:
                logger.info(
                    f'[{mode}/mp3s] 第 {frame_idx + 1:>2d} 帧   | '
                    f'帧间隔: {interval_ms:>7.1f} ms | '
                    f'编码: {encode_ms:.1f} ms'
                )

            frame_idx += 1
            prev_time = time.perf_counter()
            if mp3_chunk:
                yield bytes(mp3_chunk)

        tail = encoder.flush()
        if tail:
            yield bytes(tail)

    finally:
        total_elapsed_ms = (time.perf_counter() - request_time) * 1000
        logger.info(f'[{mode}/mp3s] 合成完成 | 共 {frame_idx} 帧 | 总耗时: {total_elapsed_ms:.1f} ms')
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── 非流式 MP3：收齐 PCM 后编码，并打点时延（合成 / 编码 / 总 = 首帧等价时间）────────────────
def _collect_pcm_and_encode_mp3(model_output, sample_rate: int, request_time: float,
                                 mode: str, tmp_path: str = None, mp3_bitrate: str = "128k"):
    """
    收集模型输出的全部 PCM，编码为 MP3 并返回 (mp3_bytes, headers)。
    日志：合成耗时、编码耗时、总耗时（即非流式下的“首帧”响应时间），以及与流式首帧的对比提示。
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail='服务端输出 MP3 需要安装 pydub，且系统需安装 ffmpeg。请: pip install pydub'
        )

    chunks = []
    first_chunk_time = None
    frame_count = 0

    for output in model_output:
        tts_speech = output['tts_speech']
        audio_bytes = (tts_speech.numpy() * 32768).astype(np.int16).tobytes()
        if first_chunk_time is None:
            first_chunk_time = time.perf_counter()
        chunks.append(audio_bytes)
        frame_count += 1

    synthesis_done_time = time.perf_counter()
    synthesis_ms = (synthesis_done_time - request_time) * 1000
    first_chunk_ms = (first_chunk_time - request_time) * 1000 if first_chunk_time else 0

    pcm_data = b''.join(chunks)
    if not pcm_data:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail='合成无音频输出')

    audio = AudioSegment(
        data=pcm_data,
        sample_width=2,
        frame_rate=sample_rate,
        channels=1,
    )
    buf = __import__('io').BytesIO()
    audio.export(buf, format='mp3', bitrate=mp3_bitrate)
    mp3_bytes = buf.getvalue()

    encode_done_time = time.perf_counter()
    encode_ms = (encode_done_time - synthesis_done_time) * 1000
    total_ms = (encode_done_time - request_time) * 1000

    logger.info(
        f'[{mode}] 非流式 MP3 | '
        f'合成耗时: {synthesis_ms:>7.1f} ms | '
        f'编码耗时: {encode_ms:>7.1f} ms | '
        f'总耗时(首帧等价): {total_ms:>7.1f} ms | '
        f'若流式PCM首帧约: {first_chunk_ms:.0f} ms'
    )

    if tmp_path and os.path.exists(tmp_path):
        os.unlink(tmp_path)

    headers = {
        'X-Sample-Rate': str(sample_rate),
        'X-Channels': '1',
        'X-Format': 'mp3',
    }
    return mp3_bytes, headers


# ── 接口：TTS 合成（支持可选 voice_id，不传则用默认音色）────────────────────
@app.post('/tts')
async def tts(
    tts_text: str = Form(..., description='待合成文本'),
    voice_id: str = Form('', description='音色 ID，留空使用默认音色'),
    stream: bool = Form(True, description='是否流式输出'),
    speed: float = Form(1.0, description='语速倍率'),
    format: str = Form('pcm', description='输出格式: pcm 流式 | mp3 非流式'),
):
    vid = voice_id.strip() if voice_id else ''
    spk_id = vid if vid else default_spk_id

    if not spk_id:
        raise HTTPException(
            status_code=400,
            detail='服务未配置默认音色且未指定 voice_id，请启动时传入 --prompt_wav 或先调用 /voice/clone'
        )

    if spk_id not in cosyvoice_model.frontend.spk2info:
        raise HTTPException(status_code=404, detail=f'音色 {spk_id} 不存在，请先通过 /voice/clone 创建')

    request_time = time.perf_counter()
    out_fmt = (format or 'pcm').strip().lower()
    logger.info(f'[tts] 收到请求 | spk={spk_id} | 文本: {tts_text} | format={out_fmt}')

    use_mp3 = out_fmt == 'mp3'

    model_output = cosyvoice_model.inference_zero_shot(
        tts_text, '', '',
        zero_shot_spk_id=spk_id,
        stream=True, speed=speed,
    )
    if use_mp3:
        return StreamingResponse(
            _mp3_stream_generator(model_output, cosyvoice_model.sample_rate, request_time, 'tts'),
            media_type='audio/mpeg',
            headers={'X-Sample-Rate': str(cosyvoice_model.sample_rate), 'X-Channels': '1'},
        )
    return StreamingResponse(
        _pcm_generator(model_output, cosyvoice_model.sample_rate,
                       request_time, 'tts'),
        media_type='audio/pcm',
        headers={
            'X-Sample-Rate': str(cosyvoice_model.sample_rate),
            'X-Channels': '1',
            'X-Bit-Depth': '16',
        }
    )


# ── 接口：零样本克隆 ──────────────────────────────────────────────────────────
@app.post('/inference_zero_shot')
async def inference_zero_shot(
    tts_text: str = Form(..., description='待合成文本'),
    prompt_text: str = Form(..., description='prompt 参考文本'),
    prompt_wav: UploadFile = File(..., description='prompt 参考音频（WAV）'),
    stream: bool = Form(True, description='是否流式输出'),
    speed: float = Form(1.0, description='语速倍率'),
    format: str = Form('pcm', description='输出格式: pcm 流式 | mp3 非流式'),
):
    request_time = time.perf_counter()
    out_fmt = (format or 'pcm').strip().lower()
    logger.info(f'[zero_shot] 收到请求 | 文本: {tts_text} | format={out_fmt}')

    audio_bytes = await prompt_wav.read()
    cache_key = _get_cache_key(audio_bytes, prompt_text)

    with _spk_cache_lock:
        cached_spk_id = _spk_cache.get(cache_key)

    if cached_spk_id:
        tmp_path = None
        spk_id = cached_spk_id
        logger.info(f'[spk_cache] 命中缓存 → spk_id={spk_id}，跳过音频预处理')
    else:
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp.write(audio_bytes)
        tmp.close()
        tmp_path = tmp.name
        spk_id = _get_or_register_spk(audio_bytes, prompt_text, tmp_path)

    use_mp3 = out_fmt == 'mp3'
    model_output = cosyvoice_model.inference_zero_shot(
        tts_text, '', '',
        zero_shot_spk_id=spk_id,
        stream=True, speed=speed,
    )
    if use_mp3:
        return StreamingResponse(
            _mp3_stream_generator(model_output, cosyvoice_model.sample_rate, request_time, 'zero_shot', tmp_path),
            media_type='audio/mpeg',
            headers={'X-Sample-Rate': str(cosyvoice_model.sample_rate), 'X-Channels': '1'},
        )
    return StreamingResponse(
        _pcm_generator(model_output, cosyvoice_model.sample_rate,
                       request_time, 'zero_shot', tmp_path),
        media_type='audio/pcm',
        headers={
            'X-Sample-Rate': str(cosyvoice_model.sample_rate),
            'X-Channels': '1',
            'X-Bit-Depth': '16',
        }
    )


# ── 接口：跨语言合成 ──────────────────────────────────────────────────────────
@app.post('/inference_cross_lingual')
async def inference_cross_lingual(
    tts_text: str = Form(..., description='待合成文本'),
    prompt_wav: UploadFile = File(..., description='prompt 参考音频（WAV）'),
    stream: bool = Form(True, description='是否流式输出'),
    speed: float = Form(1.0, description='语速倍率'),
    format: str = Form('pcm', description='输出格式: pcm 流式 | mp3 非流式'),
):
    request_time = time.perf_counter()
    out_fmt = (format or 'pcm').strip().lower()
    logger.info(f'[cross_lingual] 收到请求 | 文本: {tts_text} | format={out_fmt}')

    audio_bytes = await prompt_wav.read()
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.write(audio_bytes)
    tmp.close()
    tmp_path = tmp.name

    use_mp3 = out_fmt == 'mp3'
    model_output = cosyvoice_model.inference_cross_lingual(
        tts_text, tmp_path, stream=True, speed=speed,
    )
    if use_mp3:
        return StreamingResponse(
            _mp3_stream_generator(model_output, cosyvoice_model.sample_rate, request_time, 'cross_lingual', tmp_path),
            media_type='audio/mpeg',
            headers={'X-Sample-Rate': str(cosyvoice_model.sample_rate), 'X-Channels': '1'},
        )
    return StreamingResponse(
        _pcm_generator(model_output, cosyvoice_model.sample_rate,
                       request_time, 'cross_lingual', tmp_path),
        media_type='audio/pcm',
        headers={
            'X-Sample-Rate': str(cosyvoice_model.sample_rate),
            'X-Channels': '1',
            'X-Bit-Depth': '16',
        }
    )


# ── 接口：instruct2（CosyVoice2/3 专属）──────────────────────────────────────
@app.post('/inference_instruct2')
async def inference_instruct2(
    tts_text: str = Form(..., description='待合成文本'),
    instruct_text: str = Form(..., description='指令文本，末尾加 <|endofprompt|>'),
    prompt_wav: UploadFile = File(..., description='prompt 参考音频（WAV）'),
    stream: bool = Form(True, description='是否流式输出'),
    speed: float = Form(1.0, description='语速倍率'),
    format: str = Form('pcm', description='输出格式: pcm 流式 | mp3 非流式'),
):
    if not hasattr(cosyvoice_model, 'inference_instruct2'):
        raise HTTPException(status_code=400, detail='当前模型不支持 inference_instruct2')

    request_time = time.perf_counter()
    out_fmt = (format or 'pcm').strip().lower()
    logger.info(f'[instruct2] 收到请求 | 文本: {tts_text} | format={out_fmt}')

    audio_bytes = await prompt_wav.read()
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.write(audio_bytes)
    tmp.close()
    tmp_path = tmp.name

    use_mp3 = out_fmt == 'mp3'
    model_output = cosyvoice_model.inference_instruct2(
        tts_text, instruct_text, tmp_path, stream=True, speed=speed,
    )
    if use_mp3:
        return StreamingResponse(
            _mp3_stream_generator(model_output, cosyvoice_model.sample_rate, request_time, 'instruct2', tmp_path),
            media_type='audio/mpeg',
            headers={'X-Sample-Rate': str(cosyvoice_model.sample_rate), 'X-Channels': '1'},
        )
    return StreamingResponse(
        _pcm_generator(model_output, cosyvoice_model.sample_rate,
                       request_time, 'instruct2', tmp_path),
        media_type='audio/pcm',
        headers={
            'X-Sample-Rate': str(cosyvoice_model.sample_rate),
            'X-Channels': '1',
            'X-Bit-Depth': '16',
        }
    )


# ── 接口：SFT（内置音色）──────────────────────────────────────────────────────
@app.post('/inference_sft')
async def inference_sft(
    tts_text: str = Form(..., description='待合成文本'),
    spk_id: str = Form(..., description='说话人 ID'),
    stream: bool = Form(True, description='是否流式输出'),
    speed: float = Form(1.0, description='语速倍率'),
    format: str = Form('pcm', description='输出格式: pcm 流式 | mp3 非流式'),
):
    request_time = time.perf_counter()
    out_fmt = (format or 'pcm').strip().lower()
    logger.info(f'[sft] 收到请求 | spk_id={spk_id} | 文本: {tts_text} | format={out_fmt}')

    use_mp3 = out_fmt == 'mp3'
    model_output = cosyvoice_model.inference_sft(
        tts_text, spk_id, stream=True, speed=speed,
    )
    if use_mp3:
        return StreamingResponse(
            _mp3_stream_generator(model_output, cosyvoice_model.sample_rate, request_time, 'sft'),
            media_type='audio/mpeg',
            headers={'X-Sample-Rate': str(cosyvoice_model.sample_rate), 'X-Channels': '1'},
        )
    return StreamingResponse(
        _pcm_generator(model_output, cosyvoice_model.sample_rate,
                       request_time, 'sft'),
        media_type='audio/pcm',
        headers={
            'X-Sample-Rate': str(cosyvoice_model.sample_rate),
            'X-Channels': '1',
            'X-Bit-Depth': '16',
        }
    )


# ── 接口：克隆音色 ────────────────────────────────────────────────────────────
@app.post('/voice/clone')
async def voice_clone(
    prompt_wav: UploadFile = File(..., description='参考音频（WAV 格式）'),
    prompt_text: str = Form(..., description='参考音频对应的文本内容'),
    voice_id: str = Form('', description='自定义音色 ID，留空自动生成'),
):
    """
    上传参考音频 → 提取音色特征 → 持久化存储。
    后续可在 /tts 中通过 voice_id 使用该音色。
    """
    request_time = time.perf_counter()
    audio_bytes = await prompt_wav.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail='上传的音频文件为空')

    vid = voice_id.strip()
    if not vid:
        vid = f'voice_{hashlib.md5(audio_bytes).hexdigest()[:10]}'

    if vid == default_spk_id:
        raise HTTPException(status_code=400, detail=f'voice_id 不能与默认音色 ID ({default_spk_id}) 相同')

    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.write(audio_bytes)
    tmp.close()
    tmp_path = tmp.name

    try:
        t0 = time.perf_counter()
        cosyvoice_model.add_zero_shot_spk(_prepare_prompt_text(prompt_text), tmp_path, vid)
        cost_ms = (time.perf_counter() - t0) * 1000
        logger.info(f'[voice/clone] 音色特征提取完成 → voice_id={vid}，耗时 {cost_ms:.0f} ms')

        _save_voice(vid, prompt_text)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    total_ms = (time.perf_counter() - request_time) * 1000
    return {
        'voice_id': vid,
        'message': f'音色克隆成功，耗时 {total_ms:.0f} ms',
    }


# ── 接口：列出所有克隆音色 ──────────────────────────────────────────────────
@app.get('/voice/list')
async def voice_list():
    """返回所有已保存的克隆音色列表"""
    with _voice_meta_lock:
        voices = list(_voice_meta.values())
    return {
        'count': len(voices),
        'default_voice_id': default_spk_id or '',
        'voices': voices,
    }


# ── 接口：删除克隆音色 ──────────────────────────────────────────────────────
class VoiceDeleteRequest(BaseModel):
    voice_id: str

@app.post('/voice/delete')
async def voice_delete(req: VoiceDeleteRequest):
    """删除指定的克隆音色"""
    vid = req.voice_id.strip()
    if not vid:
        raise HTTPException(status_code=400, detail='voice_id 不能为空')
    if vid == default_spk_id:
        raise HTTPException(status_code=400, detail='不允许删除默认音色')

    with _voice_meta_lock:
        if vid not in _voice_meta:
            raise HTTPException(status_code=404, detail=f'音色 {vid} 不存在')

    _delete_voice(vid)
    return {'message': f'音色 {vid} 已删除'}


# ── 接口：查看缓存状态 ────────────────────────────────────────────────────────
@app.get('/spk_cache')
async def spk_cache_info():
    with _spk_cache_lock:
        return {
            'cached_count': len(_spk_cache),
            'spk_ids': list(_spk_cache.values()),
        }


# ── 接口：健康检查 ────────────────────────────────────────────────────────────
@app.get('/health')
async def health():
    spks = cosyvoice_model.list_available_spks() if hasattr(cosyvoice_model, 'list_available_spks') else []
    return {
        'status': 'ok',
        'model': getattr(cosyvoice_model, 'model_dir', 'unknown'),
        'sample_rate': cosyvoice_model.sample_rate,
        'available_spks': spks,
    }


# ── 启动入口 ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CosyVoice 流式 TTS 服务')
    parser.add_argument('--model_dir', type=str,
                        default='pretrained_models/Fun-CosyVoice3-0.5B',
                        help='模型目录（本地路径或 modelscope repo id）')
    parser.add_argument('--port', type=int, default=6006, help='监听端口')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='监听地址')
    parser.add_argument('--fp16', action='store_true', help='启用 FP16 推理')
    parser.add_argument('--load_trt', action='store_true', help='启用 TensorRT 加速 flow decoder')
    parser.add_argument('--load_vllm', action='store_true', help='启用 vLLM 加速 LLM 推理')
    # 默认音色参数（可选）
    parser.add_argument('--prompt_wav', type=str, default='',
                        help='默认参考音频路径，配置后启用 /tts 简化接口')
    parser.add_argument('--prompt_text', type=str, default='',
                        help='默认参考音频对应的文本')
    parser.add_argument('--spk_id', type=str, default='default_spk',
                        help='默认音色 ID（/tts 接口使用）')
    args = parser.parse_args()

    if args.load_vllm:
        _register_vllm_model()

    logger.info(f'正在加载模型: {args.model_dir} | fp16={args.fp16} | trt={args.load_trt} | vllm={args.load_vllm}')
    cosyvoice_model = AutoModel(model_dir=args.model_dir, fp16=args.fp16,
                                load_trt=args.load_trt, load_vllm=args.load_vllm)
    logger.info(f'模型加载完成，采样率: {cosyvoice_model.sample_rate} Hz')

    # 预注册默认音色
    if args.prompt_wav:
        if not os.path.exists(args.prompt_wav):
            raise FileNotFoundError(f'prompt_wav 不存在: {args.prompt_wav}')
        default_spk_id = args.spk_id
        t0 = time.perf_counter()
        cosyvoice_model.add_zero_shot_spk(_prepare_prompt_text(args.prompt_text), args.prompt_wav, default_spk_id)
        cost_ms = (time.perf_counter() - t0) * 1000
        logger.info(f'默认音色注册完成 | spk_id={default_spk_id} | 耗时 {cost_ms:.0f} ms')
        logger.info(f'已启用 /tts 简化接口（只需传 tts_text，不传 voice_id 默认使用 {default_spk_id}）')

    # 加载已持久化的克隆音色
    _init_voice_store(ROOT_DIR)

    logger.info(f'服务启动，监听 {args.host}:{args.port}')
    uvicorn.run(app, host=args.host, port=args.port, log_level='warning')

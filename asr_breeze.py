"""
離線台語 / 中文語音辨識（Breeze-ASR-26）

提供兩種模式：
1) 函式模式：transcribe(audio_bytes, filename)
2) 即時模式：python asr_breeze.py
"""

from __future__ import annotations

import argparse
import logging
import queue
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

MODEL_ID = "MediaTek-Research/Breeze-ASR-26"


def _status_line(state: str, message: str) -> str:
    ts = time.strftime("%H:%M:%S")
    return f"[{ts}][{state}] {message}"


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pipeline_device_arg(device: str):
    if device == "cuda":
        return 0
    if device == "cpu":
        return -1
    return torch.device(device)


class BreezeASR:
    """延遲載入的 Breeze-ASR-26 單例，首次呼叫 transcribe() 時才下載模型。"""

    def __init__(self) -> None:
        self._pipe = None

    # ------------------------------------------------------------------
    def _load(self) -> None:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        t0 = time.monotonic()
        device = _best_device()
        # MPS 使用 float32 較穩定，避免部分 float16 kernel 不支援。
        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe_device = _pipeline_device_arg(device)
        logger.info("狀態：開始載入模型 %s（裝置：%s）", MODEL_ID, device)
        logger.info("狀態：首次執行可能需要下載模型，請稍候")

        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                MODEL_ID,
                dtype=dtype,
                low_cpu_mem_usage=True,
            )
        except OSError as exc:
            if getattr(exc, "errno", None) == 1455 or "pagefile" in str(exc).lower():
                raise OSError(
                    "模型載入失敗：頁面檔或虛擬記憶體太小，請增加 Windows 頁面檔大小或在記憶體較大的環境中執行。"
                ) from exc
            raise

        model.to(device)
        logger.info("狀態：模型權重已就緒")

        processor = AutoProcessor.from_pretrained(MODEL_ID)
        logger.info("狀態：前處理器已就緒")

        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            dtype=dtype,
            device=pipe_device,
        )
        elapsed = time.monotonic() - t0
        logger.info("狀態：Breeze-ASR-26 模型載入完成（%.1f 秒）", elapsed)

    # ------------------------------------------------------------------
    def transcribe_array(self, audio: np.ndarray, sampling_rate: int = 16000) -> str:
        """傳入單聲道 float32 波形陣列，回傳辨識文字。"""
        if self._pipe is None:
            self._load()

        if audio.ndim != 1:
            raise ValueError("audio 必須為單聲道一維陣列")

        # 避免傳入 NaN/inf 造成推論失敗
        audio = np.nan_to_num(audio.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)

        result = self._pipe(
            {"array": audio, "sampling_rate": sampling_rate},
            generate_kwargs={"task": "transcribe"},
            return_timestamps=False,
        )
        return result["text"].strip()

    # ------------------------------------------------------------------
    def transcribe(self, audio_bytes: bytes, filename: str = "audio.webm") -> str:
        """
        傳入原始音訊位元組，回傳辨識文字。
        支援 WAV / WebM / MP3 / OGG 等格式（需系統安裝 ffmpeg）。
        """
        try:
            import librosa
        except Exception as exc:
            raise RuntimeError(
                "librosa/Scipy 依賴載入失敗：請安裝 librosa、scipy，並確認 Windows 頁面檔或虛擬記憶體足夠。"
            ) from exc

        if self._pipe is None:
            self._load()

        suffix = Path(filename).suffix or ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = Path(tmp.name)

        try:
            # librosa 統一重取樣至 16 kHz 單聲道
            audio, _ = librosa.load(str(tmp_path), sr=16000, mono=True)
            return self.transcribe_array(audio, sampling_rate=16000)
        finally:
            tmp_path.unlink(missing_ok=True)


# 全域單例
_asr = BreezeASR()


def transcribe(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    """模組層級便利函式。"""
    return _asr.transcribe(audio_bytes, filename)


def run_realtime(
    chunk_seconds: float = 3.0,
    samplerate: int = 16000,
    status_interval: float = 8.0,
) -> int:
    """
    麥克風即時分段辨識：每 chunk_seconds 秒輸出一次文字。
    按 Ctrl+C 結束。
    """
    try:
        import sounddevice as sd
    except Exception:
        print(_status_line("ERROR", "缺少 sounddevice，請先安裝：pip install sounddevice"), file=sys.stderr)
        return 1

    if chunk_seconds <= 0:
        print(_status_line("ERROR", "chunk_seconds 必須大於 0"), file=sys.stderr)
        return 1

    if status_interval <= 0:
        print(_status_line("ERROR", "status_interval 必須大於 0"), file=sys.stderr)
        return 1

    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    chunk_samples = int(chunk_seconds * samplerate)
    transcript_lines: list[str] = []
    last_status_at = time.monotonic()
    last_result_at = time.monotonic()

    def callback(indata, frames, callback_time, status):
        del frames, callback_time
        if status:
            logger.warning("音訊輸入狀態：%s", status)
        audio_queue.put(indata[:, 0].copy())

    print(_status_line("BOOT", "即時辨識啟動中...（Ctrl+C 結束）"))
    print(_status_line("CONFIG", f"每 {chunk_seconds:.1f} 秒切一段做辨識"))
    print(_status_line("LOADING", "預先載入模型中，首次可能需等待較久"))

    try:
        _asr._load()
    except Exception as exc:
        print(_status_line("ERROR", f"模型載入失敗：{exc}"), file=sys.stderr)
        return 1

    print(_status_line("READY", "模型已就緒，開始監聽麥克風"))

    buffered = np.empty((0,), dtype=np.float32)

    try:
        with sd.InputStream(
            samplerate=samplerate,
            channels=1,
            dtype="float32",
            callback=callback,
            blocksize=0,
        ):
            while True:
                piece = audio_queue.get()
                if piece.size == 0:
                    continue
                buffered = np.concatenate([buffered, piece])

                now = time.monotonic()
                if now - last_status_at >= status_interval:
                    idle_sec = now - last_result_at
                    print(_status_line("LISTEN", f"監聽中，尚未輸出新結果（已 {idle_sec:.0f} 秒）"))
                    last_status_at = now

                while buffered.shape[0] >= chunk_samples:
                    chunk = buffered[:chunk_samples]
                    buffered = buffered[chunk_samples:]

                    # 太安靜則略過，避免一直輸出空白
                    if float(np.abs(chunk).mean()) < 0.003:
                        continue

                    print(_status_line("ASR", "偵測到語音，辨識中..."))
                    t0 = time.monotonic()
                    text = _asr.transcribe_array(chunk, sampling_rate=samplerate)
                    cost = time.monotonic() - t0
                    if text:
                        ts = time.strftime("%H:%M:%S")
                        line = f"[{ts}][TEXT] {text}"
                        transcript_lines.append(line)
                        print(line)
                        print(_status_line("DONE", f"本段辨識完成（{cost:.1f} 秒）"))
                        last_result_at = time.monotonic()
                        last_status_at = time.monotonic()
    except KeyboardInterrupt:
        print("\n" + _status_line("STOP", "已停止即時辨識"))
        if transcript_lines:
            print("\n=== 辨識結果彙整 ===")
            print("\n".join(transcript_lines))
        return 0
    except Exception as exc:
        print(_status_line("ERROR", f"即時辨識失敗：{exc}"), file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Breeze-ASR-26 台語/中文離線語音辨識")
    parser.add_argument("audio", nargs="?", help="音訊檔路徑（不填則進入即時辨識）")
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=3.0,
        help="即時辨識分段秒數（預設 3.0）",
    )
    parser.add_argument(
        "--samplerate",
        type=int,
        default=16000,
        help="即時辨識採樣率（預設 16000）",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=8.0,
        help="狀態心跳秒數（預設 8.0）",
    )
    args = parser.parse_args()

    if args.audio:
        audio_path = Path(args.audio)
        if not audio_path.exists() or not audio_path.is_file():
            print(_status_line("ERROR", f"找不到音訊檔：{audio_path}"), file=sys.stderr)
            return 1
        try:
            print(_status_line("ASR", f"開始辨識音檔：{audio_path.name}"))
            text = transcribe(audio_path.read_bytes(), audio_path.name)
            print(_status_line("TEXT", text or "（無辨識結果）"))
            return 0
        except Exception as exc:
            print(_status_line("ERROR", f"檔案辨識失敗：{exc}"), file=sys.stderr)
            return 1

    return run_realtime(
        chunk_seconds=args.chunk_seconds,
        samplerate=args.samplerate,
        status_interval=args.status_interval,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())

#!/usr/bin/env python3
"""
일본어 동영상 → 한국어 자막(.srt) 변환기

Dependencies:
    pip install openai-whisper ffmpeg-python deep-translator tqdm
    (ffmpeg 바이너리도 설치 필요: apt install ffmpeg / brew install ffmpeg)
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import whisper
from deep_translator import GoogleTranslator
from tqdm import tqdm


# ── SRT 유틸 ──────────────────────────────────────────────────────────────────

def seconds_to_srt_time(seconds: float) -> str:
    """초(float) → SRT 타임코드 문자열 (HH:MM:SS,mmm)"""
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(segments: list[dict]) -> str:
    """Whisper segments 리스트 → SRT 문자열"""
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = seconds_to_srt_time(seg["start"])
        end   = seconds_to_srt_time(seg["end"])
        text  = (seg.get("translated") or seg.get("text") or "").strip()
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


# ── 캐시 ──────────────────────────────────────────────────────────────────────

def cache_path(video_path: str) -> str:
    return str(Path(video_path).with_suffix(".whisper.json"))


def get_video_duration(video_path: str) -> float:
    """ffprobe로 동영상 길이(초) 반환"""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        return 0.0
    return float(result.stdout.strip())


def save_cache(segments: list[dict], path: str, duration: float, detected_lang: str = "auto") -> None:
    slim = [
        {"start": s["start"], "end": s["end"], "text": s["text"]}
        for s in segments
    ]
    data = {
        "completed": True,
        "video_duration": duration,
        "detected_lang": detected_lang,
        "segments": slim,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"      → 캐시 저장: {path}")


def _read_detected_lang(path: str) -> str:
    """캐시 파일에서 감지된 언어 코드를 읽음. 없으면 'auto' 반환."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("detected_lang", "auto")
    except Exception:
        return "auto"


def load_cache(path: str, duration: float) -> list[dict] | None:
    """캐시를 로드하고 유효성을 검증. 문제가 있으면 None 반환."""
    if not os.path.isfile(path):
        return None

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [경고] 캐시 파일 손상 ({e}) → 음성 인식 재실행")
        return None

    # 구 형식(segments 배열 직접 저장) 호환
    if isinstance(data, list):
        segments = data
        completed = False
    else:
        completed = data.get("completed", False)
        segments = data.get("segments", [])

    if not completed:
        print("  [경고] 캐시가 완료되지 않은 상태 → 음성 인식 재실행")
        return None

    if not segments:
        print("  [경고] 캐시에 세그먼트 없음 → 음성 인식 재실행")
        return None

    if duration > 0:
        last_end = segments[-1]["end"]
        coverage = last_end / duration
        if coverage < 0.9:
            print(
                f"  [경고] 캐시 커버리지 부족 "
                f"({last_end:.1f}s / {duration:.1f}s = {coverage:.0%}) "
                f"→ 음성 인식 재실행"
            )
            return None

    print(f"      → 캐시 로드: {path}  ({len(segments)}개 세그먼트)")
    return segments


# ── 오디오 추출 ───────────────────────────────────────────────────────────────

def extract_audio(video_path: str, audio_path: str) -> None:
    """ffmpeg으로 동영상에서 16kHz 모노 WAV 추출 (Whisper 권장 포맷)"""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg 오류:\n{result.stderr.decode(errors='replace')}"
        )


# ── 번역 ──────────────────────────────────────────────────────────────────────

# Google Translate 무료 API 제한: 초당 5회
_TRANSLATE_INTERVAL = 1.0 / 4  # 4회/초로 안전하게 제한


def translate_segments(segments: list[dict], source_lang: str = "auto") -> list[dict]:
    """각 segment의 텍스트를 한국어로 번역하여 'translated' 키 추가"""
    translator = GoogleTranslator(source=source_lang, target="ko")
    texts = [seg["text"].strip() for seg in segments]

    translated = []
    last_request_time = 0.0

    for text in tqdm(texts, desc="번역 중", unit="seg"):
        if text:
            # 속도 제한: 이전 요청으로부터 충분한 시간이 지나지 않았으면 대기
            elapsed = time.monotonic() - last_request_time
            if elapsed < _TRANSLATE_INTERVAL:
                time.sleep(_TRANSLATE_INTERVAL - elapsed)

            try:
                last_request_time = time.monotonic()
                result = translator.translate(text)
                translated.append(result if result is not None else text)
            except Exception as e:
                error_msg = str(e)
                if "too many requests" in error_msg.lower() or "429" in error_msg:
                    # 속도 초과 시 5초 대기 후 재시도
                    print(f"\n  [경고] 요청 초과, 5초 대기 후 재시도...")
                    time.sleep(5)
                    try:
                        last_request_time = time.monotonic()
                        result = translator.translate(text)
                        translated.append(result if result is not None else text)
                    except Exception:
                        print(f"  [경고] 재시도 실패, 원문 유지: {text}")
                        translated.append(text)
                else:
                    print(f"  [경고] 번역 실패 ({e}), 원문 유지: {text}")
                    translated.append(text)
        else:
            translated.append("")

    for seg, ko in zip(segments, translated):
        seg["translated"] = ko

    return segments


# ── 메인 파이프라인 ───────────────────────────────────────────────────────────

def process(
    video_path: str,
    model_size: str = "medium",
    output_path: str | None = None,
    force: bool = False,
    language: str | None = None,
) -> str:
    video_path = str(Path(video_path).resolve())
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {video_path}")

    if output_path is None:
        output_path = str(Path(video_path).with_suffix(".srt"))

    whisper_cache = cache_path(video_path)
    duration = get_video_duration(video_path)
    if duration > 0:
        print(f"      동영상 길이: {duration:.1f}초 ({duration/60:.1f}분)")

    segments = None if force else load_cache(whisper_cache, duration)

    if segments is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "audio.wav")

            print(f"[1/4] 오디오 추출: {video_path}")
            extract_audio(video_path, audio_path)

            lang_label = language if language else "자동 감지"
            print(f"[2/4] Whisper({model_size}) 음성 인식 중 … (언어: {lang_label})")
            model = whisper.load_model(model_size)
            transcribe_opts = {"verbose": False, "fp16": False}
            if language:
                transcribe_opts["language"] = language
            result = model.transcribe(audio_path, **transcribe_opts)
            segments = result["segments"]
            detected_lang = result.get("language", language or "unknown")
            print(f"      → 감지된 언어: {detected_lang}")
            print(f"      → {len(segments)}개 세그먼트 인식 완료")
            save_cache(segments, whisper_cache, duration, detected_lang)
    else:
        print(f"[1-2/4] 음성 인식 캐시 사용 (건너뜀)")

    detected_lang = _read_detected_lang(whisper_cache)

    print(f"[3/4] 한국어 번역 중 … ({detected_lang} → ko)")
    segments = translate_segments(segments, source_lang=detected_lang)

    print(f"[4/4] 자막 파일 저장: {output_path}")
    srt_content = build_srt(segments)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    print(f"\n완료! 자막 파일: {output_path}")
    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="동영상(.mp4)을 입력받아 한국어 자막(.srt)을 생성합니다."
    )
    parser.add_argument("video", help="입력 동영상 파일 경로 (.mp4)")
    parser.add_argument(
        "-m", "--model",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper 모델 크기 (기본값: medium / 정확도↑ → large-v3 권장)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="출력 .srt 파일 경로 (기본값: 입력 파일과 동일 경로)",
    )
    parser.add_argument(
        "-l", "--language",
        default=None,
        help="음성 언어 직접 지정 (예: ja, en, zh, ko). 생략 시 자동 감지",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="캐시를 무시하고 음성 인식부터 다시 실행",
    )
    args = parser.parse_args()
    process(args.video, model_size=args.model, output_path=args.output, force=args.force, language=args.language)


if __name__ == "__main__":
    main()

"""
Audio Quality Analysis Utility
------------------------------
Mô tả: Công cụ quét tập dữ liệu âm thanh để phát hiện các file không đạt chất lượng 
       (quá ngắn hoặc quá tĩnh/im lặng). 
Định dạng đầu ra: Một file JSON chứa danh sách các file lỗi phân loại theo tập Train/Test.

Tác giả: DucLai
Ngày: 11/03/2026
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import torch
import torchaudio

import configs.quality_required as cfg


def check_single_audio(file_path: Path) -> Dict[str, Any]:
    """
    Phân tích một file âm thanh đơn lẻ để lấy thông tin thời lượng và biên độ.
    
    Args:
        file_path: Đối tượng Path dẫn đến file .wav
        
    Returns:
        Dict chứa duration và mean_amplitude.
    """
    waveform, sample_rate = torchaudio.load(str(file_path))

    duration = waveform.shape[1] / sample_rate
    mean_amp = torch.mean(torch.abs(waveform)).item()

    return {
        "duration": duration,
        "mean_amplitude": mean_amp
    }


def analyze_audio_file(audio_file: Path, speaker: str, results: Dict[str, Any]) -> None:
    try:
        stats = check_single_audio(audio_file)

        if stats["duration"] < cfg.SHORT_DURATION_THRESHOLD:
            results["short_audios"].append({
                "speaker": speaker,
                "file": audio_file.name,
                "duration": round(stats["duration"], 3)
            })

        if stats["mean_amplitude"] < cfg.SILENCE_THRESHOLD:
            results["silent_audios"].append({
                "speaker": speaker,
                "file": audio_file.name,
                "mean_amplitude": f"{stats['mean_amplitude']:.6f}"
            })

    except Exception as e:
        print(f"Lỗi khi xử lý file {audio_file}: {e}")


def analyze_dataset(root_dir: str) -> Dict[str, Any]:
    """
    Duyệt qua cấu trúc thư mục dataset (speaker/file.wav) để phân tích chất lượng.
    
    Args:
        root_dir: Đường dẫn đến thư mục gốc của tập dữ liệu.
        
    Returns:
        Thống kê tổng hợp về số lượng file lỗi và danh sách chi tiết.
    """
    root_path = Path(root_dir)

    results = {
        "total_audio": 0,
        "short_audios": [],
        "silent_audios": []
    }

    if not root_path.exists():
        print(f"Cảnh báo: Thư mục {root_dir} không tồn tại.")
        return results

    for audio_file in root_path.glob("*/*.wav"):
        results["total_audio"] += 1
        speaker = audio_file.parent.name
        analyze_audio_file(audio_file, speaker, results)

    return {
        "total_audio": results["total_audio"],
        "total_short_audio": len(results["short_audios"]),
        "total_silent_audio": len(results["silent_audios"]),
        "short_audio_files": results["short_audios"],
        "silent_audio_files": results["silent_audios"]
    }


def create_output_path(log_dir: str) -> Path:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_path / f"unquality_audios_{timestamp}.json"


def print_summary(report: Dict[str, Any]) -> None:
    for split in report:
        print(
            f"{split.capitalize()} - Tổng: {report[split]['total_audio']} | "
            f"Ngắn < 3s: {report[split]['total_short_audio']} | "
            f"Im lặng: {report[split]['total_silent_audio']}"
        )


def main():
    data_configs = {
        "train": "data/raw/vivos/train/waves",
        "test": "data/raw/vivos/test/waves"
    }

    output_file = create_output_path("logs/analysis_logs")

    final_report = {}

    for split_name, path in data_configs.items():
        print(f"Đang phân tích tập {split_name.upper()}...")
        analysis = analyze_dataset(path)
        final_report[split_name] = analysis
        print(f"Hoàn thành: {analysis['total_audio']} files.")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)

    print(f"\nBáo cáo đã được lưu tại: {output_file}")

    print_summary(final_report)


if __name__ == "__main__":
    main()
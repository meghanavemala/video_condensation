import os
import json
import time
import torch
import whisper
from moviepy import VideoFileClip

class AudioSubtitleExtractor:
    def __init__(self, video_path: str, base_output_dir: str = "output_data", model_size: str = "base"):
        self.video_path = video_path
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.output_dir = os.path.join(base_output_dir, self.video_name)
        os.makedirs(self.output_dir, exist_ok=True)

        self.audio_path = os.path.join(self.output_dir, f"{self.video_name}_audio.wav")
        self.subtitle_path = os.path.join(self.output_dir, f"{self.video_name}_subtitles.srt")
        self.metadata_path = os.path.join(self.output_dir, f"{self.video_name}_metadata.json")
        self.model_size = model_size

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_size).to(self.device)

    def extract_audio(self) -> float:
        print("[INFO] Extracting audio from video...")
        start_time = time.time()
        try:
            clip = VideoFileClip(self.video_path)
            duration = clip.duration  # in seconds
            clip.audio.write_audiofile(self.audio_path, logger=None)
            elapsed = time.time() - start_time
            print(f"✅ Audio saved at: {self.audio_path} (took {elapsed:.2f} sec)")
            return duration, elapsed
        except Exception as e:
            print(f"[ERROR] Audio extraction failed: {e}")
            return 0.0, 0.0

    def transcribe_audio(self) -> (dict, float):
        print("[INFO] Transcribing audio to subtitles...")
        start_time = time.time()
        try:
            if not os.path.exists(self.audio_path):
                raise FileNotFoundError(f"Audio file not found at: {self.audio_path}")

            result = self.model.transcribe(self.audio_path)
            elapsed = time.time() - start_time
            print(f"✅ Transcription complete (took {elapsed:.2f} sec)")
            return result, elapsed
        except Exception as e:
            print(f"[ERROR] Transcription failed: {e}")
            return {}, 0.0

    def save_subtitles(self, result: dict) -> str:
        print("[INFO] Saving subtitles...")
        try:
            with open(self.subtitle_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(result['segments']):
                    start = self._format_timestamp(segment["start"])
                    end = self._format_timestamp(segment["end"])
                    text = segment["text"].strip()
                    f.write(f"{i+1}\n{start} --> {end}\n{text}\n\n")
            print(f"✅ Subtitles saved at: {self.subtitle_path}")
            return self.subtitle_path
        except Exception as e:
            print(f"[ERROR] Saving subtitles failed: {e}")
            return ""

    def _format_timestamp(self, seconds: float) -> str:
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"

    def save_metadata(self, video_length: float, audio_time: float, transcribe_time: float):
        metadata = {
            "video_file": os.path.basename(self.video_path),
            "video_length_sec": round(video_length, 2),
            "audio_extraction_time_sec": round(audio_time, 2),
            "transcription_time_sec": round(transcribe_time, 2),
            "total_processing_time_sec": round(audio_time + transcribe_time, 2),
            "device_used": self.device
        }
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"✅ Metadata saved at: {self.metadata_path}")

    def run(self):
        video_length, audio_time = self.extract_audio()
        result, transcribe_time = self.transcribe_audio()
        if result:
            self.save_subtitles(result)
            self.save_metadata(video_length, audio_time, transcribe_time)

if __name__ == "__main__":
    video_path = "Input_data/maths.mp4"
    extractor = AudioSubtitleExtractor(video_path, base_output_dir= "output_data", model_size="base")
    extractor.run()

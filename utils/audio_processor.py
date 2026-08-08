import yt_dlp 
from pydub import AudioSegment
import os

DOWNLOAD_DIR = 'downloads'

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_youtube_audio(url:str) -> str:
  output_path = os.path.join(DOWNLOAD_DIR, '%(title)s.%(ext)s')
  ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': output_path,
    'postprocessors': [{
      'key': 'FFmpegExtractAudio',
      'preferredcodec': 'wav',
      'preferredquality': '192'
    }
    ],
    "quiet": True
  }
  with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info_dict = ydl.extract_info(url, download=True)
    filename = ydl.prepare_filename(info_dict)
    wav_filename = os.path.splitext(filename)[0] + '.wav'
  return wav_filename




def convert_to_wav(input_file:str) -> str:
  output_path = os.path.splitext(input_file)[0] + '._converted.wav'
  audio = AudioSegment.from_file(input_file)
  audio = audio.set_channels(1).set_frame_rate(16000)  # Convert to mono and set frame rate
  audio.export(output_path, format='wav')
  return output_path




def chunk_audio(input_file:str, chunk_length_ms:int=60000) -> list:
  audio = AudioSegment.from_wav(input_file)
  chunks = []
  for i in range(0, len(audio), chunk_length_ms):
    chunk = audio[i:i + chunk_length_ms]
    chunk_filename = os.path.splitext(input_file)[0] + f'_chunk_{i // chunk_length_ms}.wav'
    chunk.export(chunk_filename, format='wav')
    chunks.append(chunk_filename)
  return chunks


def process_audio_file(input_file:str) -> list:
  if input_file.startswith('http'):
    input_file = download_youtube_audio(input_file)
  # Convert to WAV if not already
  if not input_file.lower().endswith('.wav'):
    input_file = convert_to_wav(input_file)
  
  # Chunk the audio file
  chunks = chunk_audio(input_file)
  
  return chunks 
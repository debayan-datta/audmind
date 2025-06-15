# import os
# import re
# #import ffmpeg
# from git import Repo

# #from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_audio
# #from moviepy.editor import VideoFileClip

# #from moviepy.editor import VideoFileClip
# from moviepy import *
# from pydub import AudioSegment

# from pytubefix import YouTube
# import time




# def download_video(url, max_retries=3):
#     retries = 0
#     while retries < max_retries:
#         try:
#             print(f"\nDownloading: {url}")
#             yt = YouTube(url)  # Removed use_oauth parameters
#             print(f"Downloading: {yt.title}")
#             video_stream = yt.streams.get_lowest_resolution()
#             video_id = yt.video_id
#             video_path = os.path.join("Videos", f"{video_id}.mp4")
#             video_stream.download(output_path="Videos", filename=f"{video_id}.mp4")
#             print("Download complete!")
#             return True, video_path, video_id
        
#         except Exception as e:
#             print(f"Error downloading {url}: {e}")
#             retries += 1
#             time.sleep(2)  # Wait a bit before retrying
            
#     print(f"Skipping {url} due to repeated errors.")
#     return False, None, None



# if __name__ == "__main__":
        
#     videos_dir = "Videos"
#     audios_dir = "Audios"
#     os.makedirs(videos_dir, exist_ok=True)
#     os.makedirs(audios_dir, exist_ok=True)
    
#     print("Hello")

#     with open('URL_sources.txt', 'r') as file:
#         for i, line in enumerate(file):
#             url = line.strip()
            
#             print(i,"hello world")
#             try:


#                 result, video_path, video_id = download_video(url)

#                 # Extract audio using moviepy
#                 audio_path = os.path.join(audios_dir, f"{video_id}.wav")
#                 video_clip = VideoFileClip(video_path)
#                 video_clip.audio.write_audiofile(audio_path)
#                 video_clip.close()
#                 print(f"Extracted audio: {audio_path}")

#                 # Create a folder for this video's audio in the Audios directory
#                 audio_output_folder = os.path.join(audios_dir, video_id)
#                 os.makedirs(audio_output_folder, exist_ok=True)

#                 # Split audio into 30-second chunks
#                 audio = AudioSegment.from_wav(audio_path)
#                 duration = len(audio)  # Total duration in milliseconds
#                 chunk_size = 30 * 1000  # 30 seconds in milliseconds

#                 for i, start_time in enumerate(range(0, duration, chunk_size)):
#                     end_time = min(start_time + chunk_size, duration)
#                     chunk = audio[start_time:end_time]
#                     chunk_filename = f"{video_id}_part{i+1}.wav"
#                     chunk_path = os.path.join(audio_output_folder, chunk_filename)
#                     chunk.export(chunk_path, format="wav")
#                     # print(f"Saved chunk: {chunk_filename}")

#                 # Delete the original video and full extracted audio to free space
#                 os.remove(video_path)
#                 os.remove(audio_path)
#                 print(f"Deleted temporary files for {video_id}")


#             except Exception as e:
#                 print(f"Skipping {url} due to error: {e}")

#     print("\n Process complete.")

#------------- had to change the code due to some problems in PYTUBE---------------#

import os
import subprocess
from pytubefix import YouTube
from pytubefix.cli import on_progress
import ffmpeg

# Path to the text file containing YouTube URLs
text_file = "URL_sources.txt"

# Output directory
output_dir = "YT_videos"
os.makedirs(output_dir, exist_ok=True)

# Read YouTube links
with open(text_file, 'r') as f:
    links = [line.strip() for line in f if line.strip()]


# Download each video's audio as .m4a
for url in links:
    try:
        yt = YouTube(url, on_progress_callback=on_progress)
        video_id = yt.video_id
        # print(f"Downloading audio for: {video_id} - {yt.title}")

        # Get audio-only stream
        ys = yt.streams.get_audio_only()
        
        # Download audio as .m4a format
        ys.download(output_path=output_dir, filename=f"{video_id}.wav")
        
        print(f"Saved: {video_id}.wav\n")
    except Exception as e:
        print(f"Failed to process {url}: {e}")
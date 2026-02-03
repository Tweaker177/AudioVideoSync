#!/usr/bin/env python3
"""
Audio-Video Sync Tool - Intelligent Multi-Point Alignment
Handles offset detection, preserves audio quality
"""

import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
from pydub import AudioSegment
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import tkinter as tk
from tkinter import filedialog
import os
import subprocess

def select_file(title, filetypes):
    """Open file picker dialog"""
    root = tk.Tk()
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)
    filepath = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return filepath

def extract_audio_from_video(video_path, temp_audio_path="temp_video_audio.wav"):
    """Extract audio track from video for analysis"""
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(temp_audio_path, codec='pcm_s16le', verbose=False, logger=None)
    video.close()
    return temp_audio_path

def get_audio_codec_info(audio_path):
    """Detect original audio format to preserve it"""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'a:0', 
             '-show_entries', 'stream=codec_name,bit_rate', 
             '-of', 'default=noprint_wrappers=1', audio_path],
            capture_output=True,
            text=True
        )
        
        codec = None
        bitrate = None
        
        for line in result.stdout.split('\n'):
            if 'codec_name=' in line:
                codec = line.split('=')[1]
            if 'bit_rate=' in line:
                bitrate = line.split('=')[1]
        
        return codec, bitrate
    except:
        return None, None

def load_audio_as_array(audio_path, target_sr=8000, max_duration=120):
    """Load audio file as numpy array for correlation"""
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1).set_frame_rate(target_sr)
    
    # Limit to first max_duration seconds for speed
    max_samples = target_sr * max_duration
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)[:max_samples]
    samples = samples / (np.max(np.abs(samples)) + 1e-8)
    
    return samples, target_sr

def find_offset_cross_correlation(video_audio_path, target_audio_path):
    """
    Find time offset between video audio and target audio using cross-correlation
    Returns offset in seconds (positive = target starts later, negative = target starts earlier)
    """
    print("Detecting offset between audio tracks...")
    
    # Load both audio files
    video_samples, sr = load_audio_as_array(video_audio_path)
    target_samples, _ = load_audio_as_array(target_audio_path)
    
    # Use shorter of the two for correlation
    min_len = min(len(video_samples), len(target_samples))
    video_samples = video_samples[:min_len]
    target_samples = target_samples[:min_len]
    
    # Cross-correlate
    correlation = signal.correlate(target_samples, video_samples, mode='full')
    
    # Find peak
    lag_samples = np.argmax(correlation) - (len(video_samples) - 1)
    offset = lag_samples / sr
    
    print(f"  Detected offset: {offset:.3f} seconds")
    if offset > 0:
        print(f"  (Target audio starts {offset:.3f}s after video)")
    elif offset < 0:
        print(f"  (Target audio starts {-offset:.3f}s before video)")
    else:
        print(f"  (No significant offset detected)")
    
    return offset

def compute_chroma_features(audio_path, sr=22050, hop_length=512, start_time=0, duration=None):
    """
    Compute chroma features (pitch content over time)
    Can specify start_time and duration to analyze only a portion
    """
    print(f"Computing chroma features for {os.path.basename(audio_path)}...")
    if start_time > 0 or duration:
        print(f"  (from {start_time:.2f}s, duration {duration:.2f}s)" if duration else f"  (from {start_time:.2f}s)")
    
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1).set_frame_rate(sr)
    
    # Extract portion if needed
    if start_time > 0:
        start_ms = int(start_time * 1000)
        audio = audio[start_ms:]
    
    if duration:
        duration_ms = int(duration * 1000)
        audio = audio[:duration_ms]
    
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    samples = samples / (np.max(np.abs(samples)) + 1e-8)
    
    # Compute spectrogram
    f, t, Sxx = signal.spectrogram(samples, sr, nperseg=2048, noverlap=1536)
    
    # Map frequencies to chroma (12 pitch classes)
    chroma = np.zeros((12, Sxx.shape[1]))
    
    for i, freq in enumerate(f):
        if freq > 0:
            midi = 69 + 12 * np.log2(freq / 440.0)
            chroma_idx = int(np.round(midi)) % 12
            chroma[chroma_idx, :] += Sxx[i, :]
    
    # Normalize each frame
    chroma = chroma / (np.max(chroma, axis=0, keepdims=True) + 1e-8)
    
    # Time per frame
    time_per_frame = hop_length / sr
    
    return chroma, time_per_frame

def find_alignment_path_dtw(chroma1, chroma2, band_width=0.25):
    """
    DTW alignment with constrained band
    """
    n, m = chroma1.shape[1], chroma2.shape[1]
    
    # Constrained band width
    band = int(max(n, m) * band_width)
    
    print(f"Running DTW alignment ({n} x {m} frames, band={band})...")
    print("This may take a few minutes for long songs...")
    
    # Initialize DTW matrix
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0
    
    # Fill DTW matrix with band constraint
    for i in range(1, n + 1):
        j_start = max(1, i - band)
        j_end = min(m + 1, i + band)
        
        for j in range(j_start, j_end):
            # Euclidean distance between chroma vectors
            cost = np.sqrt(np.sum((chroma1[:, i-1] - chroma2[:, j-1]) ** 2))
            
            dtw[i, j] = cost + min(
                dtw[i-1, j],
                dtw[i, j-1],
                dtw[i-1, j-1]
            )
        
        if i % 100 == 0:
            pct = (i / n) * 100
            print(f"  Progress: {pct:.1f}%")
    
    # Backtrack
    path = []
    i, j = n, m
    
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        
        candidates = [
            (dtw[i-1, j-1], i-1, j-1),
            (dtw[i-1, j], i-1, j),
            (dtw[i, j-1], i, j-1)
        ]
        _, i, j = min(candidates)
    
    path.reverse()
    
    print(f"  Done! Path length: {len(path)}")
    return path

def extract_key_sync_points(path, time_per_frame, num_points=20):
    """
    Extract evenly-spaced sync points from DTW path with smoothing
    """
    # Sample path evenly
    indices = np.linspace(0, len(path) - 1, num_points, dtype=int)
    
    sync_points = []
    for idx in indices:
        video_frame, audio_frame = path[idx]
        video_time = video_frame * time_per_frame
        audio_time = audio_frame * time_per_frame
        sync_points.append((video_time, audio_time))
    
    # Smooth the alignment
    video_times = np.array([v for v, a in sync_points])
    audio_times = np.array([a for v, a in sync_points])
    
    # Apply Gaussian smoothing
    audio_times_smooth = gaussian_filter1d(audio_times, sigma=2.0)
    
    sync_points_smooth = list(zip(video_times, audio_times_smooth))
    
    return sync_points_smooth

def create_synced_video(video_path, audio_path, sync_points, video_offset, output_path):
    """
    Create time-stretched video using sync points
    Handles initial offset properly
    """
    print(f"\nLoading video...")
    video = VideoFileClip(video_path).without_audio()
    audio = AudioFileClip(audio_path)
    
    # Apply offset to video start time
    # If video_offset is positive, video needs to start later (trim from beginning)
    # If negative, video needs to start earlier (we'll handle with black frame padding later if needed)
    
    video_start_trim = max(0, video_offset)
    if video_start_trim > 0:
        print(f"Trimming {video_start_trim:.3f}s from video start to align with audio")
        video = video.subclip(video_start_trim)
    
    segments = []
    
    print(f"\nCreating {len(sync_points)-1} video segments...")
    
    for i in range(len(sync_points) - 1):
        vid_start, aud_start = sync_points[i]
        vid_end, aud_end = sync_points[i + 1]
        
        vid_duration = vid_end - vid_start
        aud_duration = aud_end - aud_start
        
        if vid_duration > 0.1 and aud_duration > 0.1:
            speed = vid_duration / aud_duration
            
            # Clamp to reasonable range
            speed = max(0.85, min(1.15, speed))
            
            try:
                vid_start = min(vid_start, video.duration - 0.1)
                vid_end = min(vid_end, video.duration)
                
                if vid_end > vid_start:
                    segment = video.subclip(vid_start, vid_end)
                    segment = segment.speedx(speed)
                    segments.append(segment)
                    
                    print(f"  Segment {i+1:2d}: {vid_start:6.2f}s -> {vid_end:6.2f}s | Speed: {speed:.3f}x")
                    
            except Exception as e:
                print(f"  Warning: Skipped segment {i+1}: {e}")
    
    if not segments:
        print("ERROR: No valid segments created!")
        return
    
    print("\nCombining segments...")
    final_video = concatenate_videoclips(segments, method="compose")
    
    # Match to audio duration
    final_duration = min(final_video.duration, audio.duration)
    final_video = final_video.subclip(0, final_duration)
    audio_trimmed = audio.subclip(0, final_duration)
    
    final_video = final_video.set_audio(audio_trimmed)
    
    # Determine audio encoding settings
    codec, bitrate = get_audio_codec_info(audio_path)
    
    audio_codec = 'aac'
    audio_bitrate = '320k'
    
    if codec == 'pcm_s16le' or codec == 'pcm_s24le' or audio_path.endswith('.wav'):
        audio_bitrate = '320k'
        print(f"\nDetected lossless audio - encoding at 320k AAC")
    elif codec == 'flac':
        audio_bitrate = '320k'
        print(f"\nDetected FLAC - encoding at 320k AAC")
    elif bitrate and int(bitrate) < 320000:
        audio_bitrate = f"{int(int(bitrate)/1000)}k"
        print(f"\nMatching original audio bitrate: {audio_bitrate}")
    else:
        print(f"\nUsing 320k AAC for audio")
    
    print(f"\nWriting final video...")
    print(f"  Output: {output_path}")
    print(f"  Duration: {final_duration:.2f}s")
    print(f"  Audio: {audio_codec} @ {audio_bitrate}")
    
    final_video.write_videofile(
        output_path,
        codec='libx264',
        audio_codec=audio_codec,
        audio_bitrate=audio_bitrate,
        preset='medium',
        bitrate='8000k',
        threads=4,
        verbose=False,
        logger=None
    )
    
    # Cleanup
    video.close()
    audio.close()
    audio_trimmed.close()
    final_video.close()
    for seg in segments:
        seg.close()

def main():
    print("=" * 60)
    print("Audio-Video Sync Tool - Offset + DTW Alignment")
    print("=" * 60)
    print()
    
    # Select files
    print("Select VIDEO file...")
    video_path = select_file("Select Video", [("Video files", "*.mp4 *.mov *.avi *.mkv")])
    if not video_path:
        print("Cancelled.")
        return
    print(f"Video: {os.path.basename(video_path)}\n")
    
    print("Select AUDIO file (target to sync to)...")
    audio_path = select_file("Select Audio", [("Audio files", "*.mp3 *.wav *.m4a *.aac *.flac")])
    if not audio_path:
        print("Cancelled.")
        return
    print(f"Audio: {os.path.basename(audio_path)}\n")
    
    # Extract video audio
    print("Extracting audio from video...")
    video_audio_path = extract_audio_from_video(video_path)
    print("Done\n")
    
    # Step 1: Find offset
    offset = find_offset_cross_correlation(video_audio_path, audio_path)
    
    # Step 2: Compute chroma on ALIGNED portions
    # Determine overlap region
    video_clip = VideoFileClip(video_path)
    video_duration = video_clip.duration
    video_clip.close()
    
    audio_clip = AudioFileClip(audio_path)
    audio_duration = audio_clip.duration
    audio_clip.close()
    
    # Calculate aligned start times and duration
    if offset >= 0:
        # Target audio starts later
        video_start = offset
        audio_start = 0
        overlap_duration = min(video_duration - offset, audio_duration)
    else:
        # Target audio starts earlier
        video_start = 0
        audio_start = -offset
        overlap_duration = min(video_duration, audio_duration + offset)
    
    print(f"\nAnalyzing overlapping region:")
    print(f"  Video: from {video_start:.2f}s, duration {overlap_duration:.2f}s")
    print(f"  Audio: from {audio_start:.2f}s, duration {overlap_duration:.2f}s\n")
    
    # Compute chroma on aligned portions
    video_chroma, time_step = compute_chroma_features(video_audio_path, start_time=video_start, duration=overlap_duration)
    audio_chroma, _ = compute_chroma_features(audio_path, start_time=audio_start, duration=overlap_duration)
    
    print(f"\nChroma features:")
    print(f"  Video: {video_chroma.shape[1]} frames")
    print(f"  Audio: {audio_chroma.shape[1]} frames\n")
    
    # Step 3: Run DTW on aligned portions
    alignment_path = find_alignment_path_dtw(video_chroma, audio_chroma)
    
    # Step 4: Extract sync points (these are relative to the aligned start times)
    print("\nExtracting sync points...")
    sync_points = extract_key_sync_points(alignment_path, time_step, num_points=20)
    
    # Adjust sync points back to absolute times in original files
    # Video times stay as-is (they're relative to video_start which we'll handle in video processing)
    # Audio times stay as-is (relative to audio_start which is 0 after we extract the portion)
    
    print(f"\nCreated {len(sync_points)} sync points")
    
    # Generate output path
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(os.path.dirname(video_path), f"{base_name}_synced.mp4")
    
    # Step 5: Create synced video
    create_synced_video(video_path, audio_path, sync_points, video_start, output_path)
    
    # Cleanup
    if os.path.exists(video_audio_path):
        os.remove(video_audio_path)
    
    print("\n" + "=" * 60)
    print("SUCCESS!")
    print(f"Synced video: {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()

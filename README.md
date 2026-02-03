# AudioVideoSync

Automatically synchronize video of a musical performance with audio from a different recording of the same song using intelligent multi-point alignment.

## Overview

This tool is designed for musicians who want to sync performance video with studio audio (or vice versa). It handles:

- **Different performances** of the same song (live vs. studio, different takes)
- **Tempo variations** between recordings
- **Initial offset detection** (recordings that don't start at exactly the same moment)
- **Audio quality preservation** (won't unnecessarily re-encode or upsample your audio)
- **Out-of-tune instruments** (robust to pitch variations and tuning differences)

## How It Works

1. **Offset Detection**: Uses cross-correlation to find the initial time offset between video and audio
2. **Chroma Analysis**: Extracts pitch content (chroma features) from both audio tracks
3. **Dynamic Time Warping (DTW)**: Finds the optimal alignment path between the two performances
4. **Smooth Speed Adjustment**: Creates 20 video segments with gentle speed adjustments (0.85x-1.15x) to maintain natural-looking sync
5. **Video Assembly**: Combines segments with your target audio at preserved quality

## Requirements

### System Requirements
- macOS (tested on High Sierra and newer)
- Python 3.7+
- ffmpeg

### Python Dependencies
```bash
pip install moviepy pydub numpy scipy

  

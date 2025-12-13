# SAM3 Video Segmentation - User Guide

## Overview

SAM3 Video Segmentation allows you to automatically segment objects in videos using simple text prompts. Powered by Meta's Segment Anything Model 3 (SAM 3), you can describe what you want to track and the system will generate segmentation masks for every frame.

## Getting Started

### 1. Upload Your Video

- **Supported formats**: MP4, WebM, MOV
- **Maximum file size**: 100MB
- **Maximum duration**: 60 seconds
- **Recommended resolution**: 720p or 1080p

Simply drag and drop your video onto the upload area, or click to browse your files.

### 2. Enter Your Prompt

Describe what you want to segment in the video. Be specific but concise:

**Good prompts:**
- "person"
- "red car"
- "dog"
- "basketball"
- "hand holding phone"

**Tips for better results:**
- Use singular nouns when possible
- Include color or distinguishing features if multiple similar objects exist
- Keep prompts short (1-3 words work best)

### 3. Wait for Processing

Processing time depends on video length and resolution:
- Short videos (< 10s): ~30-60 seconds
- Medium videos (10-30s): ~1-2 minutes
- Long videos (30-60s): ~2-5 minutes

You'll see a progress indicator while processing.

### 4. View and Download Results

Once complete, you can:

**View Results:**
- **Composite view**: Original video with colored overlay on segmented objects
- **Mask view**: Black and white mask showing segmented regions
- **Original view**: Your uploaded video for comparison

**Download:**
- Download the composite video (with overlay)
- Download the mask video (for further editing)

## Output Formats

### Composite Video
- Format: MP4
- Shows the original video with a semi-transparent green overlay on detected objects
- Great for visualization and presentations

### Mask Video
- Format: MP4 (grayscale)
- White pixels = detected object
- Black pixels = background
- Useful for video editing software (After Effects, Premiere, DaVinci Resolve)

## Troubleshooting

### "No objects found"
- Try a different prompt
- Ensure the object is visible in the first frame
- Use more general terms (e.g., "person" instead of "man in blue shirt")

### Processing takes too long
- Try a shorter video
- Reduce video resolution before uploading
- Check your internet connection

### Video won't upload
- Verify file format (MP4, WebM, MOV)
- Check file size (< 100MB)
- Try a different browser

### Results look incorrect
- The model works best with clearly visible objects
- Fast-moving or heavily occluded objects may have gaps
- Try a more specific or different prompt

## Best Practices

1. **Start with clear videos**: Good lighting and minimal motion blur help
2. **Object should be visible in frame 1**: SAM 3 identifies objects in the first frame
3. **One prompt at a time**: For multiple objects, process separately
4. **Test with short clips first**: Before processing long videos

## Privacy & Data

- Uploaded videos are stored temporarily for processing
- Results are stored for 24 hours for download
- All data is automatically deleted after 24 hours
- We do not use your videos for training or any other purpose

## Support

Having issues? Check our [Troubleshooting Guide](./TROUBLESHOOTING.md) or open an issue on GitHub.

# SAM3 Video Segmentation - Troubleshooting Guide

## Common Issues

### Upload Issues

#### "File type not supported"
**Cause:** The uploaded file is not a supported video format.

**Solution:**
- Convert your video to MP4 using FFmpeg:
  ```bash
  ffmpeg -i input.avi -c:v libx264 -c:a aac output.mp4
  ```
- Supported formats: MP4, WebM, MOV, AVI, MKV

#### "File too large"
**Cause:** Video exceeds the 100MB limit.

**Solution:**
- Compress your video:
  ```bash
  ffmpeg -i input.mp4 -vcodec libx264 -crf 28 output.mp4
  ```
- Reduce resolution:
  ```bash
  ffmpeg -i input.mp4 -vf scale=1280:720 output.mp4
  ```
- Trim to a shorter duration

#### "Video duration exceeds limit"
**Cause:** Video is longer than 60 seconds.

**Solution:**
- Trim your video:
  ```bash
  ffmpeg -i input.mp4 -ss 00:00:00 -t 00:01:00 -c copy output.mp4
  ```

---

### Processing Issues

#### "No objects matching prompt found"
**Cause:** SAM 3 couldn't identify the described object in the first frame.

**Solutions:**
1. **Check the first frame**: The object must be visible at the start
2. **Try different prompts**:
   - More general: "person" instead of "woman in red dress"
   - More specific: "red car" instead of "vehicle"
3. **Ensure good visibility**: Object should be clearly visible, not occluded

#### "Processing stuck at 0%"
**Cause:** GPU worker may be starting up or experiencing issues.

**Solutions:**
1. Wait 2-3 minutes (cold start can take time)
2. Check `/api/health/worker` endpoint
3. If persists, cancel and retry

#### "Processing failed"
**Cause:** Various - check the error message.

**Common causes:**
- Video codec not supported
- Corrupted video file
- GPU out of memory (very high resolution)

**Solutions:**
1. Re-encode video with standard codec:
   ```bash
   ffmpeg -i input.mp4 -c:v libx264 -c:a aac -strict experimental output.mp4
   ```
2. Reduce resolution to 1080p or 720p
3. Try a shorter clip

---

### Result Quality Issues

#### "Mask has gaps or holes"
**Cause:** Object tracking lost the target temporarily.

**Solutions:**
- Use videos with smoother motion
- Avoid videos where object goes fully off-screen
- Try a more specific prompt

#### "Wrong object segmented"
**Cause:** Multiple similar objects in frame.

**Solutions:**
- Use more specific prompts ("person on left", "red car")
- Crop video to focus on target object
- Process when target is most prominent in frame 1

#### "Mask edges are rough"
**Cause:** Normal for fast-moving objects or low resolution.

**Solutions:**
- Use higher resolution source video
- Apply post-processing smoothing in video editor

---

### API Issues

#### "429 Too Many Requests"
**Cause:** Rate limit exceeded.

**Solution:**
- Wait for the time specified in `Retry-After` header
- Implement exponential backoff in your client
- Contact us for higher limits if needed

#### "500 Internal Server Error"
**Cause:** Server-side issue.

**Solutions:**
1. Retry after a few seconds
2. Check `/api/health` endpoint
3. If persists, report the `error_id` from the response

#### "Connection timeout"
**Cause:** Network issues or server overload.

**Solutions:**
1. Check your internet connection
2. Retry with longer timeout
3. Try during off-peak hours

---

## Debugging Checklist

### Before Reporting an Issue

1. **Check service status**
   ```bash
   curl http://localhost:8000/api/health
   curl http://localhost:8000/api/health/worker
   ```

2. **Verify video file**
   ```bash
   ffprobe -v error -show_format -show_streams input.mp4
   ```

3. **Test with sample video**
   - Try a simple, short video first
   - Use a clear, well-lit scene

4. **Check browser console**
   - Open Developer Tools (F12)
   - Look for network errors or JavaScript errors

### Information to Include in Bug Reports

- Video details (format, resolution, duration, file size)
- Prompt used
- Job ID (if available)
- Error message (exact text)
- Browser and version
- Steps to reproduce

---

## Performance Tips

### For Faster Processing

1. **Optimize video before upload**
   - Resolution: 720p is often sufficient
   - Frame rate: 24-30 fps is ideal
   - Codec: H.264 processes fastest

2. **Keep videos short**
   - Process only the relevant portion
   - Trim unnecessary intro/outro

3. **Use specific prompts**
   - Specific prompts can be faster to process
   - Avoid very generic terms

### For Better Quality

1. **Use high-quality source**
   - Good lighting
   - Minimal motion blur
   - Clear object boundaries

2. **Ensure object visibility**
   - Object should be clearly visible in frame 1
   - Avoid heavy occlusion

---

## Getting Help

### Self-Service Resources
- [User Guide](./USER_GUIDE.md)
- [API Documentation](./API_DOCUMENTATION.md)
- [GitHub Issues](https://github.com/Virgelsnake/sam3_test/issues)

### Contact Support
- Open a GitHub issue with the bug report template
- Include all relevant debugging information

### Community
- Check existing GitHub issues for similar problems
- Contribute fixes via pull requests

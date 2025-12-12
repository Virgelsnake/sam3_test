"""Video upload endpoints."""

import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from ..config import get_settings
from ..services.storage import storage_service

router = APIRouter(prefix="/api/uploads", tags=["uploads"])


@router.post("")
async def upload_video(file: UploadFile = File(...)) -> dict:
    """
    Upload a video file to storage.

    Args:
        file: The video file to upload.

    Returns:
        dict: Contains video_id and signed URL for the uploaded file.

    Raises:
        HTTPException: If file validation fails or upload errors occur.
    """
    settings = get_settings()

    # Validate file extension
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required",
        )

    ext = Path(file.filename).suffix.lower()
    if ext not in settings.allowed_video_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {settings.allowed_video_extensions}",
        )

    # Validate file size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)

    if size_mb > settings.max_video_size_mb:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Maximum size: {settings.max_video_size_mb}MB",
        )

    # Generate unique video ID
    video_id = str(uuid.uuid4())
    storage_path = f"{video_id}{ext}"

    try:
        # Upload to storage
        url = await storage_service.upload_video(
            content=content,
            path=storage_path,
            content_type=file.content_type or "video/mp4",
        )

        return {
            "video_id": video_id,
            "filename": file.filename,
            "size_mb": round(size_mb, 2),
            "url": url,
            "path": storage_path,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload video: {str(e)}",
        )

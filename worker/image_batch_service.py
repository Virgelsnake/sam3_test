"""
Image Batch Service - Static image inventory processing with cross-image deduplication.

Processes multiple static images to generate an inventory, using SAM3 for segmentation
and GPT-4V for object identification. Implements cross-image duplicate detection
similar to the video frame-to-frame logic.
"""

import os
from typing import Optional
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

from classification_service import ClassificationService


# Similarity threshold for cross-image duplicate detection
DUPLICATE_SIMILARITY_THRESHOLD = 0.85

# Colors for different object categories (hex format)
CATEGORY_COLORS = [
    "#22c55e",  # Green
    "#3b82f6",  # Blue
    "#ef4444",  # Red
    "#eab308",  # Yellow
    "#a855f7",  # Purple
    "#06b6d4",  # Cyan
    "#f97316",  # Orange
    "#ec4899",  # Pink
    "#14b8a6",  # Teal
    "#8b5cf6",  # Violet
    "#84cc16",  # Lime
    "#f43f5e",  # Rose
]


@dataclass
class DetectedObject:
    """Represents a detected object across images."""
    obj_id: int
    category: str
    source_image_idx: int
    mask: np.ndarray
    bbox: tuple  # (x, y, w, h)
    embedding: Optional[np.ndarray] = None  # For similarity comparison
    confidence: float = 1.0
    merged_count: int = 1  # How many duplicates were merged into this


@dataclass
class ImageResult:
    """Processing result for a single image."""
    image_idx: int
    objects: list[DetectedObject]
    composite_image: Optional[np.ndarray] = None


class ImageBatchService:
    """Service for processing image batches with cross-image deduplication."""

    def __init__(self, device: str = "cuda:0"):
        """Initialize the image batch service."""
        self.device = device
        self.predictor = None
        self._initialized = False
        self.classifier = ClassificationService()
        self.next_obj_id = 0

    def initialize(self) -> None:
        """Initialize SAM3 model for image segmentation."""
        if self._initialized:
            return

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)

        from sam3.model_builder import build_sam3_video_predictor
        
        gpu_idx = 0
        if ":" in self.device:
            gpu_idx = int(self.device.split(":")[1])

        self.predictor = build_sam3_video_predictor(gpus_to_use=[gpu_idx])
        self._initialized = True

    def process_image_batch(
        self,
        images: list[np.ndarray],
        prompt: str,
        progress_callback: Optional[callable] = None,
    ) -> dict:
        """
        Process a batch of images for inventory generation.

        NEW APPROACH - GPT-4V Multi-Image Unified Inventory:
        1. Send ALL images to GPT-4V at once for unified inventory
        2. GPT-4V understands they're the same space and deduplicates automatically
        3. Use identified items to guide SAM3 segmentation for visualization
        4. Return accurate, deduplicated inventory

        Args:
            images: List of RGB numpy arrays.
            prompt: Context prompt for inventory.
            progress_callback: Optional callback for progress updates.

        Returns:
            dict: Results containing inventory, per-image results, etc.
        """
        if not self._initialized:
            self.initialize()

        if not images:
            return {
                "inventory": {},
                "objects_detected": 0,
                "per_image_results": [],
                "message": "No images provided",
            }

        total_images = len(images)

        # ========== PHASE 1: Multi-Image GPT-4V Inventory ==========
        # Send ALL images to GPT-4V at once - it understands same-room context
        if progress_callback:
            progress_callback(10, "Analyzing all images for unified inventory...")

        print(f"[ImageBatch] Sending {total_images} images to GPT-4V for unified inventory...")
        
        self.classifier.initialize()
        inventory_result = self.classifier.generate_multi_image_inventory(images, context=prompt)
        
        inventory = inventory_result.get("inventory", {})
        items = inventory_result.get("items", [])
        scene_description = inventory_result.get("scene_description", "")
        
        print(f"[ImageBatch] GPT-4V unified inventory: {inventory}")
        print(f"[ImageBatch] Scene: {scene_description}")

        if not inventory:
            return {
                "inventory": {},
                "objects_detected": 0,
                "per_image_results": [],
                "composite_images": [img.copy() for img in images],
                "message": "No objects detected",
            }

        if progress_callback:
            progress_callback(40, "Creating visualizations...")

        # ========== PHASE 2: Assign Colors to Categories ==========
        category_colors = {}
        for i, category in enumerate(sorted(inventory.keys())):
            category_colors[category] = CATEGORY_COLORS[i % len(CATEGORY_COLORS)]

        # ========== PHASE 3: Create Per-Image Results ==========
        per_image_results = []
        for img_idx in range(total_images):
            # Find which items appear in this image
            items_in_image = [
                item for item in items 
                if (img_idx + 1) in item.get("appears_in_images", [])
            ]
            per_image_results.append({
                "image_idx": img_idx,
                "objects_found": len(items_in_image),
                "categories": [item.get("name", "") for item in items_in_image],
            })

        # ========== PHASE 4: Optional SAM3 Visualization ==========
        # Try to segment detected items for visual overlay
        composite_images = []
        
        for img_idx, image in enumerate(images):
            if progress_callback:
                progress = 40 + int((img_idx / total_images) * 50)
                progress_callback(progress, f"Segmenting image {img_idx + 1}/{total_images}...")

            # Get items that appear in this image
            items_in_image = [
                item.get("name", "") for item in items 
                if (img_idx + 1) in item.get("appears_in_images", [])
            ]
            
            if items_in_image:
                # Try SAM3 segmentation for visualization
                composite, bboxes = self._segment_and_overlay(image, items_in_image, category_colors)
                # Store bboxes in per_image_results
                per_image_results[img_idx]["item_bboxes"] = bboxes
            else:
                composite = image.copy()
                per_image_results[img_idx]["item_bboxes"] = []
            
            composite_images.append(composite)

        if progress_callback:
            progress_callback(95, "Finalizing results...")

        total_objects = sum(inventory.values())
        print(f"[ImageBatch] Complete. {len(inventory)} categories, {total_objects} total items")

        return {
            "inventory": inventory,
            "inventory_colors": category_colors,
            "objects_detected": total_objects,
            "per_image_results": per_image_results,
            "composite_images": composite_images,
            "scene_description": scene_description,
            "items_detail": items,
        }

    def _segment_and_overlay(
        self,
        image: np.ndarray,
        categories: list[str],
        category_colors: dict,
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Segment specific categories in an image and create colored overlay.
        
        This is for VISUALIZATION only - inventory is already determined by GPT-4V.
        
        Returns:
            tuple: (composite_image, item_bboxes)
                - composite_image: Image with mask overlays
                - item_bboxes: List of {category, bbox: {x, y, width, height}, color}
        """
        item_bboxes = []
        import tempfile
        import cv2
        
        composite = image.copy().astype(np.float32)
        
        # Create temporary single-frame video for SAM3
        temp_video_path = tempfile.mktemp(suffix=".mp4")
        try:
            height, width = image.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, 1, (width, height))
            out.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            out.release()

            for category in categories:
                try:
                    session_id = self._create_session(temp_video_path)
                    
                    response = self.predictor.handle_request({
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": 0,
                        "text": category
                    })

                    outputs = response.get("outputs", {})
                    binary_masks = outputs.get("out_binary_masks")
                    
                    # DIAGNOSTIC: Log what SAM3 returns for each category
                    print(f"[ImageBatch] SAM3 response for '{category}':")
                    print(f"  - outputs keys: {list(outputs.keys())}")
                    print(f"  - binary_masks type: {type(binary_masks)}")
                    if binary_masks is not None:
                        print(f"  - binary_masks shape: {binary_masks.shape if hasattr(binary_masks, 'shape') else 'N/A'}")

                    if binary_masks is not None:
                        if hasattr(binary_masks, 'cpu'):
                            binary_masks = binary_masks.cpu().numpy()
                        
                        # DIAGNOSTIC: Log mask details
                        print(f"  - numpy masks shape: {binary_masks.shape}")
                        print(f"  - num masks: {len(binary_masks)}")

                        # Get color for this category
                        hex_color = category_colors.get(category.lower(), "#22c55e")
                        rgb_color = self._hex_to_rgb(hex_color)

                        for mask_idx, mask in enumerate(binary_masks):
                            if mask.sum() > 100:
                                # Extract bounding box from mask
                                mask_2d = mask.squeeze() if mask.ndim > 2 else mask
                                rows = np.any(mask_2d > 0.5, axis=1)
                                cols = np.any(mask_2d > 0.5, axis=0)
                                
                                if rows.any() and cols.any():
                                    y_min, y_max = np.where(rows)[0][[0, -1]]
                                    x_min, x_max = np.where(cols)[0][[0, -1]]
                                    
                                    # Scale bbox to image dimensions if mask was resized
                                    img_height, img_width = image.shape[:2]
                                    mask_height, mask_width = mask_2d.shape[:2]
                                    
                                    # Calculate scaled bbox coordinates
                                    scale_x = img_width / mask_width
                                    scale_y = img_height / mask_height
                                    
                                    bbox = {
                                        "x": int(x_min * scale_x),
                                        "y": int(y_min * scale_y),
                                        "width": int((x_max - x_min + 1) * scale_x),
                                        "height": int((y_max - y_min + 1) * scale_y),
                                    }
                                    
                                    # Store bbox for this category
                                    item_bboxes.append({
                                        "category": category,
                                        "bbox": bbox,
                                        "color": hex_color,
                                    })
                                    
                                    print(f"  - mask[{mask_idx}] bbox: {bbox}")
                                
                                # Apply overlay
                                if mask.ndim > 2:
                                    mask = mask.squeeze()
                                
                                if mask.shape[:2] != image.shape[:2]:
                                    from PIL import Image as PILImage
                                    mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))
                                    mask_pil = mask_pil.resize((image.shape[1], image.shape[0]), PILImage.NEAREST)
                                    mask = np.array(mask_pil) / 255.0

                                overlay = np.zeros_like(composite)
                                overlay[mask > 0.5] = rgb_color

                                alpha = 0.4
                                mask_3d = np.stack([mask] * 3, axis=-1)
                                composite = np.where(
                                    mask_3d > 0.5,
                                    composite * (1 - alpha) + overlay * alpha,
                                    composite
                                )

                    self._close_session(session_id)

                except Exception as e:
                    print(f"[ImageBatch] Error segmenting '{category}': {e}")
                    continue

        except Exception as e:
            print(f"[ImageBatch] SAM3 visualization failed: {e}")
        finally:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)

        return composite.astype(np.uint8), item_bboxes

    def _process_single_image(
        self,
        image: np.ndarray,
        image_idx: int,
        prompt: str,
    ) -> list[DetectedObject]:
        """Process a single image to detect and segment objects."""
        detected_objects = []

        # Step 1: Use GPT-4V to identify objects
        try:
            self.classifier.initialize()
            categories = self.classifier.identify_objects_in_frame(image, context=prompt)
            print(f"[ImageBatch] Image {image_idx}: GPT-4V found categories: {categories}")
        except Exception as e:
            print(f"[ImageBatch] Image {image_idx}: GPT-4V failed: {e}")
            categories = []

        if not categories:
            return detected_objects

        # Step 2: Create a temporary video-like structure for SAM3
        # SAM3 expects video input, so we create a single-frame "video"
        import tempfile
        import cv2
        
        temp_video_path = tempfile.mktemp(suffix=".mp4")
        try:
            # Create a single-frame video
            height, width = image.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, 1, (width, height))
            out.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            out.release()

            # Step 3: Run SAM3 segmentation for each category
            for category in categories:
                try:
                    session_id = self._create_session(temp_video_path)
                    
                    response = self.predictor.handle_request({
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": 0,
                        "text": category
                    })

                    outputs = response.get("outputs", {})
                    obj_ids = outputs.get("out_obj_ids", [])
                    binary_masks = outputs.get("out_binary_masks")

                    if isinstance(obj_ids, np.ndarray):
                        obj_ids = obj_ids.tolist()

                    if binary_masks is not None:
                        if hasattr(binary_masks, 'cpu'):
                            binary_masks = binary_masks.cpu().numpy()

                        for i, mask in enumerate(binary_masks):
                            if mask.sum() > 100:  # Min mask size
                                # Extract bounding box
                                bbox = self._mask_to_bbox(mask)
                                
                                # Extract embedding for deduplication
                                embedding = self._extract_embedding(image, mask)

                                obj = DetectedObject(
                                    obj_id=self.next_obj_id,
                                    category=category,
                                    source_image_idx=image_idx,
                                    mask=mask.astype(np.float32),
                                    bbox=bbox,
                                    embedding=embedding,
                                )
                                detected_objects.append(obj)
                                self.next_obj_id += 1

                    self._close_session(session_id)

                except Exception as e:
                    print(f"[ImageBatch] Error segmenting '{category}': {e}")
                    continue

        finally:
            # Cleanup temp video
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)

        return detected_objects

    def _deduplicate_objects(
        self,
        objects: list[DetectedObject],
    ) -> list[DetectedObject]:
        """
        Deduplicate objects across images using visual similarity.

        Objects appearing in multiple images (from different angles) should
        only be counted once. Uses embedding similarity to detect duplicates.
        """
        if len(objects) <= 1:
            return objects

        unique_objects = []
        used_indices = set()

        # Sort by image index to prioritize earlier images
        sorted_objects = sorted(objects, key=lambda x: (x.source_image_idx, x.obj_id))

        for i, obj in enumerate(sorted_objects):
            if i in used_indices:
                continue

            # Check if this object matches any already-added unique object
            is_duplicate = False
            for unique_obj in unique_objects:
                # Only compare objects of the same category
                if obj.category.lower() != unique_obj.category.lower():
                    continue

                # Compare embeddings
                similarity = self._compute_similarity(obj.embedding, unique_obj.embedding)
                
                if similarity >= DUPLICATE_SIMILARITY_THRESHOLD:
                    # This is a duplicate - merge into existing
                    unique_obj.merged_count += 1
                    is_duplicate = True
                    print(f"[ImageBatch] Merged duplicate '{obj.category}' (sim={similarity:.2f})")
                    break

            if not is_duplicate:
                unique_objects.append(obj)
                used_indices.add(i)

        return unique_objects

    def _extract_embedding(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Extract a feature embedding for an object region.

        Uses color histogram and spatial features for comparison.
        """
        # Ensure mask is 2D
        if mask.ndim > 2:
            mask = mask.squeeze()

        # Resize mask if needed
        if mask.shape[:2] != image.shape[:2]:
            from PIL import Image as PILImage
            mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))
            mask_pil = mask_pil.resize((image.shape[1], image.shape[0]), PILImage.NEAREST)
            mask = np.array(mask_pil) / 255.0

        # Extract masked region
        mask_bool = mask > 0.5
        
        if not mask_bool.any():
            return np.zeros(64)

        # Color histogram features
        features = []
        for channel in range(3):
            channel_data = image[:, :, channel][mask_bool]
            hist, _ = np.histogram(channel_data, bins=16, range=(0, 255))
            hist = hist.astype(np.float32) / (hist.sum() + 1e-6)
            features.extend(hist)

        # Add shape features (aspect ratio, relative size)
        bbox = self._mask_to_bbox(mask)
        if bbox[2] > 0 and bbox[3] > 0:
            aspect_ratio = bbox[2] / bbox[3]
            relative_size = mask_bool.sum() / (image.shape[0] * image.shape[1])
            features.extend([aspect_ratio, relative_size])
        else:
            features.extend([1.0, 0.0])

        # Centroid position (normalized)
        y_coords, x_coords = np.where(mask_bool)
        if len(x_coords) > 0:
            centroid_x = np.mean(x_coords) / image.shape[1]
            centroid_y = np.mean(y_coords) / image.shape[0]
            features.extend([centroid_x, centroid_y])
        else:
            features.extend([0.5, 0.5])

        return np.array(features[:64], dtype=np.float32)

    def _compute_similarity(
        self,
        emb1: Optional[np.ndarray],
        emb2: Optional[np.ndarray],
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        if emb1 is None or emb2 is None:
            return 0.0

        # Pad to same length if needed
        max_len = max(len(emb1), len(emb2))
        if len(emb1) < max_len:
            emb1 = np.pad(emb1, (0, max_len - len(emb1)))
        if len(emb2) < max_len:
            emb2 = np.pad(emb2, (0, max_len - len(emb2)))

        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0

        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def _mask_to_bbox(self, mask: np.ndarray) -> tuple:
        """Convert mask to bounding box (x, y, w, h)."""
        if mask.ndim > 2:
            mask = mask.squeeze()

        rows = np.any(mask > 0.5, axis=1)
        cols = np.any(mask > 0.5, axis=0)

        if not rows.any() or not cols.any():
            return (0, 0, 0, 0)

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        return (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

    def _create_composite_image(
        self,
        image: np.ndarray,
        objects: list[DetectedObject],
        category_colors: dict,
    ) -> np.ndarray:
        """Create composite image with colored mask overlays."""
        composite = image.copy().astype(np.float32)

        for obj in objects:
            # Get color for this category
            hex_color = category_colors.get(obj.category.lower(), "#22c55e")
            rgb_color = self._hex_to_rgb(hex_color)

            mask = obj.mask
            if mask.ndim > 2:
                mask = mask.squeeze()

            # Resize mask if needed
            if mask.shape[:2] != image.shape[:2]:
                from PIL import Image as PILImage
                mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))
                mask_pil = mask_pil.resize((image.shape[1], image.shape[0]), PILImage.NEAREST)
                mask = np.array(mask_pil) / 255.0

            # Create overlay
            overlay = np.zeros_like(composite)
            overlay[mask > 0.5] = rgb_color

            # Blend
            alpha = 0.4
            mask_3d = np.stack([mask] * 3, axis=-1)
            composite = np.where(
                mask_3d > 0.5,
                composite * (1 - alpha) + overlay * alpha,
                composite
            )

        return composite.astype(np.uint8)

    def _hex_to_rgb(self, hex_color: str) -> tuple:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def _create_session(self, video_path: str) -> str:
        """Create a SAM3 session."""
        response = self.predictor.handle_request({
            "type": "start_session",
            "resource_path": video_path
        })
        return response["session_id"]

    def _close_session(self, session_id: str) -> None:
        """Close a SAM3 session."""
        if self.predictor is not None:
            try:
                self.predictor.handle_request({
                    "type": "close_session",
                    "session_id": session_id
                })
            except Exception:
                pass

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

"""
Classification Service - OpenAI GPT-4V for object identification.

Stage 2 of the pipeline: Analyzes segmented regions to identify objects
and compile an inventory list. Supports individual object classification
and dynamic inventory tracking across frames.
"""

import base64
import io
import os
from typing import Optional

import numpy as np
from PIL import Image


class ClassificationService:
    """Service for classifying segmented objects using OpenAI GPT-4V."""

    def __init__(self):
        """Initialize the classification service."""
        self.client = None
        self._initialized = False

    def initialize(self) -> None:
        """
        Initialize the OpenAI client.
        
        Requires OPENAI_API_KEY environment variable.
        """
        if self._initialized:
            return

        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=api_key)
        self._initialized = True

    def identify_objects_in_frame(
        self,
        frame: np.ndarray,
        context: str = "inventory scan",
    ) -> list[str]:
        """
        Use GPT-4V to identify ALL objects visible in a raw frame (no masks).
        
        This is the open-vocabulary identification step that runs BEFORE SAM3 detection.
        Returns a list of object labels that can be used as prompts for SAM3.

        Args:
            frame: Original video frame (RGB numpy array).
            context: Context hint for identification.

        Returns:
            list[str]: List of identified object labels suitable for SAM3 prompts.
        """
        if not self._initialized:
            self.initialize()

        # Encode frame to base64
        image_b64 = self._encode_image(frame)

        prompt = f"""Analyze this image and identify ALL distinct physical objects visible.

Context: {context}

Instructions:
1. List every distinct physical object you can see (furniture, electronics, items on surfaces, etc.)
2. Use simple, specific labels (e.g., "office chair", "computer monitor", "keyboard", "coffee mug")
3. Include count if multiple of same item (e.g., if 3 monitors, list "monitor" once)
4. Focus on objects that could be segmented/highlighted - not background elements like walls/floor
5. Be thorough - don't miss smaller items like mice, phones, cups, plants

Respond in this exact JSON format:
{{
    "objects": ["object1", "object2", "object3"],
    "scene_description": "Brief description of the scene"
}}

List objects as simple noun phrases suitable for object detection."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,
                response_format={"type": "json_object"}
            )

            result_text = response.choices[0].message.content
            if result_text is None:
                print("[Classification] GPT-4V returned empty response - check OPENAI_API_KEY")
                return []
            
            import json
            result = json.loads(result_text)
            
            objects = result.get("objects", [])
            print(f"[Classification] GPT-4V identified {len(objects)} object types: {objects}")
            
            return objects

        except Exception as e:
            print(f"[Classification] Error in open-vocabulary identification: {e}")
            # Return empty list but don't crash the pipeline
            return []

    def classify_frame_objects(
        self,
        frame: np.ndarray,
        masks: dict[int, np.ndarray],
        context: str = "office environment",
    ) -> dict:
        """
        Classify objects in a single frame using their masks.

        Args:
            frame: Original video frame (RGB numpy array).
            masks: Dictionary mapping object ID to binary mask.
            context: Context hint for classification.

        Returns:
            dict: Classification results with object IDs and labels.
        """
        if not self._initialized:
            self.initialize()

        if not masks:
            return {"objects": [], "inventory": {}}

        # Create a composite image showing the frame with mask overlays
        composite = self._create_composite_image(frame, masks)
        
        # Encode image to base64
        image_b64 = self._encode_image(composite)

        # Build the prompt
        prompt = f"""Analyze this image from an {context}. The image shows segmented objects with colored overlays.

Each colored region represents a detected object. Please:
1. Identify what each colored region/object is
2. Be specific (e.g., "office chair" not just "chair", "computer monitor" not just "screen")
3. If you cannot identify an object, label it as "unidentified object"

Respond in this exact JSON format:
{{
    "objects": [
        {{"color": "green", "label": "object name", "confidence": "high/medium/low"}},
        {{"color": "blue", "label": "object name", "confidence": "high/medium/low"}}
    ],
    "summary": "Brief description of the scene"
}}

Only include objects you can see highlighted in the image."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                response_format={"type": "json_object"}
            )

            result_text = response.choices[0].message.content
            import json
            result = json.loads(result_text)
            
            return result

        except Exception as e:
            print(f"[Classification] Error calling OpenAI: {e}")
            return {"objects": [], "inventory": {}, "error": str(e)}

    def classify_video_sample(
        self,
        frames: list[np.ndarray],
        all_masks: dict[int, np.ndarray],
        sample_frames: list[int] = None,
        context: str = "office environment",
        individual_masks: dict[int, dict[int, np.ndarray]] = None,
        tracked_objects: dict = None,
    ) -> dict:
        """
        Classify objects from sampled video frames.

        Args:
            frames: List of video frames.
            all_masks: Dictionary mapping frame index to combined mask.
            sample_frames: List of frame indices to sample. If None, samples automatically.
            context: Context hint for classification.
            individual_masks: Optional dict[frame_idx][obj_id] -> mask for per-object analysis.
            tracked_objects: Optional tracking data with category hints.

        Returns:
            dict: Aggregated classification results and inventory.
        """
        if not self._initialized:
            self.initialize()

        if not frames or not all_masks:
            return {"inventory": {}, "total_objects": 0}

        # If we have tracking data with categories, use that as primary source
        if tracked_objects:
            inventory = self._build_inventory_from_tracking(tracked_objects)
            print(f"[Classification] Built inventory from tracking: {inventory}")
            
            # Vision verification is disabled - it only sees one frame and misses objects
            # The tracking data is more reliable since it sees all frames
            # TODO: Could use vision to correct category names, but not counts
            
            return {
                "inventory": inventory,
                "total_objects": sum(inventory.values()),
                "source": "tracking",
                "tracked_objects": tracked_objects,
            }

        # Fall back to original classification logic
        if sample_frames is None:
            total_frames = len(frames)
            if total_frames <= 3:
                sample_frames = list(range(total_frames))
            else:
                sample_frames = [
                    0,
                    total_frames // 4,
                    total_frames // 2,
                    3 * total_frames // 4,
                    total_frames - 1
                ]

        valid_sample_frames = [f for f in sample_frames if f in all_masks and f < len(frames)]
        
        if not valid_sample_frames:
            valid_sample_frames = [list(all_masks.keys())[0]] if all_masks else []

        if not valid_sample_frames:
            return {"inventory": {}, "total_objects": 0}

        print(f"[Classification] Analyzing {len(valid_sample_frames)} sample frames...")

        all_objects = []
        for frame_idx in valid_sample_frames:
            if frame_idx >= len(frames):
                continue
                
            frame = frames[frame_idx]
            
            # Use individual masks if available, otherwise use combined
            if individual_masks and frame_idx in individual_masks:
                frame_masks = individual_masks[frame_idx]
            else:
                mask = all_masks.get(frame_idx)
                if mask is None:
                    continue
                frame_masks = {0: mask}
            
            result = self.classify_frame_objects(frame, frame_masks, context)
            
            if "objects" in result:
                for obj in result["objects"]:
                    obj["frame_index"] = frame_idx
                    all_objects.append(obj)

        inventory = self._aggregate_inventory(all_objects)

        return {
            "inventory": inventory,
            "total_objects": len(inventory),
            "sample_frames_analyzed": len(valid_sample_frames),
            "raw_detections": all_objects,
        }

    def _build_inventory_from_tracking(
        self,
        tracked_objects: dict,
    ) -> dict:
        """
        Build inventory directly from tracking data categories.

        Args:
            tracked_objects: Dict with obj_id -> {category, first_frame, last_frame, is_active}

        Returns:
            dict: Inventory mapping category to count.
        """
        category_counts = {}
        
        for obj_id, obj_data in tracked_objects.items():
            category = obj_data.get("category", "unknown")
            # Normalize category name
            category = category.lower().strip()
            
            if category not in category_counts:
                category_counts[category] = 0
            category_counts[category] += 1
        
        return category_counts

    def _verify_inventory_with_vision(
        self,
        frames: list[np.ndarray],
        individual_masks: dict[int, dict[int, np.ndarray]],
        tracked_objects: dict,
        context: str,
    ) -> dict:
        """
        Verify/refine inventory using visual classification on sample frames.

        Analyzes a few frames with GPT-4V to verify or correct the tracked categories.

        Args:
            frames: Video frames.
            individual_masks: Per-frame, per-object masks.
            tracked_objects: Tracking data with initial categories.
            context: Context hint.

        Returns:
            dict: Verified inventory, or None if verification fails.
        """
        try:
            # Pick a frame from the middle of the video
            available_frames = sorted(individual_masks.keys())
            if not available_frames:
                return None
            
            sample_frame_idx = available_frames[len(available_frames) // 2]
            if sample_frame_idx >= len(frames):
                sample_frame_idx = available_frames[0]
            
            frame = frames[sample_frame_idx]
            frame_masks = individual_masks[sample_frame_idx]
            
            # Create composite showing all objects
            composite = self._create_composite_image(frame, frame_masks)
            image_b64 = self._encode_image(composite)
            
            # Build prompt with tracking hints
            obj_hints = []
            for obj_id, mask in frame_masks.items():
                if obj_id in tracked_objects:
                    obj_hints.append(f"Object {obj_id}: possibly {tracked_objects[obj_id].get('category', 'unknown')}")
            
            hints_text = "\n".join(obj_hints) if obj_hints else "No category hints available."
            
            prompt = f"""Analyze this image from an {context}. Multiple objects are highlighted with different colored overlays.

Tracking system detected these objects:
{hints_text}

Please:
1. Verify or correct each object identification
2. Be specific (e.g., "office chair" not just "chair")
3. Count the number of distinct objects of each type

Respond in this exact JSON format:
{{
    "inventory": {{
        "object_type": count,
        "another_type": count
    }},
    "corrections": [
        {{
            "object_id": 0,
            "original": "original category",
            "corrected": "corrected category",
            "reason": "why correction was made"
        }}
    ]
}}"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                response_format={"type": "json_object"}
            )

            import json
            result = json.loads(response.choices[0].message.content)
            
            if "inventory" in result:
                print(f"[Classification] Vision verification complete. Inventory: {result['inventory']}")
                if "corrections" in result and result["corrections"]:
                    print(f"[Classification] Corrections made: {result['corrections']}")
                return result["inventory"]
            
            return None

        except Exception as e:
            print(f"[Classification] Vision verification failed: {e}")
            return None

    def _create_composite_image(
        self,
        frame: np.ndarray,
        masks: dict[int, np.ndarray],
        colors: list[tuple] = None,
    ) -> np.ndarray:
        """
        Create a composite image with mask overlays.

        Args:
            frame: Original frame (RGB).
            masks: Dictionary of masks.
            colors: Optional list of RGB colors for overlays.

        Returns:
            np.ndarray: Composite image with overlays.
        """
        if colors is None:
            colors = [
                (0, 255, 0),    # Green
                (0, 0, 255),    # Blue
                (255, 0, 0),    # Red
                (255, 255, 0),  # Yellow
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Cyan
            ]

        composite = frame.copy().astype(np.float32)
        
        for i, (obj_id, mask) in enumerate(masks.items()):
            color = colors[i % len(colors)]
            
            # Ensure mask is 2D
            if mask.ndim > 2:
                mask = mask.squeeze()
            
            # Resize mask if needed
            if mask.shape[:2] != frame.shape[:2]:
                from PIL import Image as PILImage
                mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))
                mask_pil = mask_pil.resize((frame.shape[1], frame.shape[0]), PILImage.NEAREST)
                mask = np.array(mask_pil) / 255.0

            # Create colored overlay
            overlay = np.zeros_like(composite)
            overlay[mask > 0.5] = color

            # Blend with original
            alpha = 0.4
            mask_3d = np.stack([mask] * 3, axis=-1)
            composite = np.where(
                mask_3d > 0.5,
                composite * (1 - alpha) + overlay * alpha,
                composite
            )

        return composite.astype(np.uint8)

    def _encode_image(self, image: np.ndarray) -> str:
        """
        Encode image to base64 JPEG string.

        Args:
            image: RGB numpy array.

        Returns:
            str: Base64 encoded JPEG.
        """
        pil_image = Image.fromarray(image)
        
        # Resize if too large (max 2048px on longest side for API efficiency)
        max_size = 2048
        if max(pil_image.size) > max_size:
            ratio = max_size / max(pil_image.size)
            new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
            pil_image = pil_image.resize(new_size, Image.LANCZOS)

        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _aggregate_inventory(self, all_objects: list[dict]) -> dict:
        """
        Aggregate detected objects into an inventory count.

        Args:
            all_objects: List of detected objects from all frames.

        Returns:
            dict: Inventory mapping object label to count.
        """
        # Count unique objects (deduplicate across frames)
        label_counts = {}
        
        for obj in all_objects:
            label = obj.get("label", "unknown").lower().strip()
            confidence = obj.get("confidence", "medium")
            
            # Skip low confidence detections
            if confidence == "low":
                continue
            
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1

        # Since same object appears in multiple frames, estimate unique count
        # by taking the max count across frames (rough heuristic)
        # For better accuracy, would need object tracking
        
        # For now, just report what was detected
        inventory = {}
        for label, count in label_counts.items():
            # Normalize count - if seen in multiple frames, likely 1 object
            # This is a simplification; real tracking would be more accurate
            estimated_count = 1 if count <= 3 else (count // 3)
            inventory[label] = max(1, estimated_count)

        return inventory

    def generate_multi_image_inventory(
        self,
        images: list[np.ndarray],
        context: str = "room inventory",
    ) -> dict:
        """
        Generate inventory from multiple images of the same space using GPT-4V.
        
        This is the correct approach for multi-image inventory:
        - Send ALL images to GPT-4V at once
        - GPT-4V understands they're different views of the same space
        - Returns deduplicated inventory (same object in multiple images = 1 count)
        
        Args:
            images: List of RGB numpy arrays (different views of same space).
            context: Context hint for inventory.
            
        Returns:
            dict: {
                "inventory": {"item": count, ...},
                "items": [{"name": str, "count": int, "appears_in_images": [int, ...]}, ...],
                "scene_description": str
            }
        """
        if not self._initialized:
            self.initialize()

        if not images:
            return {"inventory": {}, "items": [], "scene_description": "No images provided"}

        # Encode all images
        image_contents = []
        for i, img in enumerate(images):
            image_b64 = self._encode_image(img)
            image_contents.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}",
                    "detail": "high"
                }
            })

        prompt = f"""You are a professional inventory specialist analyzing {len(images)} photographs of the SAME room/space taken from different angles.

Context: {context}

YOUR TASK: Create a comprehensive, categorized inventory of EVERY physical item visible across all images.

CRITICAL RULES:
1. These images show the SAME physical space from different viewpoints
2. The SAME physical object appearing in multiple images = count it ONCE
3. Be EXHAUSTIVE - capture every single item, no matter how small
4. Use DESCRIPTIVE labels (e.g., "Height-adjustable desk (wood top, white legs)" not just "desk")
5. Group items into logical categories

SCAN SYSTEMATICALLY - Look for:
- FURNITURE: Desks, chairs, tables, shelving units, cabinets, drawers
- ELECTRONICS: Monitors, laptops, keyboards, mice, webcams, speakers, cables
- OFFICE SUPPLIES: Pens, notebooks, papers, folders, staplers, tape
- STORAGE: Boxes, bags, backpacks, containers, file organizers
- DÉCOR: Plants, artwork, photos, decorations, ornaments
- LIGHTING: Lamps, light fixtures
- FLOORING: Rugs, mats
- FIXTURES: Radiators, blinds, windows
- MISCELLANEOUS: Water bottles, mugs, toys, cables on floor, anything else

For quantities of similar small items (cables, papers), use "multiple" or "several" or estimate.

Respond in this exact JSON format:
{{
    "scene_description": "Detailed description of the room/space",
    "categories": {{
        "Furniture & Fixtures": [
            {{"name": "Height-adjustable desk (wood top, white legs)", "count": 1, "appears_in_images": [2, 3, 4]}},
            {{"name": "Office chair (wheeled, adjustable)", "count": 1, "appears_in_images": [2, 3, 4]}}
        ],
        "Electronics & Computing": [
            {{"name": "External monitor", "count": 2, "appears_in_images": [2, 3, 4]}},
            {{"name": "Laptop", "count": 1, "appears_in_images": [2, 3, 4]}}
        ],
        "Music Equipment": [
            {{"name": "Digital keyboard/piano", "count": 1, "appears_in_images": [1, 2]}}
        ]
    }},
    "items": [
        {{"name": "Height-adjustable desk (wood top, white legs)", "count": 1, "category": "Furniture & Fixtures", "appears_in_images": [2, 3, 4]}}
    ]
}}

BE THOROUGH. Examine every corner, surface, floor, and shelf. Miss nothing."""

        try:
            # Build message with all images
            content = [{"type": "text", "text": prompt}] + image_contents

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": content}],
                max_tokens=4096,
                response_format={"type": "json_object"}
            )

            result_text = response.choices[0].message.content
            if result_text is None:
                print("[Classification] GPT-4V returned empty response")
                return {"inventory": {}, "items": [], "scene_description": "Error: empty response"}

            import json
            result = json.loads(result_text)

            # Build inventory dict from items list OR categories
            inventory = {}
            items = result.get("items", [])
            categories = result.get("categories", {})
            
            # If categories provided, flatten them into items list
            if categories and not items:
                items = []
                for category_name, category_items in categories.items():
                    for item in category_items:
                        item["category"] = category_name
                        items.append(item)
            
            # Build inventory from items
            for item in items:
                name = item.get("name", "unknown").strip()
                count = item.get("count", 1)
                # Handle string counts like "multiple" or "several"
                if isinstance(count, str):
                    if count.lower() in ["multiple", "several", "many"]:
                        count = 3
                    else:
                        try:
                            count = int(count)
                        except:
                            count = 1
                inventory[name.lower()] = count

            print(f"[Classification] Multi-image inventory: {len(items)} unique items")
            print(f"[Classification] Categories: {list(categories.keys()) if categories else 'none'}")

            return {
                "inventory": inventory,
                "items": items,
                "categories": categories,
                "scene_description": result.get("scene_description", ""),
                "total_unique_items": len(items),
                "total_item_count": sum(inventory.values()),
            }

        except Exception as e:
            print(f"[Classification] Error in multi-image inventory: {e}")
            return {"inventory": {}, "items": [], "scene_description": f"Error: {str(e)}"}

# Copyright 2025 - PAPO
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from PIL import Image
from typing import List, Union


def random_patch_blackening(pil_img, patch_size=14, black_prob=0.6):
    """
    Randomly blacken square patches in a PIL image.
    Used for image data augmentation in PAPO.
    """
    img = np.array(pil_img).astype(np.float32)
    h, w = img.shape[:2]
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            if np.random.rand() < black_prob:
                y_end = min(y + patch_size, h)
                x_end = min(x + patch_size, w)
                if img.ndim == 3:
                    img[y:y_end, x:x_end, :] = 0
                else:
                    img[y:y_end, x:x_end] = 0
    return Image.fromarray(img.astype(np.uint8))


def random_patch_blackening_video(
    frames: List[Image.Image],
    patch_size: int = 14,
    black_prob: float = 0.6,
    enable_spatial_consistent: bool = True,
    enable_spatial_random: bool = False,
    enable_temporal_blackout: bool = False,
    blackout_ratio: float = 0.5,
    enable_temporal_shuffle: bool = False,
    shuffle_ratio: float = 0.5,
    min_segment_len: int = 1,
    max_segment_len: int = 4
) -> List[Image.Image]:
    """
    Randomly blacken square patches in video frames with configurable augmentation modes.
    Used for video data augmentation in PAPO.

    All four augmentation modes can be enabled/disabled independently and will be applied
    sequentially in the following order:
        1. Spatial consistent -> 2. Spatial random -> 3. Temporal blackout -> 4. Temporal shuffle

    Args:
        frames: List of PIL Images, each representing a video frame
        patch_size: Size of the square patch to blacken
        black_prob: Probability of blackening each patch
        enable_spatial_consistent: Whether to apply consistent spatial masking (all frames same mask)
        enable_spatial_random: Whether to apply random spatial masking (each frame independent)
        enable_temporal_blackout: Whether to enable temporal blackout (consecutive frames blackening)
        blackout_ratio: Ratio of frames to blackout (e.g., 0.5 means 50% of frames)
        enable_temporal_shuffle: Whether to enable temporal shuffling (segment-level shuffling)
        shuffle_ratio: Ratio of segments to shuffle (e.g., 0.5 means 50% of segments)
        min_segment_len: Minimum length of a segment for shuffling
        max_segment_len: Maximum length of a segment for shuffling

    Returns:
        List of augmented PIL Images (frames)
    """
    if len(frames) == 0:
        return frames

    num_frames = len(frames)
    # Start with original frames
    aug_frames = [frame.copy() for frame in frames]

    # Step 1: Apply consistent spatial augmentation if enabled
    if enable_spatial_consistent:
        first_frame = np.array(aug_frames[0])
        h, w = first_frame.shape[:2]

        # Pre-generate mask
        mask = np.zeros((h, w), dtype=bool)
        for y in range(0, h, patch_size):
            for x in range(0, w, patch_size):
                if np.random.rand() < black_prob:
                    y_end = min(y + patch_size, h)
                    x_end = min(x + patch_size, w)
                    mask[y:y_end, x:x_end] = True

        # Apply to all frames
        for i, frame in enumerate(aug_frames):
            img = np.array(frame).astype(np.float32)
            if img.ndim == 3:
                img[mask] = 0
            else:
                img[mask] = 0
            aug_frames[i] = Image.fromarray(img.astype(np.uint8))

    # Step 2: Apply random spatial augmentation if enabled
    if enable_spatial_random:
        for i, frame in enumerate(aug_frames):
            aug_frames[i] = random_patch_blackening(frame, patch_size, black_prob)

    # Step 3: Apply temporal blackout if enabled
    if enable_temporal_blackout and num_frames > 1:
        # Calculate number of frames to blackout
        num_blackout_frames = max(1, int(num_frames * blackout_ratio))
        num_blackout_frames = min(num_blackout_frames, num_frames)  # Ensure not exceeding total frames

        # Randomly select start index
        max_start_idx = num_frames - num_blackout_frames
        if max_start_idx >= 0:
            start_idx = np.random.randint(0, max_start_idx + 1)

            # Blackout consecutive frames from start_idx
            for i in range(start_idx, min(start_idx + num_blackout_frames, num_frames)):
                img = np.array(aug_frames[i]).astype(np.float32)
                img[:] = 0  # Complete blackening
                aug_frames[i] = Image.fromarray(img.astype(np.uint8))

    # Step 4: Apply temporal shuffling if enabled
    if enable_temporal_shuffle and num_frames > 1:
        # Split frames into segments
        segments = []
        current_idx = 0

        while current_idx < num_frames:
            # Randomly determine segment length
            remaining = num_frames - current_idx
            if remaining <= min_segment_len:
                segment_len = remaining
            else:
                segment_len = np.random.randint(min_segment_len, min(max_segment_len, remaining) + 1)

            segments.append(list(range(current_idx, current_idx + segment_len)))
            current_idx += segment_len

        # Decide which segments to shuffle based on shuffle_ratio
        num_segments_to_shuffle = max(1, int(len(segments) * shuffle_ratio))
        num_segments_to_shuffle = min(num_segments_to_shuffle, len(segments))

        # Randomly select segments to shuffle
        if len(segments) > 1:
            shuffle_indices = np.random.choice(len(segments), size=num_segments_to_shuffle, replace=False)

            # Create new frame order
            new_order = []
            for i, seg in enumerate(segments):
                if i in shuffle_indices:
                    # Shuffle this segment internally
                    np.random.shuffle(seg)
                new_order.extend(seg)

            # Reorder frames
            aug_frames = [aug_frames[i] for i in new_order]

    return aug_frames
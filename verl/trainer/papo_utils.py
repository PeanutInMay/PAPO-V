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

def random_patch_blackening(pil_img, patch_size=14, black_prob=0.6):
    """Randomly blacken square patches in a PIL image."""
    img = np.array(pil_img).astype(np.float32)
    # print(f"[DEBUG random_patch_blackening] type(pil_img)={type(pil_img)}, img.shape={img.shape}")
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
    
    
# 需要针对video frame设计新的数据增强方法，以下是一些可能的思路：
# 1. 空间一致性：
    # 随机从某一帧开始，连续黑化若干帧的相同位置的patch，以模拟视频中的遮挡或损坏。
# 2. 时间扰动：
    # 在连续的某一段帧中，把这些帧替换为全黑画面
# 3. 打乱顺序
    # 对帧进行重排列（具体的规则）
# 4. 删除一段连续时间的帧
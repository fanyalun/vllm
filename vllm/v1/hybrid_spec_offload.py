# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import IntEnum


class HybridSpecReloadMode(IntEnum):
    NONE = 0
    CPU_SHADOW = 1
    PRELOADED = 2

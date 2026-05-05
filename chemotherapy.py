import numpy as np
import random

EMPTY, HEALTHY, CANCER, CANCER_STEM, IMMUNE, NECROTIC = 0, 1, 2, 3, 4, 5

def apply_chemotherapy(grid, p_kill_rtc=0.8, p_kill_stc=0.3, p_tox_healthy=0.05, p_tox_immune=0.15):
    """
    Модель хіміотерапії.

    Аргументи:
        p_kill_rtc: ефективність проти звичайних ракових клітин (висока)
        p_kill_stc: ефективність проти стовбурових клітин (низька, бо вони резистентні)
        p_tox_healthy: токсичність для здорових клітин (побічний ефект)
        p_tox_immune: токсичність для імунних клітин (імуносупресія)
    """
    new_grid = np.copy(grid)

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            state = grid[r, c]

            if state == CANCER:
                if random.random() < p_kill_rtc:
                    new_grid[r, c] = NECROTIC

            elif state == CANCER_STEM:
                if random.random() < p_kill_stc:
                    new_grid[r, c] = NECROTIC

            elif state == HEALTHY:
                if random.random() < p_tox_healthy:
                    new_grid[r, c] = NECROTIC

            elif state == IMMUNE:
                if random.random() < p_tox_immune:
                    new_grid[r, c] = EMPTY

    return new_grid

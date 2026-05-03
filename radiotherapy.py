import numpy as np
import random

EMPTY, HEALTHY, CANCER, CANCER_STEM, IMMUNE, NECROTIC = 0, 1, 2, 3, 4, 5

def apply_radiotherapy(grid, target_r, target_c, radius, p_kill_rtc=0.95, p_kill_stc=0.6, p_tox=0.3):
    """
    Модель радіотерапії

    Аргументи:
        target_r: координата X (рядок) епіцентру, куди цілиться промінь.
        target_c: координата Y (стовпець) епіцентру, куди цілиться промінь.
        radius: радіус дії променя (у клітинках). Усе, що далі — не опромінюється.
        p_kill_rtc: ефективність проти звичайних ракових клітин (висока)
        p_kill_stc: ефективність проти стовбурових клітин (низька, бо вони резистентні)
        p_tox_healthy: токсичність для здорових клітин (побічний ефект)
        p_tox_immune: токсичність для імунних клітин (імуносупресія)
    """
    new_grid = np.copy(grid)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            distance = ((r - target_r)**2 + (c - target_c)**2)**0.5
            if distance <= radius:
                state = grid[r, c]
                if state == CANCER:
                    if random.random() < p_kill_rtc:
                        new_grid[r, c] = NECROTIC

                elif state == CANCER_STEM:
                    if random.random() < p_kill_stc:
                        new_grid[r, c] = NECROTIC

                elif state in (HEALTHY, IMMUNE):
                    if random.random() < p_tox:
                        new_grid[r, c] = NECROTIC
    return new_grid

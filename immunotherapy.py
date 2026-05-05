import numpy as np
import random

EMPTY, HEALTHY, CANCER, CANCER_STEM, IMMUNE, NECROTIC = 0, 1, 2, 3, 4, 5


def apply_immunotherapy(grid, boost_radius=2, p_kill_cancer=0.85, p_kill_stem=0.4, recruitment_rate=0.03):
    """
    Модель системної імунотерапії.
    Діє у два етапи:
    1. Активація: Існуючі імунні клітини (IMMUNE) отримують здатність "бачити" і вбивати
       ракові клітини в заданому радіусі.
    2. Рекрутинг: У порожніх клітинах або на місці некрозу з'являються нові імунні клітини (Т-лімфоцити).
    """
    new_grid = np.copy(grid)
    rows, cols = grid.shape

    for r in range(rows):
        for c in range(cols):
            state = grid[r, c]

            if state in [EMPTY, NECROTIC]:
                if random.random() < recruitment_rate:
                    new_grid[r, c] = IMMUNE

            elif state == IMMUNE:
                for dr in range(-boost_radius, boost_radius + 1):
                    for dc in range(-boost_radius, boost_radius + 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            target_state = grid[nr, nc]
                            if target_state == CANCER:
                                if random.random() < p_kill_cancer:
                                    new_grid[nr, nc] = NECROTIC
                            elif target_state == CANCER_STEM:
                                if random.random() < p_kill_stem:
                                    new_grid[nr, nc] = NECROTIC

    return new_grid

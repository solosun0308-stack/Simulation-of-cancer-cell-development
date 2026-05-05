"""
Порівняння чотирьох сценаріїв лікування:
  1. Без лікування
  2. Хіміотерапія
  3. Радіотерапія
  4. Імунотерапія

Запуск:
  python compare_treatments.py
  python compare_treatments.py 80 200
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

from tumor_new import TumorCA, SimulationParams
from chemotherapy import apply_chemotherapy
from radiotherapy import apply_radiotherapy
from immunotherapy import apply_immunotherapy

SEED = 42


def make_ca(grid_size):
    params = SimulationParams(
        grid_size=grid_size,
        prob_stem_division=0.02,
        initial_healthy_density=0.2,
        initial_immune_count=20,
        prob_immune_kill=0.2
    )
    ca = TumorCA(params)
    ca.initialize(cancer_type='stem')
    return ca


def run_no_treatment(grid_size, n_steps):
    np.random.seed(SEED)
    ca = make_ca(grid_size)
    for _ in range(n_steps):
        ca.simulation_step()
    return ca.history


def run_chemo(grid_size, n_steps, chemo_interval=10):
    np.random.seed(SEED)
    ca = make_ca(grid_size)
    for step in range(n_steps):
        ca.simulation_step()
        if (step + 1) % chemo_interval == 0:
            ca.grid = apply_chemotherapy(ca.grid)
    return ca.history


def run_radio(grid_size, n_steps, radio_at=None, radio_radius=6):
    np.random.seed(SEED)
    ca = make_ca(grid_size)
    if radio_at is None:
        radio_at = n_steps // 2
    center = grid_size // 2
    for step in range(n_steps):
        ca.simulation_step()
        if (step + 1) == radio_at:
            ca.grid = apply_radiotherapy(
                ca.grid,
                target_r=center,
                target_c=center,
                radius=radio_radius
            )
            print(f"  Радіотерапія застосована на кроці {step+1}")
    return ca.history


def run_immuno(grid_size, n_steps, immuno_interval=10):
    np.random.seed(SEED)
    ca = make_ca(grid_size)
    for step in range(n_steps):
        ca.simulation_step()
        if (step + 1) % immuno_interval == 0:
            ca.grid = apply_immunotherapy(ca.grid)
    return ca.history


def plot_comparison(h_no, h_chemo, h_radio, h_immuno, n_steps):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Порівняння методів лікування', fontsize=16, fontweight='bold')

    # --- Загальний розмір пухлини ---
    ax = axes[0, 0]
    ax.plot(h_no['step'],     [c + s for c, s in zip(h_no['cancer'],     h_no['stem'])],     color='crimson',   lw=2, label='Без лікування')
    ax.plot(h_chemo['step'],  [c + s for c, s in zip(h_chemo['cancer'],  h_chemo['stem'])],  color='royalblue', lw=2, label='Хіміотерапія')
    ax.plot(h_radio['step'],  [c + s for c, s in zip(h_radio['cancer'],  h_radio['stem'])],  color='green',     lw=2, label='Радіотерапія')
    ax.plot(h_immuno['step'], [c + s for c, s in zip(h_immuno['cancer'], h_immuno['stem'])], color='orange',    lw=2, label='Імунотерапія')
    ax.set_title('Загальний розмір пухлини (RTC + STC)')
    ax.set_xlabel('Крок (дні)')
    ax.set_ylabel('Кількість клітин')
    ax.legend()
    ax.grid(alpha=0.3)

    # --- Некроз ---
    ax = axes[0, 1]
    ax.plot(h_no['step'],     h_no['necrotic'],     color='crimson',   lw=2, label='Без лікування')
    ax.plot(h_chemo['step'],  h_chemo['necrotic'],  color='royalblue', lw=2, label='Хіміотерапія')
    ax.plot(h_radio['step'],  h_radio['necrotic'],  color='green',     lw=2, label='Радіотерапія')
    ax.plot(h_immuno['step'], h_immuno['necrotic'], color='orange',    lw=2, label='Імунотерапія')
    ax.set_title('Некротичні клітини')
    ax.set_xlabel('Крок (дні)')
    ax.set_ylabel('Кількість клітин')
    ax.legend()
    ax.grid(alpha=0.3)

    # --- Імунні клітини ---
    ax = axes[1, 0]
    ax.plot(h_no['step'],     h_no['immune'],     color='crimson',   lw=2, label='Без лікування')
    ax.plot(h_chemo['step'],  h_chemo['immune'],  color='royalblue', lw=2, label='Хіміотерапія')
    ax.plot(h_radio['step'],  h_radio['immune'],  color='green',     lw=2, label='Радіотерапія')
    ax.plot(h_immuno['step'], h_immuno['immune'], color='orange',    lw=2, label='Імунотерапія')
    ax.set_title('Імунні клітини')
    ax.set_xlabel('Крок (дні)')
    ax.set_ylabel('Кількість клітин')
    ax.legend()
    ax.grid(alpha=0.3)

    # --- Здорові клітини ---
    ax = axes[1, 1]
    ax.plot(h_no['step'],     h_no['healthy'],     color='crimson',   lw=2, label='Без лікування')
    ax.plot(h_chemo['step'],  h_chemo['healthy'],  color='royalblue', lw=2, label='Хіміотерапія')
    ax.plot(h_radio['step'],  h_radio['healthy'],  color='green',     lw=2, label='Радіотерапія')
    ax.plot(h_immuno['step'], h_immuno['healthy'], color='orange',    lw=2, label='Імунотерапія')
    ax.set_title('Здорові клітини')
    ax.set_xlabel('Крок (дні)')
    ax.set_ylabel('Кількість клітин')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('treatment_comparison.png', dpi=150, bbox_inches='tight')
    print("Графік збережено: treatment_comparison.png")
    plt.show()


def print_summary(h_no, h_chemo, h_radio, h_immuno, n_steps):
    def peak(h):
        return max(c + s for c, s in zip(h['cancer'], h['stem']))

    def final(h):
        return h['cancer'][-1] + h['stem'][-1]

    print("\n" + "═" * 55)
    print(f"  ПОРІВНЯННЯ ЛІКУВАННЯ  (кроків: {n_steps})")
    print("═" * 55)
    print(f"{'Сценарій':<20} {'Пік пухлини':>12} {'Фінал':>10}")
    print("─" * 55)
    for name, h in [
        ("Без лікування", h_no),
        ("Хіміотерапія",  h_chemo),
        ("Радіотерапія",  h_radio),
        ("Імунотерапія",  h_immuno),
    ]:
        print(f"{name:<20} {peak(h):>12,} {final(h):>10,}")
    print("═" * 55 + "\n")


if __name__ == '__main__':
    N     = int(sys.argv[1]) if len(sys.argv) > 1 else 70
    STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 150

    print(f"Решітка: {N}×{N} | Кроків: {STEPS}\n")

    print("Запускаю сценарій 1/4: без лікування...")
    h_no = run_no_treatment(N, STEPS)

    print("Запускаю сценарій 2/4: хіміотерапія...")
    h_chemo = run_chemo(N, STEPS)

    print("Запускаю сценарій 3/4: радіотерапія...")
    h_radio = run_radio(N, STEPS)

    print("Запускаю сценарій 4/4: імунотерапія...")
    h_immuno = run_immuno(N, STEPS)

    print_summary(h_no, h_chemo, h_radio, h_immuno, STEPS)
    plot_comparison(h_no, h_chemo, h_radio, h_immuno, STEPS)
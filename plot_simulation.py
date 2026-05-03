"""
Статистика і графіки для клітинного автомату пухлини.

Запуск:
  python plot_simulation.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from tumor_new import (
    TumorCA, SimulationParams,
    EMPTY, HEALTHY, CANCER, CANCER_STEM, IMMUNE, NECROTIC
)

np.random.seed(42)

def _run(params, n_steps, cancer_type='stem'):
    ca = TumorCA(params)
    ca.initialize(cancer_type=cancer_type)
    for _ in range(n_steps):
        ca.simulation_step()
    return ca


def _run_multi(params, n_tumors, n_steps):
    ca = TumorCA(params)
    ca.initialize_multi(n_tumors=n_tumors, cancer_type='stem')
    for _ in range(n_steps):
        ca.simulation_step()
    return ca


def collect_data(N=70, STEPS=150):
    print("Запускаю симуляції...")

    print("  [1/4] RTC сценарій...")
    ca_rtc = _run(SimulationParams(
        grid_size=N, initial_healthy_density=0,
        initial_immune_count=0, prob_apoptosis=0
    ), STEPS, cancer_type='regular')

    print("  [2/4] STC сценарій...")
    ca_stc = _run(SimulationParams(
        grid_size=N, initial_healthy_density=0,
        initial_immune_count=0, prob_stem_division=0
    ), STEPS, cancer_type='stem')

    print("  [3/4] STC + імунна...")
    ca_immune = _run(SimulationParams(
        grid_size=N, initial_healthy_density=0.2,
        initial_immune_count=20, prob_stem_division=0.02,
        prob_immune_kill=0.2
    ), STEPS, cancer_type='stem')

    print("  [4/4] Мультипухлинний сценарій...")
    ca_multi = _run_multi(SimulationParams(
        grid_size=N, initial_healthy_density=0.2,
        initial_immune_count=30, prob_stem_division=0.02,
        prob_immune_kill=0.15
    ), n_tumors=5, n_steps=STEPS)

    return ca_rtc, ca_stc, ca_immune, ca_multi


def print_stats(ca_rtc, ca_stc, ca_immune, ca_multi, steps):
    def peak(hist):
        return max(hist['cancer'][i] + hist['stem'][i] for i in range(len(hist['step'])))

    def final(hist):
        return hist['cancer'][-1] + hist['stem'][-1]

    def survival(hist):
        """На якому кроці пухлина зникла, або None якщо вижила."""
        for i, (c, s) in enumerate(zip(hist['cancer'], hist['stem'])):
            if c + s == 0 and i > 5:
                return hist['step'][i]
        return None

    print("\n" + "═" * 56)
    print(f"  ПІДСУМКОВА СТАТИСТИКА  (кроків: {steps})")
    print("═" * 56)
    rows = [
        ("RTC",         ca_rtc),
        ("STC",         ca_stc),
        ("STC+імунна",  ca_immune),
        ("Мульти",      ca_multi),
    ]
    print(f"{'Сценарій':<14} {'Пік пухлини':>12} {'Фінал':>8} {'Зникла на':>11}")
    print("─" * 56)
    for name, ca in rows:
        p = peak(ca.history)
        f = final(ca.history)
        s = survival(ca.history)
        survived = f"крок {s}" if s else "—  (жива)"
        print(f"{name:<14} {p:>12,} {f:>8,} {survived:>11}")
    print("═" * 56 + "\n")

CMAP_COLORS = ['#1a1a2e', '#4caf50', '#f44336', '#b71c1c', '#2196f3', '#9e9e9e']
LEGEND_PATCHES = [
    mpatches.Patch(color='#1a1a2e', label='Порожньо'),
    mpatches.Patch(color='#4caf50', label='Здорова'),
    mpatches.Patch(color='#f44336', label='RTC'),
    mpatches.Patch(color='#b71c1c', label='STC'),
    mpatches.Patch(color='#2196f3', label='Імунна'),
    mpatches.Patch(color='#9e9e9e', label='Некроз'),
]


def _grid_to_color(grid):
    m = np.zeros(grid.shape, dtype=float)
    m[grid == HEALTHY]     = 1
    m[grid == CANCER]      = 2
    m[grid == CANCER_STEM] = 3
    m[grid == IMMUNE]      = 4
    m[grid == NECROTIC]    = 5
    return m
def _total(history):
    return [c + s for c, s in zip(history['cancer'], history['stem'])]


def plot_all(ca_rtc, ca_stc, ca_immune, ca_multi, steps):
    cmap = plt.cm.colors.ListedColormap(CMAP_COLORS)

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Аналіз клітинного автомату пухлини',
                 fontsize=17, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)
    ax1 = fig.add_subplot(gs[0, 0])
    s = ca_rtc.history['step']
    ax1.plot(s, ca_rtc.history['cancer'],   color='crimson', lw=2,   label='RTC')
    ax1.plot(s, ca_rtc.history['necrotic'], color='gray',    lw=1.5, ls='--', label='Некроз')
    ax1.set_title('Сценарій 1 — тільки RTC\n(пухлина зникає)', fontsize=11)
    ax1.set_xlabel('Крок (дні)')
    ax1.set_ylabel('Кількість клітин')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.25)


    ax2 = fig.add_subplot(gs[0, 1])
    s = ca_stc.history['step']
    ax2.plot(s, ca_stc.history['cancer'], color='crimson', lw=2,   label='RTC')
    ax2.plot(s, ca_stc.history['stem'],   color='darkred', lw=2.5, label='STC')
    ax2.set_title('Сценарій 2 — STC\n(безсмертна стовбурова)', fontsize=11)
    ax2.set_xlabel('Крок (дні)')
    ax2.set_ylabel('Кількість клітин')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)


    ax3 = fig.add_subplot(gs[0, 2])
    s = ca_immune.history['step']
    ax3.plot(s, ca_immune.history['cancer'],   color='crimson',   lw=2,   label='RTC')
    ax3.plot(s, ca_immune.history['stem'],     color='darkred',   lw=2.5, label='STC')
    ax3.plot(s, ca_immune.history['immune'],   color='royalblue', lw=1.5, label='Імунні')
    ax3.plot(s, ca_immune.history['necrotic'], color='gray',      lw=1,   ls='--', label='Некроз')
    ax3.set_title('Сценарій 3 — STC + імунна система', fontsize=11)
    ax3.set_xlabel('Крок (дні)')
    ax3.set_ylabel('Кількість клітин')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.25)


    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(ca_rtc.history['step'],    _total(ca_rtc.history),    color='orange',    lw=2, label='RTC')
    ax4.plot(ca_stc.history['step'],    _total(ca_stc.history),    color='red',       lw=2, label='STC')
    ax4.plot(ca_immune.history['step'], _total(ca_immune.history), color='royalblue', lw=2, label='STC+імунна')
    ax4.plot(ca_multi.history['step'],  _total(ca_multi.history),  color='purple',    lw=2, label='Мульти')
    ax4.set_title('Порівняння: загальний розмір пухлини', fontsize=11)
    ax4.set_xlabel('Крок (дні)')
    ax4.set_ylabel('RTC + STC разом')
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.25)

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(_grid_to_color(ca_immune.grid), cmap=cmap,
               vmin=0, vmax=5, interpolation='nearest')
    ax5.set_title(f'Решітка: STC+імунна (крок {steps})', fontsize=11)
    ax5.axis('off')
    ax5.legend(handles=LEGEND_PATCHES, loc='lower right', fontsize=8, framealpha=0.8)


    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(_grid_to_color(ca_multi.grid), cmap=cmap,
               vmin=0, vmax=5, interpolation='nearest')
    ax6.set_title(f'Решітка: мульти-пухлина (крок {steps})', fontsize=11)
    ax6.axis('off')
    ax6.legend(handles=LEGEND_PATCHES, loc='lower right', fontsize=8, framealpha=0.8)

    plt.savefig('tumor_analysis.png', dpi=150, bbox_inches='tight')
    print("Графік збережено: tumor_analysis.png")
    plt.show()

if __name__ == '__main__':
    N     = int(sys.argv[1]) if len(sys.argv) > 1 else 70
    STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 150

    ca_rtc, ca_stc, ca_immune, ca_multi = collect_data(N=N, STEPS=STEPS)
    print_stats(ca_rtc, ca_stc, ca_immune, ca_multi, STEPS)
    plot_all(ca_rtc, ca_stc, ca_immune, ca_multi, STEPS)

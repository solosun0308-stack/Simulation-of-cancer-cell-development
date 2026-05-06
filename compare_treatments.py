"""
Порівняння чотирьох сценаріїв лікування:
  1. Без лікування
  2. Хіміотерапія
  3. Радіотерапія
  4. Імунотерапія

Запуск:
  python compare_treatments.py
  python compare_treatments.py --runs 30 --steps 200 --grid-size 80
"""

import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tumor_new import TumorCA, SimulationParams
from chemotherapy import apply_chemotherapy
from radiotherapy import apply_radiotherapy
from immunotherapy import apply_immunotherapy


def make_ca(grid_size, seed):
    np.random.seed(seed)
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


def run_no_treatment(grid_size, n_steps, seed=42):
    ca = make_ca(grid_size, seed)
    for _ in range(n_steps):
        ca.simulation_step()
    return ca.history


def run_chemo(grid_size, n_steps, chemo_interval=10, seed=42):
    ca = make_ca(grid_size, seed)
    for step in range(n_steps):
        ca.simulation_step()
        if (step + 1) % chemo_interval == 0:
            ca.grid = apply_chemotherapy(ca.grid)
    return ca.history


def run_radio(grid_size, n_steps, radio_at=None, radio_radius=6, seed=42):
    ca = make_ca(grid_size, seed)
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
    return ca.history


def run_immuno(grid_size, n_steps, immuno_interval=10, seed=42):
    ca = make_ca(grid_size, seed)
    for step in range(n_steps):
        ca.simulation_step()
        if (step + 1) % immuno_interval == 0:
            ca.grid = apply_immunotherapy(ca.grid)
    return ca.history


def average_histories(histories):
    """Рахує середнє по всіх запусках."""
    keys = ['cancer', 'stem', 'immune', 'necrotic', 'healthy']
    avg = {'step': histories[0]['step']}
    for k in keys:
        avg[k] = np.mean([h[k] for h in histories], axis=0).tolist()
    return avg


def run_experiment(grid_size, n_steps, runs):
    """Запускає всі сценарії runs разів і повертає усереднені history."""
    scenarios = {
        'no_treatment': [],
        'chemo':        [],
        'radio':        [],
        'immuno':       [],
    }

    for i in range(runs):
        seed = i
        if runs > 1:
            print(f"  Запуск {i+1}/{runs}...", end='\r')

        scenarios['no_treatment'].append(run_no_treatment(grid_size, n_steps, seed))
        scenarios['chemo'].append(run_chemo(grid_size, n_steps, seed=seed))
        scenarios['radio'].append(run_radio(grid_size, n_steps, seed=seed))
        scenarios['immuno'].append(run_immuno(grid_size, n_steps, seed=seed))

    if runs > 1:
        print()

    return {name: average_histories(histories) for name, histories in scenarios.items()}


def save_csv(results, filename='treatment_comparison.csv'):
    fieldnames = ['scenario', 'step', 'cancer', 'stem', 'immune', 'necrotic', 'healthy', 'total_tumor']
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        labels = {
            'no_treatment': 'Без лікування',
            'chemo':        'Хіміотерапія',
            'radio':        'Радіотерапія',
            'immuno':       'Імунотерапія',
        }
        for key, h in results.items():
            for i in range(len(h['step'])):
                writer.writerow({
                    'scenario':    labels[key],
                    'step':        h['step'][i],
                    'cancer':      round(h['cancer'][i], 2),
                    'stem':        round(h['stem'][i], 2),
                    'immune':      round(h['immune'][i], 2),
                    'necrotic':    round(h['necrotic'][i], 2),
                    'healthy':     round(h['healthy'][i], 2),
                    'total_tumor': round(h['cancer'][i] + h['stem'][i], 2),
                })
    print(f"CSV збережено: {filename}")


def plot_comparison(results, n_steps, runs):
    h_no     = results['no_treatment']
    h_chemo  = results['chemo']
    h_radio  = results['radio']
    h_immuno = results['immuno']

    title_suffix = f" (середнє по {runs} запусках)" if runs > 1 else ""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Порівняння методів лікування' + title_suffix,
                 fontsize=16, fontweight='bold')

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
    plt.close()


def print_summary(results, n_steps, runs):
    def peak(h):
        return max(c + s for c, s in zip(h['cancer'], h['stem']))

    def final(h):
        return h['cancer'][-1] + h['stem'][-1]

    suffix = f" (середнє по {runs} запусках)" if runs > 1 else ""
    print(f"  ПОРІВНЯННЯ ЛІКУВАННЯ  (кроків: {n_steps}){suffix}")
    print(f"{'Сценарій':<20} {'Пік пухлини':>12} {'Фінал':>10}")
    labels = {
        'no_treatment': 'Без лікування',
        'chemo':        'Хіміотерапія',
        'radio':        'Радіотерапія',
        'immuno':       'Імунотерапія',
    }
    for key, h in results.items():
        print(f"{labels[key]:<20} {peak(h):>12.1f} {final(h):>10.1f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Порівняння методів лікування пухлини")
    parser.add_argument("--runs",      type=int, default=1,  help="Кількість запусків для усереднення")
    parser.add_argument("--steps",     type=int, default=150, help="Кількість кроків симуляції")
    parser.add_argument("--grid-size", type=int, default=70,  help="Розмір решітки N×N")
    parser.add_argument("N",     type=int, nargs='?', help=argparse.SUPPRESS)
    parser.add_argument("STEPS", type=int, nargs='?', help=argparse.SUPPRESS)
    args = parser.parse_args()

    N     = args.N     or args.grid_size
    STEPS = args.STEPS or args.steps
    RUNS  = args.runs

    print(f"Решітка: {N}×{N} | Кроків: {STEPS} | Запусків: {RUNS}\n")

    results = run_experiment(N, STEPS, RUNS)
    print_summary(results, STEPS, RUNS)
    save_csv(results)
    plot_comparison(results, STEPS, RUNS)

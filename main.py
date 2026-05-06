"""
  python3 main.py --scenario no-treatment --steps 200 --grid-size 80
  python3 main.py --scenario stem --steps 150 --grid-size 80
  python3 main.py --scenario immune --steps 150 --grid-size 80
  python3 main.py --scenario multi --steps 150 --grid-size 120
  python3 main.py --scenario chemo --steps 150 --chemo-interval 10
  python3 main.py --scenario radio --radio-radius 8 --seed 42
"""

import argparse
from tumor_new import (SimulationParams, TumorCA, run_simulation, scenario_nonclonogenic, scenario_stem, scenario_immune, scenario_multi)
from chemotherapy import apply_chemotherapy
from radiotherapy import apply_radiotherapy
from immunotherapy import apply_immunotherapy



def parse_args():
    parser = argparse.ArgumentParser(
        description="Симуляція росту ракової пухлини")
    parser.add_argument("--steps", type=int, default=200, help="Кількість кроків симуляції")
    parser.add_argument("--grid-size", type=int, default=80, help="Розмір решітки N×N")
    parser.add_argument("--scenario", type=str, default="no-treatment", choices=["no-treatment", "rtc", "stem", "immune", "multi", "chemo", "radio", "immuno"],
                        help="Сценарій: no-treatment | rtc | stem | immune | multi | chemo | radio")
    parser.add_argument("--chemo-interval", type=int, default=10, help="Кожні скільки кроків застосовувати хімію")
    parser.add_argument("--radio-radius", type=int, default=6, help="Радіус дії радіотерапії")
    parser.add_argument("--seed", type=int, default=None, help="Random seed для відтворюваності")
    return parser.parse_args()

def _simple_run(ca, steps):
    """Запуск без анімації"""
    for step in range(steps):
        ca.simulation_step()
        if (step + 1) % 10 == 0:
            print(ca.stats_str())
    return ca

def run_no_treatment(args):
    import numpy as np
    if args.seed is not None:
        np.random.seed(args.seed)
    params = SimulationParams(grid_size=args.grid_size,
                              prob_stem_division=0.02,
                              initial_healthy_density=0.2,
                              initial_immune_count=20)
    ca = TumorCA(params)
    ca.initialize(cancer_type='stem')
    return _simple_run(ca, args.steps)

def run_rtc(args):
    import numpy as np
    if args.seed is not None:
        np.random.seed(args.seed)
    params = SimulationParams(grid_size=args.grid_size,
                              prob_apoptosis=0.0,
                              initial_healthy_density=0.0,
                              initial_immune_count=0)
    ca = TumorCA(params)
    ca.initialize(cancer_type='regular')
    return _simple_run(ca, args.steps)

def run_stem(args):
    import numpy as np
    if args.seed is not None:
        np.random.seed(args.seed)
    params = SimulationParams(grid_size=args.grid_size,
                              prob_apoptosis=0.0,
                              prob_stem_division=0.0,
                              initial_healthy_density=0.0,
                              initial_immune_count=0)
    ca = TumorCA(params)
    ca.initialize(cancer_type='stem')
    return _simple_run(ca, args.steps)


def run_immune(args):
    import numpy as np
    if args.seed is not None:
        np.random.seed(args.seed)
    params = SimulationParams(grid_size=args.grid_size,
                              prob_stem_division=0.02,
                              initial_healthy_density=0.2,
                              initial_immune_count=20,
                              prob_immune_kill=0.2)
    ca = TumorCA(params)
    ca.initialize(cancer_type='stem')
    return _simple_run(ca, args.steps)


def run_multi(args):
    import numpy as np
    if args.seed is not None:
        np.random.seed(args.seed)
    params = SimulationParams(grid_size=args.grid_size,
                              prob_stem_division=0.02,
                              initial_healthy_density=0.2,
                              initial_immune_count=30,
                              prob_immune_kill=0.15)
    ca = TumorCA(params)
    ca.initialize_multi(n_tumors=5, cancer_type='stem')
    return _simple_run(ca, args.steps)


def run_chemo(args):
    import numpy as np
    if args.seed is not None:
        np.random.seed(args.seed)
    params = SimulationParams(grid_size=args.grid_size,
                              prob_stem_division=0.02,
                              initial_healthy_density=0.2,
                              initial_immune_count=20)
    ca = TumorCA(params)
    ca.initialize(cancer_type='stem')
    for step in range(args.steps):
        ca.simulation_step()
        if (step + 1) % args.chemo_interval == 0:
            ca.grid = apply_chemotherapy(ca.grid)
            print(f"[Крок {step+1}] Хімія застосована. {ca.stats_str()}")
        elif (step + 1) % 10 == 0:
            print(ca.stats_str())
    return ca

def run_radio(args):
    import numpy as np
    if args.seed is not None:
        np.random.seed(args.seed)
    params = SimulationParams(grid_size=args.grid_size,
                              prob_stem_division=0.02,
                              initial_healthy_density=0.2,
                              initial_immune_count=20)
    ca = TumorCA(params)
    ca.initialize(cancer_type='stem')
    _simple_run(ca, args.steps)
    center = args.grid_size // 2
    ca.grid = apply_radiotherapy(ca.grid,
                                 target_r=center,
                                 target_c=center,
                                 radius=args.radio_radius)
    print(f"\nРадіотерапія застосована (центр={center}, радіус={args.radio_radius})")
    print(ca.stats_str())
    return ca

def run_immuno(args):
    import numpy as np
    if args.seed is not None:
        np.random.seed(args.seed)
    params = SimulationParams(grid_size=args.grid_size,
                              prob_stem_division=0.02,
                              initial_healthy_density=0.2,
                              initial_immune_count=20)
    ca = TumorCA(params)
    ca.initialize(cancer_type='stem')
    for step in range(args.steps):
        ca.simulation_step()
        if (step + 1) % 10 == 0:
            ca.grid = apply_immunotherapy(ca.grid)
            print(f"[Крок {step+1}] Імунотерапія застосована. {ca.stats_str()}")
        elif (step + 1) % 10 == 0:
            print(ca.stats_str())
    return ca


def main():
    args = parse_args()
    print(f"Сценарій: {args.scenario} | Кроки: {args.steps} | Решітка: {args.grid_size}×{args.grid_size}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")

    runners = {
        "no-treatment": run_no_treatment,
        "rtc":          run_rtc,
        "stem":         run_stem,
        "immune":       run_immune,
        "multi":        run_multi,
        "chemo":        run_chemo,
        "radio":        run_radio,
        "immuno": run_immuno
    }
    runners[args.scenario](args)


if __name__ == "__main__":
    main()

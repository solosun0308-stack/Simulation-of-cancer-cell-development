"""
!Клітинний автомат для симуляції росту ракової пухлини!

Формальний запис: CA = (C, n, S, f)
  C — множина клітин (2D решітка N×N)
  n — функція сусідства (Moore: 8 сусідів)
  S — множина станів {EMPTY, HEALTHY, CANCER, CANCER_STEM, IMMUNE, NECROTIC}
  f — стохастична функція переходу станів

Джерела:
  [1] Cellular-automata modeling of tumor growth dynamics,
      Computers in Biology and Medicine, 2022
  [2] Stochastic cellular automata model of avascular tumor growth
      with chemotherapy, Math. and Comp. Modelling, 2019
"""
import numpy as np
import random
import os
import time

EMPTY       = 0
HEALTHY     = 1
CANCER      = 2
CANCER_STEM = 3
IMMUNE      = 4
NECROTIC    = 5

class SimulationParams:
    """
    Параметри моделі клітинного автомату.

    Значення ймовірностей взяті із зазначениї вище джерел:
      PP = Δt / CCT = 1 день / 24 год ≈ 4.17%  (час поділу клітини)
      PA = 0.5%  (програмована загибель клітини, апоптоз)
      pdT = 15%, pdI = 5%  (ефективність Т-лімфоцитів з досліджень)
      pmax = 10  (ліміт Хейфліка — клітини мають обмежену кількість поділів)
    """
    def __init__(self, **kwargs):
        self.grid_size = kwargs.get('grid_size', 100)


        self.prob_proliferation = kwargs.get('prob_proliferation', 0.04)
        self.prob_migration     = kwargs.get('prob_migration', 0.02)
        self.prob_apoptosis     = kwargs.get('prob_apoptosis', 0.005)


        self.prob_stem_division        = kwargs.get('prob_stem_division', 0.01)
        self.max_proliferation_potential = kwargs.get('max_proliferation_potential', 10)


        self.prob_immune_kill    = kwargs.get('prob_immune_kill', 0.15)
        self.prob_immune_death   = kwargs.get('prob_immune_death', 0.05)
        self.immune_migration_speed = kwargs.get('immune_migration_speed', 0.3)
        self.immune_recruit_rate = kwargs.get('immune_recruit_rate', 0.001)


        self.initial_healthy_density = kwargs.get('initial_healthy_density', 0.3)
        self.initial_immune_count    = kwargs.get('initial_immune_count', 15)
        self.necrotic_decay_time     = kwargs.get('necrotic_decay_time', 50)

MOORE_DIRS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

def get_neighbors(row, col, grid_size):
    """Список координат сусідів"""
    return [
        (row + dr, col + dc)
        for dr, dc in MOORE_DIRS
        if 0 <= row + dr < grid_size and 0 <= col + dc < grid_size
    ]

class TumorCA:
    """
    Клітинний автомат, власною персоною.
    """
    def __init__(self, params=None):
        self.p = params or SimulationParams()
        self.N = self.p.grid_size
        self.grid     = np.full((self.N, self.N), EMPTY, dtype=np.int8)
        self.potential = np.zeros((self.N, self.N), dtype=np.int16)
        self.necrotic_timer = np.zeros((self.N, self.N), dtype=np.int16)
        self.step_count = 0
        self.history = {
            'step': [], 'cancer': [], 'stem': [],
            'immune': [], 'healthy': [], 'necrotic': []
        }

    def initialize(self, cancer_positions=None, cancer_type='stem'):
        """
        Ініціалізує поляну для симуляції.
        Args:
            cancer_positions: список координат початкових ракових клітин.
                              За замовчуванням - одна клітина в центрі.
            cancer_type: 'stem' | 'regular'
        """
        self.grid.fill(EMPTY)
        self.potential.fill(0)
        self.necrotic_timer.fill(0)
        self.step_count = 0
        self.history = {k: [] for k in self.history}
        mask = np.random.random((self.N, self.N)) < self.p.initial_healthy_density
        self.grid[mask] = HEALTHY
        center = self.N // 2
        if cancer_positions is None:
            cancer_positions = [(center, center)]


        for r, c in cancer_positions:
            if cancer_type == 'stem':
                self.grid[r, c] = CANCER_STEM
                self.potential[r, c] = self.p.max_proliferation_potential + 1
            else:
                self.grid[r, c] = CANCER
                self.potential[r, c] = self.p.max_proliferation_potential
        placed = 0
        attempts = 0
        while placed < self.p.initial_immune_count and attempts < self.N * self.N:
            r, c = np.random.randint(0, self.N, size=2)
            if self.grid[r, c] in (EMPTY, HEALTHY):
                self.grid[r, c] = IMMUNE
                placed += 1
            attempts += 1
        self._record_stats()
    

    def initialize_multi(self, n_tumors=5, cancer_type='stem'):
        """
        Ініціалізація з кількома пухлинами рандомно по карті.
        """
        self.grid.fill(EMPTY)
        self.potential.fill(0)
        self.necrotic_timer.fill(0)
        self.step_count = 0
        self.history = {k: [] for k in self.history}
        mask = np.random.random((self.N, self.N)) < self.p.initial_healthy_density
        self.grid[mask] = HEALTHY

        max_tumors = max(1, (self.N * self.N) // 200)
        n_tumors = min(n_tumors, max_tumors)

        placed = []
        attempts = 0
        while len(placed) < n_tumors and attempts < 1000:
            r = np.random.randint(5, self.N - 5)
            c = np.random.randint(5, self.N - 5)
            too_close = any(
                abs(r - pr) < 10 and abs(c - pc) < 10
                for pr, pc in placed
            )
            if not too_close and self.grid[r, c] == HEALTHY or self.grid[r, c] == EMPTY:
                if cancer_type == 'stem':
                    self.grid[r, c] = CANCER_STEM
                    self.potential[r, c] = self.p.max_proliferation_potential + 1
                else:
                    self.grid[r, c] = CANCER
                    self.potential[r, c] = self.p.max_proliferation_potential
                placed.append((r, c))
            attempts += 1

        placed_immune = 0
        attempts = 0
        while placed_immune < self.p.initial_immune_count and attempts < self.N * self.N:
            r, c = np.random.randint(0, self.N, size=2)
            if self.grid[r, c] in (EMPTY, HEALTHY):
                self.grid[r, c] = IMMUNE
                placed_immune += 1
            attempts += 1



        self._record_stats()
        return len(placed)

    def simulation_step(self):
        """
        Один крок симуляції (24 години)
        Порядок дій для кожної клітини випадковий.
        Оновлення синхронне(оновлення стану всіх клітин тоді застосування).
        """
        self.step_count += 1
        new_grid    = self.grid.copy()
        new_pot     = self.potential.copy()
        new_nec     = self.necrotic_timer.copy()
        cells = np.argwhere(self.grid != EMPTY)
        np.random.shuffle(cells)
        for row, col in cells:
            state = self.grid[row, col]
            if state == NECROTIC:
                new_nec[row, col] += 1
                if new_nec[row, col] >= self.p.necrotic_decay_time:
                    new_grid[row, col] = EMPTY
                    new_nec[row, col] = 0

            elif state == HEALTHY:
                pass
            elif state == IMMUNE:
                self._step_immune(row, col, new_grid, new_pot, new_nec)
            elif state in (CANCER, CANCER_STEM):
                self._step_cancer(row, col, state, new_grid, new_pot)

        self._recruit_immune(new_grid)
        self.grid     = new_grid
        self.potential = new_pot
        self.necrotic_timer = new_nec
        self._record_stats()

    def _step_cancer(self, row, col, state, new_grid, new_pot):
        """
        Функція переходу ракової клітини.

        За один крок - одна дія (або нічого):
          1. Апоптоз  (тільки RTC, PA = 0.5%)
          2. Поділ    (PP = 4%)
          3. Міграція (Pm = 2%)
          4. Спокій
        """
        is_stem = (state == CANCER_STEM)
        if not is_stem:
            if np.random.random() < self.p.prob_apoptosis:
                new_grid[row, col] = NECROTIC
                new_pot[row, col] = 0
                return
            if self.potential[row, col] <= 0:
                new_grid[row, col] = NECROTIC
                new_pot[row, col] = 0
                return
        neighbors = get_neighbors(row, col, self.N)
        random.shuffle(neighbors)
        free = [(r, c) for r, c in neighbors if new_grid[r, c] in (EMPTY, HEALTHY)]

        if not free:
            return


        if np.random.random() < self.p.prob_proliferation:
            tr, tc = free[0]
            if is_stem:
                if np.random.random() < self.p.prob_stem_division:
                    new_grid[tr, tc] = CANCER_STEM
                    new_pot[tr, tc] = self.p.max_proliferation_potential + 1
                else:
                    new_grid[tr, tc] = CANCER
                    new_pot[tr, tc] = self.p.max_proliferation_potential
            else:
                p = self.potential[row, col]
                new_grid[tr, tc] = CANCER
                new_pot[tr, tc] = p - 1
                new_pot[row, col] = p - 1
            return


        if np.random.random() < self.p.prob_migration:
            tr, tc = free[0]
            new_grid[tr, tc] = state
            new_pot[tr, tc] = self.potential[row, col]
            new_grid[row, col] = EMPTY
            new_pot[row, col] = 0

    def _step_immune(self, row, col, new_grid, new_pot, new_nec):
        """
        Функція переходу імунної клітини.

        Якщо поруч є ракова клітина то вони б'ються:
          pdT = 15%: імунна перемагає (рак -> NECROTIC)
          pdI =  5%: рак перемагає (імунна -> EMPTY)
        Якщо ракових немає -> міграція.
        """
        neighbors = get_neighbors(row, col, self.N)
        random.shuffle(neighbors)
        cancer_neighbors = [
            (r, c) for r, c in neighbors
            if self.grid[r, c] in (CANCER, CANCER_STEM)
        ]

        if cancer_neighbors:
            tr, tc = cancer_neighbors[0]
            roll = np.random.random()
            if roll < self.p.prob_immune_kill:
                new_grid[tr, tc] = NECROTIC
                new_pot[tr, tc] = 0
                new_nec[tr, tc] = 0
            elif roll < self.p.prob_immune_kill + self.p.prob_immune_death:
                new_grid[row, col] = EMPTY
        else:
            if np.random.random() < self.p.immune_migration_speed:
                free = [(r, c) for r, c in neighbors if new_grid[r, c] == EMPTY]
                if free:
                    tr, tc = free[0]
                    new_grid[tr, tc] = IMMUNE
                    new_grid[row, col] = EMPTY

    def _recruit_immune(self, new_grid):
        """
        Устворення нових імунних клітин
        """
        n_cancer = int(np.sum((new_grid == CANCER) | (new_grid == CANCER_STEM)))
        n_immune = int(np.sum(new_grid == IMMUNE))
        if n_cancer == 0:
            return

        alpha = 1000.0
        rate = self.p.immune_recruit_rate * n_immune * n_cancer / (alpha + n_cancer)
        rate = max(rate, 0.01 * n_cancer / (alpha + n_cancer))
        for _ in range(np.random.poisson(max(rate, 0.1))):
            edge = np.random.randint(4)
            if edge == 0:
                r = 0
                c = np.random.randint(self.N)
            elif edge == 1:
                r = self.N - 1
                c = np.random.randint(self.N)
            elif edge == 2:
                r = np.random.randint(self.N)
                c = 0
            else:
                r = np.random.randint(self.N)
                c = self.N - 1
            if new_grid[r, c] == EMPTY:
                new_grid[r, c] = IMMUNE

    def _record_stats(self):
        self.history['step'].append(self.step_count)
        self.history['cancer'].append(int(np.sum(self.grid == CANCER)))
        self.history['stem'].append(int(np.sum(self.grid == CANCER_STEM)))
        self.history['immune'].append(int(np.sum(self.grid == IMMUNE)))
        self.history['healthy'].append(int(np.sum(self.grid == HEALTHY)))
        self.history['necrotic'].append(int(np.sum(self.grid == NECROTIC)))

    def stats_str(self):
        """Статистика"""
        c = self.history['cancer'][-1]
        s = self.history['stem'][-1]
        i = self.history['immune'][-1]
        n = self.history['necrotic'][-1]
        return (f"Крок {self.step_count:>4d} | "
                f"RTC: {c:>5d} | STC: {s:>3d} | "
                f"Пухлина: {c+s:>5d} | "
                f"Імунні: {i:>4d} | Некроз: {n:>4d}")


def run_simulation(params=None, n_steps=200, cancer_type='stem'):
    """
    Запускає симуляцію з анімацією в терміналі.
    """
    ca = TumorCA(params or SimulationParams())
    ca.initialize(cancer_type=cancer_type)
    show_at = []
    for i in range(1, n_steps//10 + 1):
        show_at.append(i*10)
    for step in range(n_steps):
        ca.simulation_step()
        is_last = (step + 1 == n_steps)
        total   = int(np.sum((ca.grid == CANCER) | (ca.grid == CANCER_STEM)))
        if (step + 1) in show_at or is_last:
            os.system('clear')
            label = "=== ФІНАЛ ===" if is_last else f"Крок {step+1}/{n_steps}"
            print(label)
            print("· порожньо  ○ здорова  ● RTC  ★ STC  ◆ імунна  ✕ некроз\n")
            print(f"\n{ca.stats_str()}")
            if not is_last:
                time.sleep(1)

        if total == 0 and step > 10:
            print(f"\nПухлина зникла на кроці {step+1}.")
            break

    return ca


def run_simulation_from(ca, n_steps=300):
    """
    Запускає симуляцію з вже ініціалізованого стану CA.
    """
    show_at = [i * 10 for i in range(1, n_steps // 10 + 1)]

    for step in range(n_steps):
        ca.simulation_step()
        is_last = (step + 1 == n_steps)
        total = int(np.sum((ca.grid == CANCER) | (ca.grid == CANCER_STEM)))

        if (step + 1) in show_at or is_last:
            os.system('clear')
            label = "=== ФІНАЛ ===" if is_last else f"Крок {step+1}/{n_steps}"
            print(label)
            print("· порожньо  ○ здорова  ● RTC  ★ STC  ◆ імунна  ✕ некроз\n")
            print(f"\n{ca.stats_str()}")
            if not is_last:
                time.sleep(1)

        if total == 0 and step > 10:
            print(f"\nВсі пухлини зникли на кроці {step+1}.")
            break

    return ca


def scenario_nonclonogenic(n_steps=300, grid_size=80):
    """
    Пухлина через RTC
    Очікуваний результат: зникає, бо потенціал поділів вичерпується.
    """
    params = SimulationParams(
        grid_size=grid_size, prob_apoptosis=0.0,
        initial_healthy_density=0.0, initial_immune_count=0
    )
    return run_simulation(params, n_steps=n_steps, cancer_type='regular')

def scenario_stem(n_steps=300, grid_size=80):
    """
    Пухлина від стовбурової STC (Ps=0)
    Очікуваний результат: стабільна пухлина — STC безсмертно відтворює RTC.
    """
    params = SimulationParams(
        grid_size=grid_size, prob_apoptosis=0.0, prob_stem_division=0.0,
        initial_healthy_density=0.0, initial_immune_count=0
    )
    return run_simulation(params, n_steps=n_steps, cancer_type='stem')

def scenario_multi(n_tumors=5, n_steps=300, grid_size=120):
    """
    Сценарій 4: кілька пухлин рандомно по карті.
    Показує взаємодію пухлин і злиття.
    """
    params = SimulationParams(
        grid_size=grid_size,
        prob_stem_division=0.02,
        initial_healthy_density=0.2,
        initial_immune_count=30,
        prob_immune_kill=0.15
    )
    ca = TumorCA(params)
    actual = ca.initialize_multi(n_tumors=n_tumors, cancer_type='stem')
    print(f"Розміщено пухлин: {actual} з {n_tumors} запитаних")

    return run_simulation_from(ca, n_steps=n_steps)

def scenario_immune(n_steps=300, grid_size=80):
    """
    Повна модель — STC + здорова тканина + імунна система.
    Очікуваний результат: імунна система сповільнює ріст, але не зупиняє.
    """
    params = SimulationParams(
        grid_size=grid_size, prob_stem_division=0.02,
        initial_healthy_density=0.2, initial_immune_count=20,
        prob_immune_kill=0.2
    )
    return run_simulation(params, n_steps=n_steps, cancer_type='stem')

# ── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    n_steps = int(input("Кількість кроків [300]: ") or 300)
    grid_size = int(input("Розмір решітки [80]: ") or 80)

    params = SimulationParams(grid_size=grid_size)
    print("Оберіть сценарій:")
    print("  1 — RTC (пухлина зникає)")
    print("  2 — STC (стабільна пухлина)")
    print("  3 — STC + імунна система")
    print("  4 — кілька пухлин (взаємодія)")
    print("  0 — швидкий тест (50×50, 100 кроків)")

    try:
        choice = input("\nВибір [0-3]: ").strip()
    except EOFError:
        choice = '0'

    if choice == '1':
        scenario_nonclonogenic(n_steps=n_steps, grid_size=grid_size)
    elif choice == '2':
        scenario_stem(n_steps=n_steps, grid_size=grid_size)
    elif choice == '3':
        scenario_immune(n_steps=n_steps, grid_size=grid_size)
    elif choice == '4':
        n = int(input("Кількість пухлин [5]: ") or 5)
        scenario_multi(n_tumors=n, n_steps=n_steps, grid_size=grid_size)
    else:
        run_simulation(params, n_steps=100)

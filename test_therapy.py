import tumor_new
import chemotherapy
import radiotherapy
import immunotherapy

EMPTY, HEALTHY, CANCER, CANCER_STEM, IMMUNE, NECROTIC = 0, 1, 2, 3, 4, 5


def print_therapy_result(ca_object, modified_grid, title):
    """
    Використовуємо рідну функцію візуалізації з ядра (to_ascii).
    Це гарантує 100% збіг стилів і правильне центрування кадру.
    """
    print(f" \n {title}")

    original_grid = ca_object.grid.copy()
    ca_object.grid = modified_grid

    print(ca_object.to_ascii(size=30))

    ca_object.grid = original_grid

if __name__ == "__main__":
    params = tumor_new.SimulationParams(
        grid_size=60,
        prob_stem_division=0.02,
        initial_immune_count=10
    )

    ca_model = tumor_new.run_simulation(params, n_steps=100, cancer_type='stem')

    grown_grid = ca_model.grid

    print_therapy_result(ca_model, grown_grid, "1. ПУХЛИНА ДО ЛІКУВАННЯ")

    chemo_grid = chemotherapy.apply_chemotherapy(grown_grid.copy(), p_kill_rtc=0.8, p_kill_stc=0.3)
    print_therapy_result(ca_model, chemo_grid, "2. ПІСЛЯ ХІМІОТЕРАПІЇ")

    radio_grid = radiotherapy.apply_radiotherapy(grown_grid.copy(), target_r=30, target_c=30, radius=6)
    print_therapy_result(ca_model, radio_grid, "3. ПІСЛЯ РАДІОТЕРАПІЇ")

    immuno_grid = immunotherapy.apply_immunotherapy(grown_grid.copy(), boost_radius=2, p_kill_cancer=0.9, recruitment_rate=0.05)
    print_therapy_result(ca_model, immuno_grid, "4. ПІСЛЯ ІМУНОТЕРАПІЇ")

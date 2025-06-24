import math

def new_radius_func(number_of_engines_new : int):
    # For super heavy booster
    diameter_engine = 1.3
    rocket_diameter = 9
    number_of_inner_engines = 3
    number_of_middle_engines = 10
    number_of_outer_engines = 20
    number_of_engines_super_heavy = 33

    number_of_engines_added = number_of_engines_new - number_of_engines_super_heavy
    number_of_middle_engines_added = 0
    number_of_outer_engines_added = 0

    for i in range(abs(number_of_engines_added)):
        if number_of_engines_added > 0:  # Adding engines
            if i % 3 == 0 or i % 3 == 1:
                number_of_outer_engines_added += 1
            else:
                number_of_middle_engines_added += 1
        else:  # Removing engines
            if i % 3 == 0:
                number_of_middle_engines_added -= 1
            else:
                number_of_outer_engines_added -= 1

    number_of_engines_per_ring = [number_of_inner_engines,
                                  number_of_middle_engines + number_of_middle_engines_added,
                                  number_of_outer_engines + number_of_outer_engines_added]

    rocket_radius_new = rocket_diameter / 2 * (1 + number_of_outer_engines_added / number_of_outer_engines)

    frontal_area_new = math.pi * (rocket_radius_new ** 2)
    return rocket_radius_new, frontal_area_new, number_of_engines_per_ring
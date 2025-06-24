from src.classical_controls.ascent_control import AscentControl
from src.classical_controls.flip_over_and_boostbackburn_control import FlipOverandBoostbackBurnControl, FlipOverandBoostbackBurnTuning
from src.classical_controls.ballisitic_arc_descent import HighAltitudeBallisticArcDescent, BallisticArcDescentTuning
from src.classical_controls.landing_burn_pure_throttle import LandingBurn, reference_landing_trajectory
from src.classical_controls.landing_burn_pure_throttle_verify_PD import LandingBurn_PDcontrol

ascent_control = AscentControl()
ascent_control.run_closed_loop()

flip_over_and_boostbackburn_tuning = FlipOverandBoostbackBurnTuning(tune_bool=True)
flip_over_and_boostbackburn_tuning.run_closed_loop()

flip_over_and_boostbackburn_control = FlipOverandBoostbackBurnControl(pitch_tuning_bool = False)
flip_over_and_boostbackburn_control.run_closed_loop()

ballistic_arc_tuning = BallisticArcDescentTuning(tune_bool=True)
ballistic_arc_tuning.run_closed_loop()

ballistic_arc_descent = HighAltitudeBallisticArcDescent()
ballistic_arc_descent.run_closed_loop()

ref = reference_landing_trajectory()
landing_burn = LandingBurn()
landing_burn.run_closed_loop()

landing_burn_stochastic = LandingBurn(test_case = 'stochastic')
landing_burn_stochastic.run_closed_loop()

landing_burn_stochastic_v_ref = LandingBurn(test_case = 'stochastic_v_ref')
landing_burn_stochastic_v_ref.run_closed_loop()

landing_burn_v_ref_control = LandingBurn_PDcontrol(test_case = 'control')
landing_burn_v_ref_control.run_closed_loop()

landing_burn_v_ref_control_stochastic = LandingBurn_PDcontrol(test_case = 'stochastic_v_ref')
landing_burn_v_ref_control_stochastic.run_closed_loop()

landing_burn_wind = LandingBurn(test_case = 'wind')
landing_burn_wind.run_closed_loop()

def compile_rtd_rl_landing_burn(trajectory_length, discount_factor, pure_throttle = False, dt = 0.1):
    max_alpha_effective = math.radians(20)
    x_0, y_0, vx_0, vy_0, theta_0, theta_dot_0, gamma_0, alpha_0, mass_0, mass_propellant_0, time_0 = load_landing_burn_initial_state()
    def done_func_lambda(state):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
        speed = math.sqrt(vx**2 + vy**2)
        dynamic_pressure = 0.5 * density * speed**2
        if y > 0 and y < 1:
            if speed < 0.5:
                return True
            else:
                return False
        else:
            return False
    
    def truncated_func_lambda(state, previous_state, info):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        air_density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
        xp, yp, vxp, vyp, thetap, theta_dotp, gammamp, alphap, massp, mass_propellantp, timep = previous_state
        speed = math.sqrt(vx**2 + vy**2)
        dynamic_pressure = 0.5 * air_density * speed**2
        speed_p = math.sqrt(vxp**2 + vyp**2)
        speed_diff = abs(speed - speed_p)
        acceleration = speed_diff/dt * 1/9.81
        if vy < 0:
            alpha_effective = abs(gamma - theta - math.pi)
        else:
            alpha_effective = abs(theta - gamma)
        if y < -10:
            return True, 1
        elif mass_propellant <= 0:
            #print(f'Truncated due to mass_propellant <= 0, mass_propellant = {mass_propellant}, y = {y}')
            return True, 2
        elif theta > math.pi + math.radians(2):
            #print(f'Truncated due to theta > math.pi + math.radians(2), theta = {theta}, y = {y}')
            return True, 3
        elif dynamic_pressure > 65000:
            #print(f'Truncated due to dynamic pressure > 65000, dynamic_pressure = {dynamic_pressure}, y = {y}')
            return True, 4
        elif info['g_load_1_sec_window'] > 6.0:
            return True, 5
        elif vy > 0.0:
            #print(f'Truncated due to vy > 0.0, vy = {vy}, y = {y}')
            return True, 6
        else:
            return False, 0


    # ---------- constants ----------
    GAMMA          = discount_factor      # discount factor
    Q_MAX_THRES    = 60_000.0             # Pa
    Q_MAX          = 65_000.0             # Pa
    G_MAX_THRES    = 5.5                  # g
    G_MAX          = 6.0                  # g
    W_Q_PENALTY    = 1.0                  # dimensionless
    W_G_PENALTY    = 1.0
    W_PROGRESS     = 0.5                  # max ≈ 0.5 per step
    W_TERMINAL     = 105.0                  # landing bonus
    W_CRASH        = 5.0                  # scaled by impact speed & altitude
    W_MASS_USED    = 0.1                  # cumulative fuel penalty
    W_SLOWDOWN     = 1.5                  # extra reward < 100 m
    CLIP_LIMIT     = 10.0                 # final reward bound
    Y0_FIXED       = y_0               # m, initial altitude in current set-up
    VY_TARGET_NEAR = 0.0                  # m s⁻¹ target at touchdown
    # ---------------------------------

    def reward_func_lambda(state, done, truncated, actions, previous_state, info):
        # unpack state
        x,  y,  vx,     vy,     theta,  theta_dot,  gamma,      alpha,  mass,   mass_propellant,    time =  state

        # atmosphere model
        air_density, _, _ = endo_atmospheric_model(y)
        speed             = math.hypot(vx, vy)
        q                 = 0.5 * air_density * speed**2
        reward = 0.0

        # 1. dynamic-pressure penalty (quadratic, bounded)
        if q > Q_MAX_THRES:
            q_excess = (q - Q_MAX_THRES) / (Q_MAX - Q_MAX_THRES)
            reward  -= W_Q_PENALTY * min(q_excess**2, 1.0)

        # 2. g-load penalty (quadratic, bounded)
        g_load = info['g_load_1_sec_window']
        if g_load > G_MAX_THRES:
            g_excess = (g_load - G_MAX_THRES) / (G_MAX - G_MAX_THRES)
            reward  -= W_G_PENALTY * min(g_excess**2, 1.0)

        # 3. dense progress reward (scaled even when unsafe, but down-weighted)
        # Descent progress: only rewards moving closer to ground
        altitude_progress = (Y0_FIXED - y) / Y0_FIXED
        w_prog = W_PROGRESS if (q <= Q_MAX_THRES and g_load <= G_MAX_THRES) else W_PROGRESS * 0.1
        reward += w_prog * altitude_progress


        # 4. slowdown shaping < 100 m
        if y < 10.0: # Added after 013947
            reward += 2 - 2*math.tanh((speed-10)/10) # Added
        elif y < 100.0:
            slowdown_reward = max(0.0, 1.0 - abs(vy - VY_TARGET_NEAR) / 50.0)
            reward         += W_SLOWDOWN * slowdown_reward

        # 6. terminal handling
        if done and not truncated:
            reward  += W_TERMINAL * mass_propellant/mass_0
        elif truncated:
            altitude_factor  = y / Y0_FIXED
            reward          -= min(W_CRASH * altitude_factor, 5.0)

        # 7. final clipping to ±10
        reward = np.clip(reward, -CLIP_LIMIT, CLIP_LIMIT)

        # N-step rewards scaling
        #reward *= (1 - discount_factor)/(1 - discount_factor**trajectory_length)


        return reward
   
    return reward_func_lambda, truncated_func_lambda, done_func_lambda
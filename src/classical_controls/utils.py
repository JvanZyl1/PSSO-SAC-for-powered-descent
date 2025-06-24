def PD_controller_single_step(Kp, Kd, N, error, previous_error, previous_derivative, dt):
    P_term = Kp * error
    derivative = (error - previous_error) / dt
    D_term = Kd * (N * derivative + (1 - N * dt) * previous_derivative)
    control_action = P_term + D_term
    return control_action, derivative

class PIDController:
    def __init__(self, Kp, Ki, Kd, N, dt, output_limits=None, previous_error=0.0, previous_derivative=0.0, initial_integral=0.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.N = N
        self.dt = dt
        self.output_limits = output_limits
        
        self.previous_error = previous_error
        self.previous_derivative = previous_derivative
        self.integral = initial_integral
    
    def reset(self, previous_error=0.0, previous_derivative=0.0, initial_integral=0.0):

        self.previous_error = previous_error
        self.previous_derivative = previous_derivative
        self.integral = initial_integral
    
    def step(self, error):
        P_term = self.Kp * error
        self.integral += error * self.dt
        I_term = self.Ki * self.integral
        
        derivative = (error - self.previous_error) / self.dt
        D_term = self.Kd * (self.N * derivative + (1 - self.N * self.dt) * self.previous_derivative)
        
        control_action = P_term + I_term + D_term
        
        if self.output_limits is not None:
            control_action = max(min(control_action, self.output_limits[1]), self.output_limits[0])
        
        self.previous_error = error
        self.previous_derivative = derivative
        
        return control_action
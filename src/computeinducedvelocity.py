import numpy as np  
import openmdao.api as om



class ComputeInducedVelocity(om.ExplicitComponent):
    """
    Compute the power required for a propeller.

    References
    [1] GA Guddmundsson. "General Aviation Aircraft Design".
    """

    def initialize(self):

        self.options.declare('n', default=1, desc='number of data points')

    def setup(self):

        # Inputs    
        self.add_input('rho', val= np.ones(self.options['n']), desc='air density', units='kg/m**3')
        self.add_input('total_thrust_req', val= np.ones(self.options['n']), desc='total aircraft thrust required', units='N')
        self.add_input('vel', val= np.ones(self.options['n']), desc='true airspeed', units='m/s')
        self.add_input('d_blade', val=0, desc='blade diameter', units='m')
        self.add_input('d_hub', val=0, desc='hub diameter', units='m')
        self.add_input('num_motors', val=1, desc='number of engines', units=None)

        # Outputs
        self.add_output('v_ind', val= np.ones(self.options['n']), desc='induced velocity', units='m/s', lower=1e-3)


    def setup_partials(self):

        # Partials
        self.declare_partials('v_ind', 'total_thrust_req')
        self.declare_partials('v_ind', 'vel')
        self.declare_partials('v_ind', 'd_blade')
        self.declare_partials('v_ind', 'd_hub')
        self.declare_partials('v_ind', 'rho')
        self.declare_partials('v_ind', 'num_motors')

    def compute(self, inputs, outputs):

        # Unpack inputs    
        rho = inputs['rho']
        total_thrust_req = inputs['total_thrust_req']
        vel = inputs['vel']
        d_blade = inputs['d_blade']
        d_hub = inputs['d_hub']
        num_motors = inputs['num_motors']

        diskarea = np.pi * ((d_blade/2)**2 - (d_hub/2)**2)

        unit_thrust_req = total_thrust_req / num_motors 

        rho = np.maximum(rho, 1e-6)
        unit_thrust_req = np.maximum(unit_thrust_req, 1e-6)
        vel = np.maximum(vel, 1e-6)

        # Compute induced airspeed [1] Eq 15-76
        v_ind = 0.5 * ( - vel + ( vel**2 + 2*unit_thrust_req / (0.5 * rho * diskarea) ) **(0.5) )

        #print('vel (m/s) = ', vel)
        #print('unit_thrust_req (N) = ', unit_thrust_req)



        # Pack outputs
        outputs['v_ind'] = v_ind

    def compute_partials(self, inputs, J):  

        # Unpack inputs
        vel = inputs['vel']
        d_blade = inputs['d_blade']
        d_hub = inputs['d_hub']
        rho = inputs['rho']
        total_thrust_req = inputs['total_thrust_req']
        num_motors = inputs['num_motors']

        # Unpack constants
        n = self.options['n']

        diskarea = np.pi * ((d_blade/2)**2 - (d_hub/2)**2)
        d_blade_d_diskarea = np.pi * d_blade / 2
        d_hub_d_diskarea = -np.pi * d_hub / 2
        unit_thrust_req = total_thrust_req / num_motors  


        J['v_ind', 'total_thrust_req'] = np.eye(n) * (0.5 * 0.5 * ( vel**2 + unit_thrust_req / (0.5 * rho * diskarea) ) **(-0.5) * ( 1 / (0.5 * rho * diskarea * num_motors) ) )
        J['v_ind', 'vel'] = np.eye(n) * 0.5*(-1 + 0.5*( vel**2 + 2*unit_thrust_req / (0.5 * rho * diskarea))**(-0.5) * 2 )
        J['v_ind', 'd_blade'] =  (0.5 * 0.5 * ( vel**2 + unit_thrust_req / (0.5 * rho * diskarea) ) **(-0.5) * (  unit_thrust_req / (0.5 * rho * diskarea * num_motors) ) * -diskarea ** (-2) * d_blade_d_diskarea )
        J['v_ind', 'd_hub'] = (0.5 * 0.5 * ( vel**2 + unit_thrust_req / (0.5 * rho * diskarea) ) **(-0.5) * ( -unit_thrust_req / (0.5 * rho * diskarea * num_motors) ) * -diskarea ** (-2) * d_hub_d_diskarea )
        J['v_ind', 'rho'] = np.eye(n) * (0.5 * 0.5 * ( vel**2 + unit_thrust_req / (0.5 * rho * diskarea) ) **(-0.5) * ( -unit_thrust_req / (0.5 * rho **2 * diskarea * num_motors) ) )
        J['v_ind', 'num_motors'] = (0.5 * 0.5 * ( vel**2 + unit_thrust_req / (0.5 * rho * diskarea) ) **(-0.5) * ( -unit_thrust_req / (0.5 * rho * diskarea * num_motors **2) ) )


if __name__ == "__main__":
    import openmdao.api as om

    p = om.Problem()
    model = p.model

    n = 10

    ivc = om.IndepVarComp()
    ivc.add_output('d_blade', 1.5, units='m')
    ivc.add_output('d_hub', 0.5, units='m')
    ivc.add_output('rho', 1.225 * np.ones(n), units='kg/m**3')
    ivc.add_output('total_thrust_req', 5000 * np.ones(n), units='N')
    ivc.add_output('num_motors', 2, units=None)
    ivc.add_output('vel', 100 * np.ones(n), units='m/s')


    model.add_subsystem('Indeps', ivc, promotes_outputs=['*'])
    model.add_subsystem('ComputeInducedVelocity', ComputeInducedVelocity(n = n), promotes_inputs=['*'])

    model.nonlinear_solver = om.NewtonSolver()
    model.linear_solver = om.DirectSolver()

    model.nonlinear_solver.options['iprint'] = 2
    model.nonlinear_solver.options['maxiter'] = 200
    model.nonlinear_solver.options['solve_subsystems'] = True
    model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()

    p.setup()

    #om.n2(p)
    p.run_model()

    print('v_ind = ', p['ComputeInducedVelocity.v_ind'])


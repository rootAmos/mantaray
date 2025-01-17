import numpy as np  
import openmdao.api as om



class ComputeFanPowerReq(om.ExplicitComponent):
    """
    Compute the power required for a propeller.

    References
    [1] GA Guddmundsson. "General Aviation Aircraft Design".
    """

    def initialize(self):
        self.options.declare('n', default=1, desc='number of data points')

    def setup(self):

        # Inputs    
        self.add_input('d_blade', val=1, desc='blade diameter', units='m')
        self.add_input('d_hub', val=1, desc='hub diameter', units='m')
        self.add_input('rho', val= np.ones(self.options['n']), desc='air density', units='kg/m**3')
        self.add_input('total_thrust_req', val= np.ones(self.options['n']), desc='total aircraft thrust required', units='N')
        self.add_input('num_motors', val=1, desc='number of engines', units=None)
        self.add_input('vel', val= np.ones(self.options['n']), desc='true airspeed', units='m/s')
        self.add_input('epsilon_r', val=1, desc='expansion ratio', units=None)
        self.add_input('eta_fan', val=1, desc='fan efficiency', units=None)
        self.add_input('eta_duct', val=1, desc='duct efficiency', units=None)

        # Outputs
        self.add_output('unit_shaft_pow', val=0, desc='power required per engine', units='W')


        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):

        # Unpack inputs
        d_blade = inputs['d_blade']
        d_hub = inputs['d_hub']
        rho = inputs['rho']
        total_thrust_req = inputs['total_thrust_req']
        num_motors = inputs['num_motors']
        vel = inputs['vel']
        epsilon_r = inputs['epsilon_r']
        eta_fan = inputs['eta_fan']
        eta_duct = inputs['eta_duct']

        unit_thrust_req = total_thrust_req / num_motors  
        diskarea = np.pi * ((d_blade/2)**2 - (d_hub/2)**2)

        # Compute the power required [1] Eq 15-90
        unit_propulsive_pow_req = 3/4 * vel * unit_thrust_req + np.sqrt((unit_thrust_req **2 * vel**2 / 4 **2) + ( unit_thrust_req**3/ ( 4* rho * diskarea  * epsilon_r)))

        # Compute induced airspeed [1] Eq 15-91
        v_ind = ( 0.5 * epsilon_r - 1)* vel + np.sqrt( (vel * epsilon_r /2)**2 + epsilon_r * unit_thrust_req  / (rho * diskarea) )
        
        # Compute station 3 velocity [1] Eq 15-87
        v3 = vel +  v_ind

        # Compute propulsive efficiency [1] Eq 15-77
        eta_prplsv = 2 / (1 + v3/vel)    

        # Compute the power required [1] Eq 15-78
        unit_shaft_pow = unit_propulsive_pow_req / eta_fan / eta_duct / eta_prplsv

        # Pack outputs
        outputs['unit_shaft_pow'] = unit_shaft_pow


if __name__ == "__main__":
    import openmdao.api as om

    p = om.Problem()
    model = p.model

    ivc = om.IndepVarComp()
    ivc.add_output('d_blade', 1.5, units='m')
    ivc.add_output('d_hub', 0.5, units='m')
    ivc.add_output('rho', 1.225, units='kg/m**3')
    ivc.add_output('total_thrust_req', 10000, units='N')
    ivc.add_output('num_motors', 2, units=None)
    ivc.add_output('vel', 100, units='m/s')
    ivc.add_output('epsilon_r', 1.5, units=None) 
    ivc.add_output('eta_fan', 0.9, units=None)
    ivc.add_output('eta_duct', 0.9, units=None)


    model.add_subsystem('Indeps', ivc, promotes_outputs=['*'])
    model.add_subsystem('ComputePropellerPowerReq',     (), promotes_inputs=['*'])

    model.nonlinear_solver = om.NewtonSolver()
    model.linear_solver = om.DirectSolver()

    model.nonlinear_solver.options['iprint'] = 2
    model.nonlinear_solver.options['maxiter'] = 200
    model.nonlinear_solver.options['solve_subsystems'] = True
    model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()

    p.setup()

    #om.n2(p)
    p.run_model()

    print('unit_shaft_pow = ', p['ComputePropellerPowerReq.unit_shaft_pow'])


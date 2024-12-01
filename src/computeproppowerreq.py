import numpy as np  
import openmdao.api as om



class ComputePropPowerReq(om.ExplicitComponent):
    """
    Compute the power required for a propeller.

    References
    [1] GA Guddmundsson. "General Aviation Aircraft Design".
    """

    def initialize(self):

        self.options.declare('n', default=1, desc='number of data points')

       

    def setup(self):

        # Inputs    
        self.add_input('d_blade', val=0, desc='blade diameter', units='m')
        self.add_input('d_hub', val=0, desc='hub diameter', units='m')
        self.add_input('rho', val= np.ones(self.options['n']), desc='air density', units='kg/m**3')
        self.add_input('total_thrust_req', val= np.ones(self.options['n']), desc='total aircraft thrust required', units='N')
        self.add_input('num_motors', val=1, desc='number of engines', units=None)
        self.add_input('utas', val= np.ones(self.options['n']), desc='true airspeed', units='m/s')
        self.options.declare('eta_prop', default=0.8, desc='propeller efficiency', units=None) 
        # Outputs
        self.add_output('unit_shaft_pow', val= np.ones(self.options['n']), desc='power required per engine', units='W')


        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):

        # Unpack inputs    
        eta_prop = inputs['eta_prop']
        d_blade = inputs['d_blade']
        d_hub = inputs['d_hub']
        rho = inputs['rho']
        total_thrust_req = inputs['total_thrust_req']
        num_motors = inputs['num_motors']
        utas = inputs['utas']

        unit_thrust_req = total_thrust_req / num_motors  
        diskarea = np.pi * ((d_blade/2)**2 - (d_hub/2)**2)

        # Compute the power required [1] Eq 15-75
        unit_propulsive_pow_req = unit_thrust_req * utas  + unit_thrust_req ** 1.5/ np.sqrt(2 * rho * diskarea)

        # Compute induced airspeed [1] Eq 15-76
        v_ind = 0.5 * ( - utas + np.sqrt( utas**2 + unit_thrust_req / (0.5 * rho * diskarea) ) )
        
        # Compute station 3 velocity [1] Eq 15-73
        v3 = utas + 2 * v_ind

        # Compute propulsive efficiency [1] Eq 15-77
        eta_prplsv = 2 / (1 + v3/utas)    

        # Compute the power required [1] Eq 15-78
        unit_shaft_pow = unit_propulsive_pow_req / eta_prop / eta_prplsv

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


    model.add_subsystem('Indeps', ivc, promotes_outputs=['*'])
    model.add_subsystem('ComputePropellerPowerReq', ComputePropellerPowerReq(), promotes_inputs=['*'])

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


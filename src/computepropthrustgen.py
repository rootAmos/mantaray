import numpy as np  
import openmdao.api as om



class ComputePropThrustGen(om.ImplicitComponent):
    """
    Compute the thrust produced by a propeller.

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
        self.add_input('unit_shaft_pow_gen', val= np.ones(self.options['n']), desc='power generated per engine', units='W')
        self.add_input('num_engines', val=1, desc='number of engines', units=None)
        self.add_input('vtas', val=np.ones(self.options['n']), desc='true airspeed', units='m/s')
        self.add_input('eta_prop', val=0.8, desc='propeller efficiency', units=None)

        # Outputs
        self.add_output('total_thrust_gen', val= 1e3* np.ones(self.options['n']), desc='total aircraft thrust generated', units='N')


        self.declare_partials('*', '*', method='fd')

    def apply_nonlinear(self, inputs, outputs, residuals):

        # Unpack inputs
        eta_prop = inputs['eta_prop']
        d_blade = inputs['d_blade']
        d_hub = inputs['d_hub']
        rho = inputs['rho']
        num_engines = inputs['num_engines']
        vtas = inputs['vtas']
        unit_shaft_pow_gen = inputs['unit_shaft_pow_gen']

        # Unpack outputs
        total_thrust_gen = outputs['total_thrust_gen']

        unit_thrust_gen = total_thrust_gen / num_engines  
        
        diskarea = np.pi * ((d_blade/2)**2 - (d_hub/2)**2)

        # Compute the power required [1] Eq 15-75
        unit_propulsive_pow_req = unit_thrust_gen * vtas  + unit_thrust_gen ** 1.5/ np.sqrt(2 * rho * diskarea)

        # Compute induced airspeed [1] Eq 15-76
        v_ind = 0.5 * ( - vtas + np.sqrt( vtas**2 + unit_thrust_gen / (0.5 * rho * diskarea) ) )
        
        # Compute station 3 velocity [1] Eq 15-73
        v3 = vtas + 2 * v_ind

        # Compute propeller efficiency [1] Eq 15-77
        eta_prplsv = 2 / (1 + v3/vtas)    

        # Compute the power required [1] Eq 15-78
        unit_shaft_pow_req = unit_propulsive_pow_req / eta_prop / eta_prplsv

        residuals['total_thrust_gen'] = unit_shaft_pow_gen - unit_shaft_pow_req


if __name__ == "__main__":
    import openmdao.api as om

    p = om.Problem()
    model = p.model

    ivc = om.IndepVarComp()
    ivc.add_output('d_blade', 1.5, units='m')
    ivc.add_output('d_hub', 0.5, units='m')
    ivc.add_output('rho', 1.225, units='kg/m**3')
    ivc.add_output('unit_shaft_pow_gen', 500e3, units='W')
    ivc.add_output('num_engines', 2, units=None)
    ivc.add_output('vtas', 100, units='m/s')



    model.add_subsystem('Indeps', ivc, promotes_outputs=['*'])
    model.add_subsystem('ComputePropThrustGen', ComputePropThrustGen(), promotes_inputs=['*'])

    model.nonlinear_solver = om.NewtonSolver()
    model.linear_solver = om.DirectSolver()

    model.nonlinear_solver.options['iprint'] = 2
    model.nonlinear_solver.options['maxiter'] = 200
    model.nonlinear_solver.options['solve_subsystems'] = True
    model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()

    p.setup()

    #om.n2(p)
    p.run_model()

    print('total_thrust_gen = ', p['ComputePropThrustGen.total_thrust_gen'])

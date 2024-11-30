import numpy as np  
import openmdao.api as om



class ComputeFanThrustGen(om.ImplicitComponent):
    """
    Compute the thrust generated for a fan.

    References
    [1] GA Guddmundsson. "General Aviation Aircraft Design".
    """

    def initialize(self):

        self.options.declare('eta_fan', default=0.9, desc='fan efficiency')
        self.options.declare('eta_duct', default=0.9, desc='duct efficiency')
       

    def setup(self):

        # Inputs    
        self.add_input('d_blade', val=0, desc='blade diameter', units='m')
        self.add_input('d_hub', val=0, desc='hub diameter', units='m')
        self.add_input('rho', val=0, desc='air density', units='kg/m**3')
        self.add_input('unit_shaft_pow_gen', val=0, desc='power supplied per engine', units='W')
        self.add_input('num_engines', val=0, desc='number of engines', units=None)
        self.add_input('vtas', val=0, desc='true airspeed', units='m/s')
        self.add_input('epsilon_r', val=0, desc='expansion ratio', units=None)

        # Outputs
        self.add_output('total_thrust_gen', val=0, desc='power required per engine', units='W')


        self.declare_partials('*', '*', method='fd')

    def apply_nonlinear(self, inputs, outputs, residuals):

        # Unpack options
        eta_fan  = self.options['eta_fan']
        eta_duct = self.options['eta_duct']
    
        # Unpack inputs
        d_blade = inputs['d_blade']
        d_hub = inputs['d_hub']
        rho = inputs['rho']
        unit_shaft_pow_gen = inputs['unit_shaft_pow_gen']
        num_engines = inputs['num_engines']
        vtas = inputs['vtas']
        epsilon_r = inputs['epsilon_r']

        total_thrust_gen = outputs['total_thrust_gen']

        unit_thrust_req = total_thrust_gen / num_engines  
        diskarea = np.pi * ((d_blade/2)**2 - (d_hub/2)**2)

        # Compute the power required [1] Eq 15-90
        unit_propulsive_pow_req = 3/4 * vtas * unit_thrust_req + np.sqrt((unit_thrust_req **2 * vtas**2 / 4 **2) + ( unit_thrust_req**3/ ( 4* rho * diskarea  * epsilon_r)))

        # Compute induced airspeed [1] Eq 15-91
        v_ind = ( 0.5 * epsilon_r - 1)* vtas + np.sqrt( (vtas * epsilon_r /2)**2 + epsilon_r * unit_thrust_req  / (rho * diskarea) )
        
        # Compute station 3 velocity [1] Eq 15-87
        v3 = vtas +  v_ind

        # Compute propulsive efficiency [1] Eq 15-77
        eta_prplsv = 2 / (1 + v3/vtas)    

        # Compute the power required [1] Eq 15-78
        unit_shaft_pow_req = unit_propulsive_pow_req / eta_fan / eta_duct / eta_prplsv

        # Pack outputs
        residuals['total_thrust_gen'] = unit_shaft_pow_req - unit_shaft_pow_gen


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
    ivc.add_output('epsilon_r', 1.5, units=None) 


    model.add_subsystem('Indeps', ivc, promotes_outputs=['*'])
    model.add_subsystem('ComputeFanThrustGen',ComputeFanThrustGen(), promotes_inputs=['*'])

    model.nonlinear_solver = om.NewtonSolver()
    model.linear_solver = om.DirectSolver()

    model.nonlinear_solver.options['iprint'] = 2
    model.nonlinear_solver.options['maxiter'] = 200
    model.nonlinear_solver.options['solve_subsystems'] = True
    model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()

    p.setup()

    #om.n2(p)
    p.run_model()

    print('total_thrust_gen = ', p['ComputeFanThrustGen.total_thrust_gen'])


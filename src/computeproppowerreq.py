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
        self.add_input('vel', val= np.ones(self.options['n']), desc='true airspeed', units='m/s')
        self.add_input('eta_prop', val=0.8, desc='propeller efficiency', units=None) 
        self.add_input('eta_prplsv', val= np.ones(self.options['n']), desc='propeller induced velocity efficiency', units=None)

        # Outputs
        self.add_output('unit_shaft_pow', val= np.ones(self.options['n']), desc='power required per engine', units='W')


    def setup_partials(self):
        self.declare_partials('unit_shaft_pow', 'total_thrust_req')
        self.declare_partials('unit_shaft_pow', 'vel')
        self.declare_partials('unit_shaft_pow', 'd_blade')
        self.declare_partials('unit_shaft_pow', 'd_hub')
        self.declare_partials('unit_shaft_pow', 'rho')
        self.declare_partials('unit_shaft_pow', 'num_motors')
        self.declare_partials('unit_shaft_pow', 'eta_prop')
        self.declare_partials('unit_shaft_pow', 'eta_prplsv')

    def compute(self, inputs, outputs):

        # Unpack inputs    
        eta_prop = inputs['eta_prop']
        d_blade = inputs['d_blade']
        d_hub = inputs['d_hub']
        rho = inputs['rho']
        total_thrust_req = inputs['total_thrust_req']
        num_motors = inputs['num_motors']
        vel = inputs['vel']
        eta_prplsv = inputs['eta_prplsv']

        unit_thrust_req = total_thrust_req / num_motors  
        diskarea = np.pi * ((d_blade/2)**2 - (d_hub/2)**2)

        # Compute the power required [1] Eq 15-75
        unit_propulsive_pow_req = unit_thrust_req * vel  + unit_thrust_req ** 1.5/ np.sqrt(2 * rho * diskarea)
   

        # Compute the power required [1] Eq 15-78
        unit_shaft_pow = unit_propulsive_pow_req / eta_prop / eta_prplsv

        # Pack outputs
        outputs['unit_shaft_pow'] = unit_shaft_pow

    def compute_partials(self, inputs, J):  

        # Unpack inputs
        vel = inputs['vel']
        d_blade = inputs['d_blade']
        d_hub = inputs['d_hub']
        rho = inputs['rho']
        total_thrust_req = inputs['total_thrust_req']
        num_motors = inputs['num_motors']
        eta_prop = inputs['eta_prop']
        eta_prplsv = inputs['eta_prplsv']

        # Unpack constants
        n = self.options['n']

        diskarea = np.pi * ((d_blade/2)**2 - (d_hub/2)**2)
        d_blade_d_diskarea = np.pi * d_blade / 2
        d_hub_d_diskarea = -np.pi * d_hub / 2
        unit_thrust_req = total_thrust_req / num_motors  

        unit_propulsive_pow_req = unit_thrust_req * vel  + unit_thrust_req ** 1.5/ np.sqrt(2 * rho * diskarea)


        J['unit_shaft_pow', 'total_thrust_req'] = np.eye(n) * (vel + 1.5 * total_thrust_req**0.5 / np.sqrt(2 * rho * diskarea)) / num_motors / eta_prop / eta_prplsv
        J['unit_shaft_pow', 'vel'] = np.eye(n) * unit_thrust_req / eta_prop / eta_prplsv
        J['unit_shaft_pow', 'd_blade'] = unit_thrust_req ** 1.5 * (-0.5)*  np.sqrt(2 * rho *diskarea)**(-1.5) * d_blade_d_diskarea/ eta_prop / eta_prplsv
        J['unit_shaft_pow', 'd_hub'] = unit_thrust_req ** 1.5 * (-0.5)*  np.sqrt(2 * rho *diskarea)**(-1.5) * d_hub_d_diskarea/ eta_prop / eta_prplsv
        J['unit_shaft_pow', 'rho'] = np.eye(n) * unit_thrust_req ** 1.5/ np.sqrt(2 * rho * diskarea)*(-1.5) * ( 2 * diskarea) / eta_prop / eta_prplsv
        J['unit_shaft_pow', 'eta_prop'] =  -unit_propulsive_pow_req / eta_prop**2 / eta_prplsv
        J['unit_shaft_pow', 'eta_prplsv'] = np.eye(n) * -unit_propulsive_pow_req / eta_prop / eta_prplsv**2
        J['unit_shaft_pow', 'num_motors'] =  -vel/ eta_prop / eta_prplsv / num_motors**2  +  0.5 * num_motors ** (-0.5) * unit_thrust_req ** 1.5/ np.sqrt(2 * rho * diskarea) / eta_prop / eta_prplsv


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
    ivc.add_output('eta_prop', 0.9 , units=None)


    model.add_subsystem('Indeps', ivc, promotes_outputs=['*'])
    model.add_subsystem('ComputePropPowerReq', ComputePropPowerReq(n = n), promotes_inputs=['*'])

    model.nonlinear_solver = om.NewtonSolver()
    model.linear_solver = om.DirectSolver()

    model.nonlinear_solver.options['iprint'] = 2
    model.nonlinear_solver.options['maxiter'] = 200
    model.nonlinear_solver.options['solve_subsystems'] = True
    model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()

    p.setup()

    #om.n2(p)
    p.run_model()

    print('unit_shaft_pow = ', p['ComputePropPowerReq.unit_shaft_pow'])


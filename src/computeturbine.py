import numpy as np  
import openmdao.api as om



class ComputeTurbine(om.ExplicitComponent):
    """
    Compute the fuel consumption.

    References
    [1] GA Guddmundsson. "General Aviation Aircraft Design".
    """

    def initialize(self):
        self.options.declare('n', default=1, desc='number of data points')

       

    def setup(self):

        # Inputs    
        self.add_input('unit_shaft_pow', val= np.ones(self.options['n']), desc='power required per engine', units='W')
        self.add_input('num_motors', val=1, desc='number of engines', units=None)
        self.add_input('num_turbines', val=1, desc='number of turbines', units=None)
        self.add_input('dt', val=np.ones(self.options['n']), desc='time step', units='s')
        self.add_input('psfc', val=np.ones(self.options['n']), desc='power specific fuel consumption', units='kg/s/W')
        self.add_input('eta_gen', val=1, desc='generator efficiency', units=None)
        self.add_input('eta_cbl', val=1, desc='cable efficiency', units=None)
        self.add_input('eta_pe', val=1, desc='power electronics efficiency', units=None)
        self.add_input('eta_motor', val=1, desc='motor efficiency', units=None)
        self.add_input('hy', val=np.ones(self.options['n']), desc='hybridization ratio', units=None)

        # Outputs
        self.add_output('obj_func', val= 1, desc='objective function. negative of total fuel consumption', units='kg')


    def setup_partials(self):
        self.declare_partials('obj_func', '*')

    def compute(self, inputs, outputs):

        # Unpack inputs
        unit_shaft_pow = inputs['unit_shaft_pow']
        num_motors = inputs['num_motors']
        num_turbines = inputs['num_turbines']
        dt = inputs['dt']
        psfc = inputs['psfc']
        eta_gen = inputs['eta_gen']
        eta_cbl = inputs['eta_cbl']
        eta_pe = inputs['eta_pe']
        eta_motor = inputs['eta_motor']
        hy = inputs['hy']

        # Turbine power required (W)
        unit_turbine_pow = unit_shaft_pow * num_motors / eta_gen / eta_cbl / eta_pe / eta_motor *  hy / num_turbines

        # Fuel consumption (kg)
        fuel_burn = np.cumsum(unit_turbine_pow * dt * psfc) * num_turbines

        outputs['obj_func'] = fuel_burn[-1]

    def compute_partials(self, inputs, J):

        # Unpack inputs
        unit_shaft_pow = inputs['unit_shaft_pow']
        dt = inputs['dt']
        psfc = inputs['psfc']
        num_turbines = inputs['num_turbines']
        num_motors = inputs['num_motors']
        eta_gen = inputs['eta_gen']
        eta_cbl = inputs['eta_cbl']
        eta_pe = inputs['eta_pe']
        eta_motor = inputs['eta_motor']
        hy = inputs['hy']

        # Compute partial derivatives
        J['obj_func', 'unit_shaft_pow'] = np.sum(dt * psfc) * num_turbines * num_motors / eta_gen / eta_cbl / eta_pe / eta_motor * hy / num_turbines   
        J['obj_func', 'dt'] = np.sum(unit_shaft_pow * psfc) * num_turbines * num_motors / eta_gen / eta_cbl / eta_pe / eta_motor * hy / num_turbines
        J['obj_func', 'psfc'] = np.sum(unit_shaft_pow * dt) * num_turbines * num_motors / eta_gen / eta_cbl / eta_pe / eta_motor * hy / num_turbines
        J['obj_func', 'num_turbines'] =0 
        J['obj_func', 'num_motors'] = np.sum(unit_shaft_pow * dt * psfc) * num_turbines / eta_gen / eta_cbl / eta_pe / eta_motor * hy 
        J['obj_func', 'eta_gen'] = -np.sum(unit_shaft_pow * dt * psfc) * num_turbines * num_motors / eta_gen**2 / eta_cbl / eta_pe / eta_motor * hy / num_turbines
        J['obj_func', 'eta_cbl'] = -np.sum(unit_shaft_pow * dt * psfc) * num_turbines * num_motors / eta_gen / eta_cbl**2 / eta_pe / eta_motor * hy / num_turbines
        J['obj_func', 'eta_pe'] = -np.sum(unit_shaft_pow * dt * psfc) * num_turbines * num_motors / eta_gen / eta_cbl / eta_pe**2 / eta_motor * hy / num_turbines
        J['obj_func', 'eta_motor'] = -np.sum(unit_shaft_pow * dt * psfc) * num_turbines * num_motors / eta_gen / eta_cbl / eta_pe / eta_motor**2 * hy / num_turbines
        J['obj_func', 'hy'] = np.sum(unit_shaft_pow * dt * psfc) * num_turbines / eta_gen / eta_cbl / eta_pe / eta_motor * hy / num_turbines


if __name__ == "__main__":
    import openmdao.api as om

    p = om.Problem()
    model = p.model

    ivc = om.IndepVarComp()
    ivc.add_output('unit_shaft_pow', 1000, units='W')
    ivc.add_output('num_motors', 2, units=None)
    ivc.add_output('dt', 1, units='s')
    ivc.add_output('psfc', 0.001, units='kg/s/W')
    ivc.add_output('eta_gen', 0.93, units=None)
    ivc.add_output('eta_cbl', 0.93, units=None)
    ivc.add_output('eta_pe', 0.93, units=None)
    ivc.add_output('eta_motor', 0.93, units=None)
    ivc.add_output('hy', 0.5, units=None)

    model.add_subsystem('Indeps', ivc, promotes_outputs=['*'])
    model.add_subsystem('ComputeTurbine', ComputeTurbine(), promotes_inputs=['*'])

    model.nonlinear_solver = om.NewtonSolver()
    model.linear_solver = om.DirectSolver()

    model.nonlinear_solver.options['iprint'] = 2
    model.nonlinear_solver.options['maxiter'] = 200
    model.nonlinear_solver.options['solve_subsystems'] = True
    model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()

    p.setup()

    #om.n2(p)
    p.run_model()

    print('fuel_consumption = ', p['ComputeTurbine.obj_func'])


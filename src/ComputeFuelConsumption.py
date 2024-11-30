import numpy as np  
import openmdao.api as om



class ComputeTurbine(om.ExplicitComponent):
    """
    Compute the fuel consumption.

    References
    [1] GA Guddmundsson. "General Aviation Aircraft Design".
    """

    def initialize(self):
        pass

       

    def setup(self):

        # Inputs    
        self.add_input('unit_shaft_pow_req', val=0, desc='power required per engine', units='W')
        self.add_input('num_engines', val=0, desc='number of engines', units=None)
        self.add_input('dt', val=0, desc='time step', units='s')
        self.add_input('psfc', val=0, desc='power specific fuel consumption', units='kg/s/W')
        self.add_input('eta_gen', val=0, desc='generator efficiency', units=None) 
        self.add_input('eta_cbl', val=0, desc='cable efficiency', units=None)
        self.add_input('eta_pe', val=0, desc='power electronics efficiency', units=None)
        self.add_input('eta_motor', val=0, desc='motor efficiency', units=None)
        self.add_input('hy', val=0, desc='hybridization ratio', units=None)

        # Outputs
        self.add_output('fuel_consumption', val=0, desc='fuel consumption', units='kg')
        self.add_output('turbine_pow_req', val=0, desc='turbine power required', units='W')


        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):

        # Unpack inputs
        unit_shaft_pow_req = inputs['unit_shaft_pow_req']
        num_engines = inputs['num_engines']
        dt = inputs['dt']
        psfc = inputs['psfc']
        eta_gen = inputs['eta_gen']
        eta_cbl = inputs['eta_cbl']
        eta_pe = inputs['eta_pe']
        eta_motor = inputs['eta_motor']
        hy = inputs['hy']

        # Turbine power required (W)
        turbine_pow_req = unit_shaft_pow_req * num_engines / eta_gen / eta_cbl / eta_pe / eta_motor * (1 - hy)

        # Fuel consumption (kg)
        fuel_consumption = np.cumsum(turbine_pow_req * dt * psfc)

        outputs['fuel_consumption'] = fuel_consumption    
        outputs['turbine_pow_req'] = turbine_pow_req


if __name__ == "__main__":
    import openmdao.api as om

    p = om.Problem()
    model = p.model

    ivc = om.IndepVarComp()
    ivc.add_output('unit_shaft_pow_req', 1000, units='W')
    ivc.add_output('num_engines', 2, units=None)
    ivc.add_output('dt', 1, units='s')
    ivc.add_output('psfc', 0.001, units='kg/s/W')
    ivc.add_output('eta_gen', 0.93, units=None)
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

    print('fuel_consumption = ', p['ComputeTurbine.fuel_consumption'])
    print('turbine_pow_req = ', p['ComputeTurbine.turbine_pow_req'])


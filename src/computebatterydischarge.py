import numpy as np  
import openmdao.api as om



class ComputeBatteryDischarge(om.ExplicitComponent):
    """
    Compute the battery discharge.

    References
    [1] GA Guddmundsson. "General Aviation Aircraft Design".
    """

    def initialize(self):
        self.options.declare('n', default=1, desc='number of data points')
       

    def setup(self):

        # Inputs    
        self.add_input('unit_shaft_pow', val= np.ones(self.options['n']), desc='power required per engine', units='W')
        self.add_input('num_motors', val=1, desc='number of engines', units=None)
        self.add_input('dt', val= np.ones(self.options['n']), desc='time step', units='s')
        self.add_input('soc_0', val= 1, desc='initial battery state of charge', units=None)
        self.add_input('batt_cap', val= 1, desc='battery capacity', units='J')
        self.add_input('eta_batt', val= 1, desc='battery efficiency', units=None) 
        self.add_input('eta_motor', val= 1, desc='motor efficiency', units=None)
        self.add_input('eta_pe', val=1, desc='power electronics efficiency', units=None)
        self.add_input('eta_cbl', val=1, desc='cables efficiency', units=None)
        self.add_input('hy', val=np.ones(self.options['n']), desc='hybridization ratio', units=None)

        # Outputs
        self.add_output('soc', val=np.ones(self.options['n']), desc='battery state of charge', units=None)


        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):

        # Unpack Inputs
        eta_batt = inputs['eta_batt']
        eta_motor = inputs['eta_motor']
        eta_pe = inputs['eta_pe']
        eta_cbl = inputs['eta_cbl']
        unit_shaft_pow = inputs['unit_shaft_pow']
        num_motors = inputs['num_motors']
        dt = inputs['dt']
        soc_0 = inputs['soc_0']
        batt_cap__J = inputs['batt_cap']
        hy = inputs['hy']

        # Battery power required (W)
        batt_pow_req = unit_shaft_pow * num_motors / eta_batt / eta_motor / eta_pe / eta_cbl * (1-hy)

        # Battery discharge (J)
        batt_usage__J = np.cumsum(batt_pow_req * dt)

        # Battery state of charge
        soc = (soc_0 * batt_cap__J - batt_usage__J) / batt_cap__J

        # Pack outputs
        outputs['soc'] = soc   



if __name__ == "__main__":
    import openmdao.api as om

    p = om.Problem()
    model = p.model

    ivc = om.IndepVarComp()
    ivc.add_output('unit_shaft_pow', 1000, units='W')
    ivc.add_output('num_motors', 2, units=None)
    ivc.add_output('dt', 1, units='s')
    ivc.add_output('soc_0', 0.5, units=None)
    ivc.add_output('batt_cap', 1000, units='Wh')
    ivc.add_output('eta_batt', 0.8, units=None)
    ivc.add_output('eta_motor', 0.93, units=None)
    ivc.add_output('eta_pe', 0.95, units=None)
    ivc.add_output('eta_cbl', 0.95, units=None)
    ivc.add_output('hy', 0.5, units=None)


    model.add_subsystem('Indeps', ivc, promotes_outputs=['*'])
    model.add_subsystem('ComputeBatteryDischarge', ComputeBatteryDischarge(), promotes_inputs=['*'])

    model.nonlinear_solver = om.NewtonSolver()
    model.linear_solver = om.DirectSolver()

    model.nonlinear_solver.options['iprint'] = 2
    model.nonlinear_solver.options['maxiter'] = 200
    model.nonlinear_solver.options['solve_subsystems'] = True
    model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()

    p.setup()

    #om.n2(p)
    p.run_model()

    print('soc = ', p['ComputeBatteryDischarge.soc'])


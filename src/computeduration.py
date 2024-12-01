import numpy as np  
import openmdao.api as om



class ComputeDuration(om.ImplicitComponent):

    def initialize(self):
        self.options.declare('n', default=1, desc='number of data points')        


    def setup(self):

        # Inputs    
        self.add_input('vz', val= np.ones(self.options['n']) * 0, desc='true airspeed', units='m/s')
        self.add_input('z1', val= 1, desc='time step', units='m')
        self.add_input('z0', val=0, desc='end position in the z-axis', units='m')
        self.add_input('t0', val=0, desc='initial time', units='s')

        # Outputs
        self.add_output('t1', val= 10 * 60, desc='time', units='s')

        self.declare_partials('*', '*', method='fd')

    def apply_nonlinear(self, inputs, outputs, residuals):

        # Unpack inputs
        vz = inputs['vz']
        z1 = inputs['z1']
        z0 = inputs['z0']
        t0 = inputs['t0']
        t1 = outputs['t1']

        # unpack options
        n = self.options['n']

        # unpack outputs
        t = np.linspace(t0, t1, n).flatten()

        dt = np.diff(t)
        dt = np.append(dt, dt[-1])

        z = np.cumsum(vz * dt) + z0
        z1_calc = z[-1]

        # Compute residuals
        residuals['t1'] = z1 - z1_calc

if __name__ == "__main__":
    import openmdao.api as om

    p = om.Problem()
    model = p.model
    n  = 10
    ivc = om.IndepVarComp()
    ivc.add_output('vz', 5 * np.ones(n), units='m/s')
    ivc.add_output('z1', 30000*0.3048, units='m')
    ivc.add_output('z0', 400 * 0.3048, units='m')
    ivc.add_output('t0', 0, units='s')

    model.add_subsystem('Indeps', ivc, promotes_outputs=['*'])
    model.add_subsystem('ComputeDuration', ComputeDuration(n = n), promotes_inputs=['*'])

    model.nonlinear_solver = om.NewtonSolver()
    model.linear_solver = om.DirectSolver()

    model.nonlinear_solver.options['iprint'] = 2
    model.nonlinear_solver.options['maxiter'] = 200
    model.nonlinear_solver.options['solve_subsystems'] = True
    model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()

    p.setup()

    #om.n2(p)
    p.run_model()

    print('time = ', p['ComputeDuration.t1'])

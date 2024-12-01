import numpy as np  
import openmdao.api as om



class ComputeAero(om.ExplicitComponent):

    def initialize(self):

        self.options.declare('wave_a', default=0.825, desc='wave drag a term')
        self.options.declare('wave_b', default=2.61, desc='wave drag b term')
        self.options.declare('MCrit', default=0.9, desc='critical Mach number')
        self.options.declare('n', default=1, desc='number of data points')

    def setup(self):

        # Inputs    
        self.add_input('Cd0', val=1, desc='zero-lift drag coefficient', units=None)
        self.add_input('e', val=1, desc='oswald efficiency factor', units=None)
        self.add_input('AR', val=1, desc='aspect ratio', units=None)
        self.add_input('c', val=330 * np.ones(self.options['n']), desc='speed of sound', units='m/s')
        self.add_input('utas', val= np.ones(self.options['n']), desc='true airspeed', units='m/s')
        self.add_input('CL', val= np.ones(self.options['n']), desc='lift coefficient', units=None)

        # Outputs
        self.add_output('CD', val= np.ones(self.options['n']), desc='drag coefficient', units=None)
        self.add_output('CD_wave', val= np.ones(self.options['n']), desc='wave drag coefficient', units=None)

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):

        # Unpack options
        MCrit = self.options['MCrit']
        wave_a = self.options['wave_a']
        wave_b = self.options['wave_b'] 
    
        # Unpack inputs
        CL = inputs['CL']
        Cd0 = inputs['Cd0']
        e = inputs['e']
        AR = inputs['AR']
        c = inputs['c']
        utas = inputs['utas']


        mach = utas / c

        #CD_wave = np.where(mach > MCrit, wave_a * (mach/MCrit-1) ** wave_b, 0)
        CD_wave =0
        CD = Cd0 + CL**2 / (np.pi * e * AR) + CD_wave


        # Pack outputs
        outputs['CD'] = CD
        outputs['CD_wave'] = CD_wave


if __name__ == "__main__":
    import openmdao.api as om

    p = om.Problem()
    model = p.model

    ivc = om.IndepVarComp()
    ivc.add_output('Cd0', 0.01, units=None)
    ivc.add_output('e', 0.8, units=None)
    ivc.add_output('AR', 10, units=None)
    ivc.add_output('c', 340, units='m/s')
    ivc.add_output('utas', 100, units='m/s')
    ivc.add_output('CL', 0.5, units=None)   

    model.add_subsystem('Indeps', ivc, promotes_outputs=['*'])
    model.add_subsystem('ComputeAero', ComputeAero(), promotes_inputs=['*'])

    model.nonlinear_solver = om.NewtonSolver()
    model.linear_solver = om.DirectSolver()

    model.nonlinear_solver.options['iprint'] = 2
    model.nonlinear_solver.options['maxiter'] = 200
    model.nonlinear_solver.options['solve_subsystems'] = True
    model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()

    p.setup()

    #om.n2(p)
    p.run_model()

    print('CL = ', p['ComputeAero.CL'])
    print('CD = ', p['ComputeAero.CD'])
    print('CD_wave = ', p['ComputeAero.CD_wave'])   

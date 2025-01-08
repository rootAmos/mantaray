import numpy as np
import matplotlib
import pandas as pd
import scipy
import openmdao.api as om
from openmdao.api import NewtonSolver, DirectSolver, ArmijoGoldsteinLS, Problem, Group


from computecl import ComputeCL
from computekinematics import ComputeKinematics
from computeaeroforces import ComputeAeroForces
from computeturbine import ComputeTurbine
from computebatterydischarge import ComputeBatteryDischarge
from computecd import ComputeCD
from computepropthrustgen import ComputePropThrustGen   
from computevelocityfromacc import ComputeVelocityFromAcc
from computeatmos import ComputeAtmos
from computeduration import ComputeDuration
from computetrajectories import ComputeTrajectories
from computeaoa import ComputeAofA
from computetimestep import ComputeTimeStep
from computeacceleration import ComputeAcceleration
from computevelocityfromcl import ComputeVelocityFromCL    
from computevelocityfromacc import ComputeVelocityFromAcc     

from computeproppowerreq import ComputePropPowerReq
from computethrustfromacc import ComputeThrustFromAcc
from computepropeta import ComputePropEta
from computeinducedvelocity import ComputeInducedVelocity
"""
Compute the performance of the aircraft during climb
"""

# Futures

# Built-in/Generic Imports

class Performance(om.Group):
    """
    Group containing the Stack MDA without IndepVarComp for inputs
    """

    def initialize(self):
        self.options.declare('n', default=1, desc='number of points')
        self.options.declare('g', default=9.806, desc='gravitational acceleration')

    def setup(self):



        
        self.add_subsystem(name='ComputeVelocityFromAcc',
                           subsys=ComputeVelocityFromAcc(n=self.options['n']),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])


        self.add_subsystem(name='ComputeKinematics',
                           subsys=ComputeKinematics(n=self.options['n']),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])
        
        self.add_subsystem(name='ComputeDuration',
                           subsys=ComputeDuration(n=self.options['n']),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])
        
        self.add_subsystem(name='ComputeTimeStep',
                           subsys=ComputeTimeStep(n=self.options['n']),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])

        self.add_subsystem(name='ComputeTrajectories',
                           subsys=ComputeTrajectories(n=self.options['n']),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])
        
        self.add_subsystem(name='ComputeAtmos',
                           subsys=ComputeAtmos(n=self.options['n']),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])

        self.add_subsystem(name='ComputeThrustFromAcc',
                           subsys=ComputeThrustFromAcc(n=self.options['n'], g=self.options['g']),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])
       
        self.add_subsystem(name='ComputePropPowerReq',
                           subsys=ComputePropPowerReq(n=self.options['n']),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])


        self.add_subsystem(name='ComputeCL',
                           subsys=ComputeCL(n=self.options['n'], g=self.options['g']),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])

        
        self.add_subsystem(name='ComputeAoA',
                           subsys=ComputeAofA(n=self.options['n']),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])
        

        self.add_subsystem(name='ComputeCD',
                           subsys=ComputeCD(
                               n=self.options['n']),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])

        self.add_subsystem(name='ComputeAeroForces',
                           subsys=ComputeAeroForces(n=self.options['n']),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])
        


        self.add_subsystem(name='ComputeTurbine',
                           subsys=ComputeTurbine(n=self.options['n']),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])

        self.add_subsystem(name='ComputeBatteryDischarge',
                           subsys=ComputeBatteryDischarge(n=self.options['n']),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])
                     


        self.nonlinear_solver = NewtonSolver()
        self.linear_solver = DirectSolver()
        self.nonlinear_solver.options['iprint'] = 2
        self.nonlinear_solver.options['maxiter'] = 200
        self.nonlinear_solver.options['solve_subsystems'] = True
        self.nonlinear_solver.options['stall_limit'] = 4
        self.nonlinear_solver.linesearch = om.BoundsEnforceLS()
        self.nonlinear_solver.options['rtol'] = 1e-4



if __name__ == "__main__":
    import numpy as np
    # from omxdsm import write_xdsm

    # generate path where xlsx with all data can be found
    n = 20

    ivc = om.IndepVarComp()

    mass =  8600 # mass flow in to compressor
    S =  30 # wing area
    b = 20 # wing span
    AR = b**2 / S # wing aspect ratio

    z0 = 400 * 0.3048 # initial altitude (m)
    z1 = 10000 *0.3048 # end altitude (m)

    # Position
    ivc.add_output('t0', val=0, units='s', desc='initial time')
    ivc.add_output('x0', val=0, units='m', desc='initial distance in x-dir of earth fixed body frame')
    ivc.add_output('z0', val = z0, units ='m', desc='initial altitude')
    ivc.add_output('v0', val=50, units='m/s', desc='initial velocity')
    ivc.add_output('z1', val=z1, units='m', desc='final altitude')

    # Velocity
    #ivc.add_output('vel', val=50 * np.ones(n), units='m/s', desc='velocity in longitudinal direction of body-fixed frame') # v2 speed assumption
    #ivc.add_output('gamma', val= 4 * np.pi/180 * np.ones(n), units='rad', desc='flight path angle')
    ivc.add_output('acc', val=0.05 * np.ones(n), units='m/s**2', desc='acceleration') # v2 speed assumption

    # Aircraft geometry
    ivc.add_output('S', val=S , units='m**2', desc='wing area')
    ivc.add_output('AR', val=AR , units=None, desc='wing aspect ratio')
    ivc.add_output('mass', val=mass, units='kg', desc='aircraft mass')


    # Aero
    ivc.add_output('alpha_0', val=-3 * np.pi / 180 , units='rad', desc='zero-lift angle of attack')   
    ivc.add_output('alpha_i', val=2 * np.pi / 180 , units='rad', desc='incidence angle')   
    ivc.add_output('CLa', val=5.5, units='1/rad', desc='lift curve slope')
    #ivc.add_output('CL', val=0.7 * np.ones(n), units=None, desc='lift coefficient')

    ivc.add_output('Cd0', val=0.02, units=None, desc='zero-lift drag coefficient')
    ivc.add_output('e', val=0.8, units=None, desc='oswald efficiency factor')

    # Powertrain
    ivc.add_output('eta_batt', val=0.9, units=None, desc='battery efficiency')
    ivc.add_output('eta_motor', val=0.9, units=None, desc='motor efficiency')
    ivc.add_output('eta_pe', val=0.9, units=None, desc='power electronics efficiency')
    ivc.add_output('eta_cbl', val=0.9, units=None, desc='cables efficiency')
    ivc.add_output('eta_prop', val=0.8, units=None, desc='propeller efficiency')
    ivc.add_output('eta_duct', val=0.8, units=None, desc='duct efficiency')
    ivc.add_output('eta_fan', val=0.8, units=None, desc='fan efficiency')  
    ivc.add_output('eta_gen', val=0.9, units=None, desc='generator efficiency')

    ivc.add_output('batt_cap', val=400e3 * 3600, units='J', desc='battery capacity')
    ivc.add_output('soc_0', val=1, units=None, desc='initial battery state of charge')
    ivc.add_output('num_motors', val=2, units=None, desc='number of engines')
    ivc.add_output('d_blade', val=2.4, units='m', desc='blade diameter')
    ivc.add_output('d_hub', val=0.5, units='m', desc='hub diameter')

    #ivc.add_output('unit_shaft_pow', val=500e3 * np.ones(n), units='W', desc='power generated per engine')
    ivc.add_output('psfc', val=0.0003 / 3600 * np.ones(n), units='kg/W/s', desc='power specific fuel consumption')
    ivc.add_output('num_turbines', val=2, units=None, desc='number of turbines')
    ivc.add_output('hy', val=0.01 * np.ones(n), units=None, desc='hybridization ratio')

    
    p = om.Problem()
    model = p.model
    p.model.add_subsystem('ivc', ivc, promotes=['*'])
    p.model.add_subsystem('System', Performance(n=n),
                          promotes_inputs=['*'], promotes_outputs=['*'])

    # set model settings
    p.model.nonlinear_solver = om.NewtonSolver()
    p.model.linear_solver = om.DirectSolver()

    p.model.nonlinear_solver.options['iprint'] = 2
    p.model.nonlinear_solver.options['maxiter'] = 200
    p.model.nonlinear_solver.options['solve_subsystems'] = True
    p.model.nonlinear_solver.linesearch = om.BoundsEnforceLS()

    
    # Analysis
    p.setup()
    # data = p.check_partials(method='fd')

    #om.n2(p)
    p.run_model()
    #p.check_partials(compact_print=True)

    """

    
    
    # setup the optimization
    p.driver = om.ScipyOptimizeDriver()
    p.driver.options['optimizer'] = 'SLSQP'

    #p.model.add_design_var('hy', lower=1e-3, upper=1 - 1e-3)
    #p.model.add_design_var('gamma', lower=1 *np.pi/180, upper=40 *np.pi/180)
    p.model.add_design_var('acc', lower=0.01, upper=0.2)
    p.model.add_constraint('soc', lower=1e-3)
    p.model.add_objective('obj_func')

    
    p.setup()

    # Setting initial values
    #p.set_val('hy', 0.5 * np.ones(n))
    p.set_val('gamma', 7 * np.pi/180 * np.ones(n))
    #p.set_val('acc', 0.05 * np.ones(n))
    #p['acc'] = 0.05 * np.ones(n)

    # Climb to 30,000 ft

    #om.n2(p)

    # Set initial values.
    #p.set_val('unit_shaft_pow', 500e3 * np.ones(n))
    #p.set_val('hy', 0.5 * np.ones(n))

    # run the optimization
    p.run_driver()
    """

    # Print the results
    

    print('Ending Altitude (ft) = ', p['z'][-1] / 0.3048)
    print('Time (min) = ', p['t1']/60)
    print('Flight path angle profile (deg) = ', p['gamma'] * 180/np.pi)
    print('True airspeed profile (m/s) = ', p['vel'])  
    print('Battery state of charge profile (%) = ', p['soc'] * 100)
    print('Fuel consumption (kg) = ', p['obj_func'])
    print('Hybridization ratio profile (%) = ', p['hy'] * 100)
    print('Power profile (kW) = ', p['unit_shaft_pow']/1000)
    print('Acceleration profile (m/s^2) = ', p['acc'])
    
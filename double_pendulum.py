import sys
import numpy as np
import csv
import argparse
import time
from multiprocessing import Pool


from scipy.integrate import odeint


# The gravitational acceleration (m.s-2).
g = 9.81


def deriv(y, t, L1, L2, m1, m2):
    """Return the first derivatives of y = theta1, z1, theta2, z2."""
    theta1, z1, theta2, z2 = y

    c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)

    theta1dot = z1
    z1dot = (m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) -
             (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2)
    theta2dot = z2
    z2dot = ((m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) +
             m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
    return theta1dot, z1dot, theta2dot, z2dot

def solve(L1, L2, m1, m2, tmax, dt, y0):
    t = np.arange(0, tmax+dt, dt)

    # Do the numerical integration of the equations of motion
    y = odeint(deriv, y0, t, args=(L1, L2, m1, m2))
    theta1, theta2 = y[:,0], y[:,2]

    # print "[0, 1, 4,..., 81]"
    # Convert to Cartesian coordinates of the two bob positions.
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)

    return theta1, theta2, x1, y1, x2, y2, y0

     

def simulate_pendulum(theta_resolution, time_max, time_step, output_file):
    # Pendulum rod lengths (m), bob masses (kg).
    L1, L2 = 1.0, 1.0
    m1, m2 = 1.0, 1.0

    # Maximum time, time point spacings (all in s).
    tmax, dt = time_max, time_step #30.0, 0.01

    csvfile = open(output_file + '.csv', 'w')
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(['Theta1_init', 'Theta2_init', 'theta1', 'theta2', 'x1', 'y1', 'x2', 'y2'])



    for theta1_init in np.linspace(0, 2*np.pi, theta_resolution):
        for theta2_init in np.linspace(0, 2*np.pi, theta_resolution):
            # Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
            y0 = np.array([
                theta1_init,
                0.0,
                theta2_init,
                0.0
            ])

            theta1, theta2, x1, y1, x2, y2,_ = solve(L1, L2, m1, m2, tmax, dt, y0)


            filewriter.writerow([theta1_init, theta2_init, theta1[-1], theta2[-1], x1[-1], y1[-1], x2[-1], y2[-1] ])

            #print theta1_init, theta2_init, theta1[-1], theta2[-1]

    csvfile.close()


def iter_generator(L1, L2, m1, m2, tmax, dt, theta_resolution):
    for theta1_init in np.linspace(0, 2*np.pi, theta_resolution):
        for theta2_init in np.linspace(0, 2*np.pi, theta_resolution):
            # Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
            y0 = np.array([
                theta1_init,
                0.0,
                theta2_init,
                0.0
            ])

            yield L1, L2, m1, m2, tmax, dt, y0


def _worker(args):
    L1, L2, m1, m2, tmax, dt, y0 = args
    return solve(L1, L2, m1, m2, tmax, dt, y0)


def simulate_pendulum_parallel(theta_resolution, time_max, time_step, output_file):
    # Pendulum rod lengths (m), bob masses (kg).
    L1, L2 = 1.0, 1.0
    m1, m2 = 1.0, 1.0

    # Maximum time, time point spacings (all in s).
    tmax, dt = time_max, time_step #30.0, 0.01



    inputs_gen = iter_generator(L1, L2, m1, m2, tmax, dt, theta_resolution)

    p = Pool(processes=4)
    results = p.imap(_worker, inputs_gen )

    #print list(results)[:5] 

    

    csvfile = open(output_file + '_parallel.csv', 'w')
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(['Theta1_init', 'Theta2_init', 'theta1', 'theta2', 'x1', 'y1', 'x2', 'y2'])



    for i in results:
        theta1, theta2, x1, y1, x2, y2, y0 = i

        filewriter.writerow([y0[0],
                             y0[2],
                             theta1[-1], 
                             theta2[-1], 
                             x1[-1], 
                             y1[-1], 
                             x2[-1], 
                             y2[-1] ])

        #print y0[0], y0[2], theta1[-1], theta2[-1]

    csvfile.close()



def setup_parser():
    parser = argparse.ArgumentParser(
        description='Double pendulum simulation by utilizing multiprocessing '
    )
    parser.add_argument(
        'resolution',
        help="Put simulation time",
        type=float
    )
    parser.add_argument(
        'sim_time_max',
        help="Put simulation time",
        type=float
    )
    parser.add_argument(
        'sim_time_step',
        help="Put the stamp time ",
        type=float
    )

    parser.add_argument(
        'output_file',
        help="Put the stamp time ",
    )

    args = parser.parse_args()
    return args

def main():
    #Input example: 10 30.0 0.01 res1
    args = setup_parser()

    t1 = time.time()
    simulate_pendulum (args.resolution, args.sim_time_max,\
    			    args.sim_time_step, args.output_file)
	
    t2 = time.time()
    simulate_pendulum_parallel(args.resolution, args.sim_time_max,\
			    args.sim_time_step, args.output_file)
    t3 = time.time()


    print ('seq time = ', t2 - t1, 'par time = ', t3 - t2)
    print('Speed up = ', (t2 - t1) / (t3 - t2))

if __name__ == "__main__":
    main()

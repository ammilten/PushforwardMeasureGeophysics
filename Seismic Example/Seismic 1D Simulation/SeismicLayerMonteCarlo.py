import MonteCarlo
import sys, getopt, os

# ---------------- Parse arguments ---------------------
def get_args(argv):
    T = 1
    dt = 0.002
    N = 1
    showDists = False
    parallel = False
    nproc = None
    MCfolder = ''
    foundMC = False
    overwrite = False
    imp = False

    try:
        opts, args = getopt.getopt(argv,"hf:T:t:n:p:doi",["mcfolder=","tmax=","dt=","nreals=","showdists","parallel","overwrite","import"])
    except getopt.GetoptError:
        print("ERTMonteCarlo.py -f <path_to_mcfolder> -T <max_time> -t <delta_t> -n <num_reals> -p <num_processors> -d -o")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("\nBASIC USAGE:\npython SeismicLayerMonteCarlo.py -f <path_to_mcfolder> -n <num_reals>\n\nSWITCHES:\n  -f, --mcfolder <folder>: Path to Monte Carlo output folder (required)\n  -T, --tmax <max_time>: Max time of seismogram recording, in seconds (default=1)\n  -t, --dt <sample_rate>: Sampling rate of seismic data, in seconds (default=0.002)\n -n, --nreals <num_reals>: Number of realizations (default=1)\n  -p, --parallel <num_processors>: If specified, run in parallel. Option to specify number of of processors. If number of processors is not specified, it is auto-selected.\n  -d, --showdists: If specified, plot the parameter distributions prior to simulations.\n  -o, --override: If specified, the Monte Carlo will overwrite the data in MCfolder if MCfolder already exists.\n  -i, --import: If specified, the program will attempt to load Monte Carlo parameters from <MCfolder>/params.dat.")
            sys.exit(2)
        elif opt in ("-T", "--tmax"):
            T = float(arg)
        elif opt in ("-t", "--dt"):
            dt = float(arg)
        elif opt in ("-n", "--nreals"): 
            N = int(arg)
        elif opt in ("-f", "--mcfolder"):
            MCfolder = arg
            foundMC = True
        elif opt in ("-d", "--showdists"):
            showDists = True
        elif opt in ("-o", "--overwrite"):
            overwrite = True
        elif opt in ("-i", "--import"):
            imp = True
        elif opt in ("-p", "--parallel"):
            parallel = True
            try:
                nproc = int(arg)
            except:
                nproc = None
             
    if not foundMC:
        print("MCfolder argument required: use '-f <path_to_mcfolder>' or '--mcfolder <path_to_mcfolder>'")
        sys.exit(2)

    if not MCfolder.endswith('/'):
        MCfolder = MCfolder+'/'

    print("MCfolder = "+MCfolder)
    print("Recording Time = " + str(T) + " seconds")
    print("Sampling Rate = " + str(dt) + " seconds (" + str(dt*1000) + " ms)")
    print("Number of realizations = " + str(N))
    if parallel:
        if nproc is None:
            print("Parallelized = true (auto-select number of processors)")
        else:
            print("Parallelized = true (" + str(nproc) + " processors)")
    else:
        print("Parallelized = false")
    if showDists:
        print("Plotting distributions = true")
    else:
        print("Plotting distributions = false")
    if imp:
        print("\n** Simulation will be imported from '" + MCfolder + "params.dat'.\n   Mesh quality and number of realizations above may be incorrect.")

    print("")
    return MCfolder, T, dt, N, showDists, parallel, nproc, overwrite, imp



#------------------ MAIN SCRIPT -----------------------
if __name__ == "__main__":


    MCfolder, T, dt, N, showDists, parallel, nproc, overwrite, imp = get_args(sys.argv[1:])
    print("\n----------- Preparing Monte Carlo -----------")
#    MCfolder = '/home/ammilten/Programs/ERTFaultMonteCarlo/data/MC/'

    rhob = [1500, 2200] #uniform
    Vb = [2500, 3500] #uniform
    rhol = [2000,2600] #uniform
    Vl = [1700, 2200] #uniform
    h = [250, 500] #uniform
    dh = [30, 300] #uniform
    fM = [50, 50] #constant [10,75] 

    if imp:
        MC = MonteCarlo.import_simulation(MCfolder,overwrite, parallel, nproc, showDists)
    else:
        MC = MonteCarlo.MonteCarlo(T, dt, rhob, Vb, rhol, Vl, h, dh, fM, N, MCfolder, overwrite, parallel, nproc, showDists)

    MC.run()


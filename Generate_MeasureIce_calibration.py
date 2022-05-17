#!/usr/bin/env python
import h5py
import copy
import tqdm
import sys
import matplotlib.pyplot as plt
from PIL import Image
import pyms
import numpy as np
import torch
import os
import torch

def straight_line(x, m):
    """ Straight line function y=f(x) """
    return m*x

def straight_line_w_intercept(x,m,c):
    return m*x+c



def invA_to_mrad(invA, eV):
    """Convert inverse Angstrom to mrad

    Parameters
    ----------
    invA : Reciprocal space length in inverse Angstrom to be converted
    eV   : Energy of electron beam in eV
    """
    return invA / pyms.wavev(eV) * 1000


def mrad_to_invA(mrad, eV):
    """Convert mrad to inverse Angstrom

    Parameters
    ----------
    mrad : Reciprocal space length in inverse Angstrom to be converted
    eV   : Energy of electron beam in eV
    """
    return mrad * pyms.wavev(eV) / 1000


def apply_objective_aperture(
    psi, real_dim, aperture, eV, f=None, d=None, app_units="mrad", qspace_in=True
):
    """
    Applies an objective aperture in reciprocal space to an electron
    wave function.

    Parameters
    ----------
    psi: array_like (ny,nx)
    electron wavefunction
    real_dim: array_like (2,)
    """

    # Calculate electron wavelength
    lam = 1 / pyms.wavev(eV)

    # Get pixel dimensions
    pix_dim = psi.shape[-2:]

    # Check if aperture is array like or scalar. If the former assume
    # that it is the aperture mask array, if the latter assume that it is
    # the dimensions of the array in units of inverse length and generate
    # the aperture mask array on the fly.
    if hasattr(aperture, "__len__"):
        app = aperture
    else:
        app = pyms.make_contrast_transfer_function(
            pix_dim, real_dim, eV, aperture, app_units=app_units
        )

    # Apply aperture mask array to electron wavefunction in reciprocal
    # space
    if not qspace_in:
        psi = np.fft.fft2(psi)
    if np.iscomplexobj(psi):
        psi = np.abs(psi) ** 2
    return psi * app


def plasmon_scattering_cross_section(gridshape, gridsize, theta_E, eV):
    """
    Calculate the normalized plasmon scattering cross-section

    Parameters
    ----------
    gridshape : (2,) array_like
    Size of grid in pixels
    gridsize : (2,) array_like
    Size of grid in Angstroms
    theta_E : float
    Impact parameter in mrad
    eV : float
    Electron energy in eV
    """
    qy, qx = pyms.utils.q_space_array(gridshape, gridsize)
    te = theta_E * pyms.wavev(eV) * 1e-3
    xsec = 1 / (te ** 2 + qy ** 2 + qx ** 2)
    return xsec / np.sum(xsec)


def scattering_probability(t, t_mfp, n):
    return (1 / np.math.factorial(n)) * (t / t_mfp) ** n * np.exp(-t / t_mfp)


def n_scatt_events(t, t_mfp, cutoff=0.01):
    """
    Calculate probabilities for different number of scattering events
    up to cutoff (capture 99% of scattering events by default)

    Parameters
    ----------
    t : float
    Thickness
    t_mfp : float
    Mean free path for scattering
    cutoff : float, optional

    Returns
    -------
    Pn : list
    List containing probabilities of scattering 0 to n times
    """
    # Initialize some variables
    Pn = []
    P = 1
    n = 0

    # Loop over different orders of multiple scattering until cutoff
    # has been reached
    while True:

        # Calculate probability of scattering inelastically this many
        # times and append to list
        P = scattering_probability(t, t_mfp, n)
        Pn.append(P)

        # Check whether cutoff probability has been exceeded yet
        if np.sum(Pn) > 1 - cutoff:
            # break while loop if true
            break

        # If false move to next order of multiple scattering
        n += 1

    # Return scattering probabilities
    return Pn


def plas_scatt(DP, gridshape, gridsize, theta_E, eV, t, t_mfp):
    """
    Apply Plasmon inelastic scattering to Diffraction pattern
    incorporating only elastic scattering

    Parameters
    ----------
    DP : (ny,nx) array_like
    Diffraction pattern incorporating only elastic scattering
    gridshape : (2,) int,array_like
    Size of grid in pixels
    gridsize : (2,) float,array_like
    Size of grid in Angstrom
    theta_E : float
    Impact parameter in mrad for plasmon scattering
    eV : float
    Electron beam energy in eV
    t : float
    Specimen thickness
    t_mfp : float
    Mean free path for inelastic scattering
    """
    #
    Pn = n_scatt_events(t, t_mfp)
    xsec = plasmon_scattering_cross_section(gridshape, gridsize, theta_E, eV)

    # Add contribution of (only) elastically electrons
    DP_out = DP * Pn[0]

    for P in Pn[1:]:
        # Convolve diffraction pattern with inelastic scattering cross
        # section (if len(Pn)> 2 this will happen multiple times)
        DP = pyms.utils.convolve(xsec, DP)

        # Add contribution for this order of (multiple) inelastic scattering
        DP_out += P * DP

    # Return diffraction pattern, now with inelastic scattering
    return DP_out


def get_t(structurefilename):
    """
    Get the thickness from the file.

    Doesn't incur the overhead of loading the whole structure.
    """
    from re import split

    f = open(structurefilename, "r")
    for i in range(2):
        line = f.readline().strip()
    f.close()
    return np.asarray([float(x) for x in split("\s+", line)])[:3]


def tile_out_amorphous_structure(structure, ny, nx):
    """Tile out amorphous structure ny x nx times with random flipping"""
    newcell = None
    for y in range(ny):
        xunit = None
        for x in range(nx):

            zrot, yrot, xrot = np.random.randint(0, 4, (3,))
            newunit = copy.deepcopy(structure)
            newunit.rot90(zrot, axes=(0, 1))
            newunit.rot90(yrot, (1, 2))
            newunit.rot90(xrot, (0, 2))

            if xunit is None:
                xunit = copy.deepcopy(newunit)
            else:
                xunit = xunit.concatenate(newunit, axis=1)
        if newcell is None:
            newcell = copy.deepcopy(xunit)
        else:
            newcell = newcell.concatenate(xunit, axis=0)

    return newcell


def imfp(E, Mw=18.015, rho=920e3, Z=10):
    """
    Calculates inelastic mean free path using Eq.(8) of:

    Vulović, Miloš, et al. "Image formation modeling in cryo-electron
    microscopy." Journal of structural biology 183.1 (2013): 19-32.

    This is approximate, but if scaled to match experimental measurements
    gives a useful interpolation for unmeasured electron energies

    Parameters:
    E : float
    Electron energy in keV
    Mw : float, optional
    Molecular weight in g/mol
    rho : float, optional
    Density in g/m^3
    Z : float, optional
    (Sum of) Atomic number(s) of molecule
    """
    mcsqr = 0.5109906e6  # rest mass of electron in keV
    betasqr = 1 - (mcsqr / (E * 1e3 + mcsqr)) ** 2

    return (
        Mw
        * betasqr
        * 1e10
        / 9.03
        / rho
        / np.sqrt(Z)
        / np.log(betasqr * (E * 1e3 + mcsqr) / 10)
    )

def carbon_imfp(eV):
    """Calculate inelastic mean free path in nm for amorphous carbon."""
    data = np.asarray([[71.95811469176782, 6492.6028902254875],
        [119.63625785913935, 7130.633928791563],
        [190.9779470418646, 9408.437214026815],
        [376.1496069126305, 13369.322510759523],
        [610.7176162238331, 19097.182408282333],
        [991.5629311098909, 27454.874387390308],
        [1609.9045127116265, 40497.73175277928],
        [2613.845736597052, 59929.005656978676],
        [4243.847682132933, 88968.9732655457],
        [6890.323670207287, 132932.2494436579],
        [11187.149925314328, 200543.01114977678],
        [18163.4897635651, 309719.95607402036],
        [29490.295794159905, 449144.634629818],
        [47880.53162402578, 654058.9873241475],
        [77738.97300322555, 956550.268801701],
        [126217.22688984855, 1297912.8948451944],
        [203925.91061759694, 1751693.3152705648],
        [321904.7658535461, 2121215.6313316543],
        [532328.7996896341, 2573736.8871517177],
        [803098.6830720035, 2810661.4692487745]]).T
    data[1] /= 1e4
    from scipy.interpolate import interp1d
    func = interp1d(data[0],data[1])
    return func(eV)

def generate_calibration_curves(
    keV,
    obj_apertures,
    app_units="invA",
    slicethick=10.0,
    dt=250,
    tmax=6000,
    gridshape=[2048, 2048],
    structuretiling=[3, 3],
    waterbox="supercooled_water.xyz",
    DWF=0.01,
    device_type=None,
    imfp_prov = None,
    carbon_thickness = None,
    carbonbox="Amorphous_carbon.xyz",
):
    """
    For a given accelerating voltage and set of apertures generate a set
    of calibration curves

    Parameters
    ----------
    keV, float : Electron energy in keV
    obj_apertures : objective apertures in units of app_units (either mrad or invA)
    slice_thick, float, optional : Thickness of multislice slicing
    dt, float, optional : Step size of calibration curve output in Angstrom
    tmax, float, optional : maximum thickness to simulate in Angstrom
    gridshape, optional : grid size of simulated diffraction pattern
    structuretiling : random tiling of water box
    DWF : vibrations added to water molecules
    device_type : string
    Either 'cpu' or 'gpu', if None (default) then pytorch will try to use
    any gpu available

    """

    # Get mean free path for plasmon (ie. inelastic) scattering,
    # Values from Yesibolati et al. for 120 kV and 300 kV
    if abs(keV - 120) < 3:
        lambda_d = 2100
    elif abs(keV - 300) < 3:
        lambda_d = 3200
    else:
        lambda_d = imfp(keV,Z=13.24) * 10

    # If user provides an inelastic mean free path use this instead,
    # converting from nm to Angstrom
    if imfp_prov is not None:
        lambda_d = imfp_prov*10

    # Many routines use eV, not keV so convert now.
    eV = keV * 1e3

    # Theta_E is linearly interpolated from experimental measurements results
    # Since this mainly effects small angle inelastic scattering the result is
    # not so sensitive to it.
    theta_E = (0.12 - 0.26) / (3e5 - 1.2e5) * (eV - 1.2e5) + 0.26

    # Generate thickness list
    thicknesses = np.arange(dt, tmax + 1, dt)
    # Number of thicknesses
    nt = len(thicknesses)
    #Number of objective apertures
    napp = len(obj_apertures)

    # Initialize
    LogI0I = np.zeros((nt+1, napp))

    # Use the GPU for the mutlislice calculation if it is not available
    if device_type is None and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Read in water box
    water = pyms.structure.fromfile(waterbox, atomic_coordinates="cartesian")
    print("Ice structure read in")
    water.atoms[:, 5] = DWF

    # Tile out structure in x and y direction
    water = tile_out_amorphous_structure(water, *structuretiling)
    gridsize = water.unitcell[:3]
    n = int(np.round(gridsize[2] / slicethick))
    subslices = (np.arange(n) + 1) / n

    # Generate multislice precursors
    propagators, transmission_functions = pyms.multislice_precursor(
        water, gridshape, eV, subslices, nT=1,showProgress=False
    )
    # The thickness_to_slices function converts thickness in Angstrom
    # (ie. the thickness increment in our thickness to intensity series)
    # to number of multislice slices - this will be passed to the
    # multislice function to advance our electron wave through successive
    # thicknesses of amorphous ice
    slices = pyms.thickness_to_slices(
        dt, gridsize[2], subslicing=True, subslices=subslices
    )[0]

    # Generate aperture masks (the function calledgenerates a strong-phase
    # contrast transfer function with a 0 defocus)
    apps = (
        np.abs(
            np.stack(
                [
                    pyms.make_contrast_transfer_function(gridshape, gridsize, eV, app,app_units=app_units)
                    for app in obj_apertures
                ]
            )
        )
        ** 2
    )

    # Generate illumination
    illum = torch.from_numpy(pyms.plane_wave_illumination(gridshape, gridsize, eV, qspace=True))

    # Option to calculate carbon thickness
    if carbon_thickness is not None:
        # Read in carbon box
        carbon = pyms.structure.fromfile(carbonbox, atomic_coordinates="cartesian")

        # Match carbon to ice size in x and y by tiling it out (if necessary) and cropping it
        ctiling = np.round(water.unitcell[:2]/carbon.unitcell[:2]).astype(int)
        if np.any(ctiling>1):
            carbon = tile_out_amorphous_structure(water, *ctiling)
        # Crop down (possibly tiled carbon) to the size of the water box
        resize = np.stack([np.zeros(2),water.unitcell[:2]/carbon.unitcell[:2]],axis=1)
        carbon = carbon.resize(resize,[0,1])

        cgridsize = carbon.unitcell[:3]
        cn = int(np.round(cgridsize[2] / slicethick))
        csubslices = (np.arange(cn) + 1) / cn
        print("Carbon structure read in")
        carbon.atoms[:, 5] = DWF
        cslices = pyms.thickness_to_slices(
            carbon_thickness*10, cgridsize[2], subslicing=True, subslices=csubslices
        )[0]

        # Do multislice through carbon
        cpropagators, ctransmission_functions = pyms.multislice_precursor(
            carbon, gridshape, eV, subslices, nT=1,showProgress=False
        )
        Image.fromarray(np.abs(transmission_functions.cpu().numpy()[0,0])).save('T.tif')
        Image.fromarray(np.angle(transmission_functions.cpu().numpy()[0,0])).save('T_phase.tif')
        illum = pyms.multislice(
                illum,
                cslices,
                cpropagators,
                ctransmission_functions,
                tiling=[16, 16],
                device_type=device,
                return_numpy=False,
                output_to_bandwidth_limit=False,
                subslicing=True,
                qspace_in=True,
                qspace_out=True,
            )
        # Caculate intensity of diffraction pattern
        DP = np.abs(illum.cpu().numpy()) ** 2

        # Apply plasmon scattering to Reciprocal space intensity
        # (assume incoherent energy channels -- don't tell Prof. Peter Schattschneider)
        if carbon_thickness is not None:
            DP_inel = plas_scatt(DP, gridshape, gridsize, theta_E, eV, carbon_thickness*10, carbon_imfp(eV)*10)
        # Apply objective aperture to Recipiprocal space intensity
        DP_inel = apply_objective_aperture(
            DP_inel, gridsize, apps, eV, qspace_in=True, app_units=app_units
        )
        # Record plasmon scattered intensity in look-up table
        LogI0I[0] = np.log(np.prod(gridshape) / np.sum(DP_inel, axis=(1, 2)))

    else:
        LogI0I[0] = 0

    for i, t in enumerate(tqdm.tqdm(thicknesses, desc="thicknesses")):
        # Apply multislice algorithm to simulate elastic scattering
        # Note the tiling = [16,16] which will generate psuedo-random
        # instances of amorphous ice structure by circularly shifting
        # our ice box in x and y
        illum = pyms.multislice(
            illum,
            slices,
            propagators,
            transmission_functions,
            tiling=[16, 16],
            device_type=device,
            return_numpy=False,
            output_to_bandwidth_limit=False,
            subslicing=True,
            qspace_in=True,
            qspace_out=True,
        )

        # Caculate intensity of diffraction pattern
        DP = np.abs(illum.cpu().numpy()) ** 2

        # Apply plasmon scattering to Reciprocal space intensity
        # (assume incoherent energy channels -- don't tell Prof. Peter Schattschneider)
        if carbon_thickness is not None:
            DP = plas_scatt(DP, gridshape, gridsize, theta_E, eV, carbon_thickness*10, carbon_imfp(eV)*10)
        DP_inel = plas_scatt(DP, gridshape, gridsize, theta_E, eV, t, lambda_d)


        # Apply objective aperture to Recipiprocal space intensity
        DP_inel = apply_objective_aperture(
            DP_inel, gridsize, apps, eV, qspace_in=True, app_units=app_units
        )
        # Record plasmon scattered intensity in look-up table
        LogI0I[i+1] = np.log(np.prod(gridshape) / np.sum(DP_inel, axis=(1, 2)))

    # Return results with origin concatenated (necessary for plotting and interpolation)
    return np.concatenate(([0],thicknesses)), LogI0I


def make_plot(
    thicknesses, logII0, title="Intensity-ice thickness calibration", labels=None
):
    """Generate an thickness - image intensity plot for reference"""
    fig, ax = plt.subplots(figsize=(4, 4), ncols=1, constrained_layout=True)
    tmax = max(thicknesses)
    colors = plt.get_cmap("viridis")(
        np.arange(len(obj_apertures)) / (len(obj_apertures) - 1)
    )

    if labels is None:
        labels = [None] * len(obj_apertures)

    mdpnt = len(thicknesses)//2
    for I, c, label in zip(logII0.T, colors, labels):
        lam = fit_straight_line(thicknesses/10,I)
        ax.plot(thicknesses / 10, np.exp(-I), linestyle="-", label=label+' {0:4d}'.format(int(1/lam)), c=c)

        ax.text(thicknesses[mdpnt]/10,np.exp(-I[mdpnt])*1.05,r'$\lambda={0:4d}$'.format(int(1/lam)))

    ax.legend()
    ax.set_ylabel("I/I$_0$")
    ax.set_xlim([0.0, thicknesses.max() / 10])
    ax.set_ylim([max(0.1, np.exp(-logII0).min() - 0.2), 1])
    ax.set_yscale("log")
    ax.set_xlabel("Thickness (nm)")

    # ax.set_ylim([0, tmax / 10])
    ax.set_title(title)
    # fig.tight_layout()
    return fig


def save_calibrationhdf5(
    filename, keV, microscopename, thicknesses, LogII0, obj_apertures, appunits,Aperturemicron
):
    # Force h5 filename extension
    outputfilename = os.path.splitext(filename)[0] + ".h5"
    # Initialise output file
    with h5py.File(outputfilename, "w") as f:
        lam = []
        for I in LogII0.T:
            lam.append(1/fit_straight_line(thicknesses,I))

        attributes = {
            "Electron energy": keV,
            "Microscope name": microscopename,
            "Aperture units": appunits,
        }

        floatarrays = {'Apertures': obj_apertures,"Thicknesses":thicknesses,"LogI0/I":LogII0,"ALS coefficients":lam}
        intarrays = {'Apertures micron' : Aperturemicron}


        for key, val in attributes.items():
            f.attrs[key] = val

        for key, val in floatarrays.items():
            f.create_dataset(key, data=np.asarray(val), dtype=float)

        for key, val in intarrays.items():
            f.create_dataset(key, data=np.asarray(val), dtype=int)

def fit_straight_line(x,y,force_zero_origin=False):
    """
    For arrays x and y fit a straight line and return the slope and intercept.

    Parameters
    ----------
    x, array_like (n,) : x datapoints
    y, array_like (n,) : y datapoints
    force_zero_origin, bool, optional : If True (default) force an axis
                                        intercept at (0,0)

    """
    from scipy.optimize import curve_fit
    # The origin is an optional parameter in the straight_line function
    # setting the starting point p0
    if force_zero_origin:
        func = straight_line
    else:
        func =straight_line_w_intercept
    popt, pcov = curve_fit(func, x,y)
    return popt[0]

def output_result(keV, microscopename, thicknesses, LogII0, obj_apertures, appunits,Aperturemicron,carbon=None):

    line = "For objective aperture #{0}: {1} {2} (labelled {3}) lambda = {4:4d}\n"
    cstring = ''
    if carbon is not None:
        cstring = " with a carbon layer of thickness {0} nm".format(carbon)
    description = "MeasureIce results for {0}, a {1} kV TEM".format(microscopename,int(keV))
    description += cstring + ": \n"
    description += "The value lambda (often referred to as the ALS coefficient) can\n"
    description += "be used in software such as Leginon and SerialEM\n\n"
    description += "For more information see:\n\n"
    description += 'Rice, William J., et al. "Routine determination of ice thickness \n'
    description += 'for cryo-EM grids." Journal of structural biology 204.1 (2018): 38-44. (Leginon)\n\n'
    description += 'Rheinberger, Jan, et al. "Optimized cryo-EM data-acquisition workflow\n'
    description += 'by sample-thickness determination." Acta Crystallographica Section D: \n'
    description += 'Structural Biology 77.5 (2021). (Serial-EM)\n\n'
    description += "---------------------------------------------\n\n"
    for i,par in enumerate(zip(LogII0.T,obj_apertures,Aperturemicron)):
        lam = fit_straight_line(thicknesses/10,par[0])
        description += line.format(i, par[1],appunits,par[2],int(np.round(1/lam)))
    print(description)



def print_help():
    description = "\nUsage: \n"
    description += " python Generate_MeasureIce_calibration.py -E 300 -A 9.9,15.0\n\n"
    description += "Generates calibration files for MeasureIce utility\n"
    description += "for measuring ice thickness in cryo-EM\n\n"
    description += "-E, --Electronenergy  Electron energy (in keV)\n"
    description += "-A, --Aperturesizes   Aperture sizes in inverse Angstrom (default) or mrad\n"
    description += "-u, --Units           Aperture size units (invA by default or mrad), optional\n"
    description += "-m, --Aperturemicron  Aperture sizes in microns (for labelling of output), optional\n"
    description += (
        "-o, --Outputfilename  Output .h5 filename (default Calibration.h5), optional\n"
    )
    description += (
        "-M, --Microscopename  Name of microscope (For book-keeping), optional\n"
    )
    description += "-P, --Plot            Generate reference I/I0 vs thickness plot, optional\n"
    description += "-I, --imfp            User provided electron inelastic mean free path in nm, optional\n"
    description += "-C, --carbon          Option of adding an amorphous carbon layer to simulate carbon backed grids\n"
    print(description)


if __name__ == "__main__":

    # Get command line options, input directory, and whether pngs, tiffs or metadata is to be outputted
    import getopt

    opts, args = getopt.getopt(
        sys.argv[1:],
        "hE:A:u:m:o:M:PI:C:",
        [
            "help",
            "Electronenergy=",
            "Aperturesizes=",
            "Units=",
            "Aperturemicron=",
            "Outputfilename=",
            "Microscopename=",
            "Plot",
            "imfp=",
            "carbon=",
        ],
    )

    microscopename = None
    keV = None
    obj_apertures = None
    app_units = "invA"
    Aperturemicron = None
    outputfilename = "Calibration.hdf"
    microscopename = None
    plot = False
    imfp_prov = None
    carbon = None

    dir_ = sys.argv[0]
    if len(opts)<1:
        print_help()
        sys.exit()
    for o, a in opts:
        if o == "-h" or o == "--help":
            print_help()
            sys.exit()
        elif o == "-E" or o == "--Electronenergy":
            keV = float(a)
        elif o == "-A" or o == "--Aperturesizes":
            obj_apertures = [float(x) for x in a.split(",")]
        elif o == "-u" or o == "--Units":
            A = a.upper()
            errstr = "Units of {0} not recognized, need invA or mrad"
            assert A == "MRAD" or A == "INVA", errstr.format(a)
            if A == "MRAD":
                app_units = "mrad"
        elif o == "-m" or o == "--Aperturemicron":
            Aperturemicron = [int(x) for x in a.split(",")]
        elif o == "-o" or o == "--Outputfilename":
            outputfilename = a
        elif o == "-M" or o == "--Microscopename":
            microscopename = a
        elif o == "-P" or o == "--Plot":
            plot = True
        elif o =='-I' or o == "--imfp":
            imfp_prov = float(a)
        elif o =='-C' or o == '--carbon':
            carbon = float(a)
        else:
            assert False, "unhandled option {0}".format(o)

    assert obj_apertures is not None, "No objective aperture measurements provided"
    assert keV is not None, "No electron energy provided"

    # If micron aperture values are not given just use the aperture
    # values in mrad or invA to label the outputs
    if Aperturemicron is None:
        Aperturemicron = obj_apertures
        labelunits = "micron"
    else:
        labelunits = app_units
    lenerror = "Require an equal number of arguments to --Aperturesizes (got {0}) and --Aperturemicron (got {1})".format(
        len(obj_apertures), len(Aperturemicron)
    )
    assert len(Aperturemicron) == len(obj_apertures), lenerror

    # Plotfile
    plotfile = os.path.splitext(outputfilename)[0] + ".pdf"

    # h5 file
    outputfilename = os.path.splitext(outputfilename)[0] + ".h5"

    # Generic microscope name if none provided
    if microscopename is None:
        microscopename = "{0}_keV_TEM".format(int(keV))

    outputstring = "Generating calibration curves for {0} which has "
    outputstring += "an accelerating voltage of {1} kV and {2} {3} apertures, "
    outputstring += "labelled {4} micron. Results will be outputed to file {5}"
    if plot:
        outputstring += " and will be plotted to file {0}.".format(plotfile)
    else:
        outputstring += "."
    if carbon is not None:
        outputstring += ' A carbon layer of thickness {0} nm will be included in simulations.'.format(carbon)
    outputstring = outputstring.format(
        microscopename,
        keV,
        ", ".join([str(x) for x in obj_apertures]),
        app_units,
        ", ".join([str(x) for x in Aperturemicron]),
        outputfilename,
    )
    import textwrap

    print(textwrap.fill(outputstring))

    # Generate thickness calibration curves
    thicknesses, LogII0 = generate_calibration_curves(
        keV, obj_apertures, app_units=app_units,imfp_prov=imfp_prov,carbon_thickness=carbon
    )

    # Plot calibration curves if requested
    if plot:
        title = "{0} ice thickness - image intensity calibration curve"
        title = title.format(microscopename)
        labels = ["{0} {1}".format(x, labelunits) for x in Aperturemicron]
        fig = make_plot(thicknesses, LogII0, title=title, labels=labels)
        fig.savefig(plotfile)

    output_result(keV, microscopename, thicknesses, LogII0, obj_apertures, app_units,Aperturemicron,carbon=carbon)

    # Save this in an hdf5 file for the MeasureIce GUI
    print('\nOutputting {0} for use with the MeasureIce GUI\n'.format(outputfilename))
    save_calibrationhdf5(
        outputfilename,
        keV,
        microscopename,
        np.asarray(thicknesses) / 10,
        LogII0,
        obj_apertures,
        app_units,
        Aperturemicron
    )

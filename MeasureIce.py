#!/usr/bin/env python
"""
GUI for measuring ice thickness
"""

# To minimize size of distributable .exe file import specific functions
# from libraries where possible
from glob import glob

from numpy import (
    amax,
    asarray,
    nan_to_num,
    log,
    clip,
    average,
)
from matplotlib import colors
import matplotlib.pyplot as plt
import mrcfile as mrc
import numpy as np
from os.path import splitext
from os.path import split as pathsplit
from os.path import exists as pathexists
from os.path import join as pathjoin
from os.path import dirname
from PIL.Image import open as tifopen
from PIL.Image import fromarray
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from scipy.interpolate import interp1d
import ser
from sys import exit as sysexit


def is_image_renormalized(array):
    """"Check for bit compression within image."""
    dtype = array.dtype
    if np.issubdtype(dtype, np.integer):
        drange = np.iinfo(dtype).max - np.iinfo(dtype).min
        return np.abs(np.ptp(array) - drange) < 3
    return False


def update_mask(val):
    """Update mask that indicates regions if image with I >= I0."""
    mask = np.where(img.image >= val, 255, 0).astype(np.uint8)
    zero = np.zeros(img.image.shape, dtype=np.uint8)

    rgba = np.stack([mask, zero, zero, mask], axis=2)

    maskimg.setImage(rgba)


def update_iso_curve():
    """Update the isocurve from the iso_line."""
    global iso_line, Manual_i0
    # iso.setLevel(iso_line.value())
    update_mask(iso_line.value())
    Manual_i0.setValue(int(iso_line.value()))


def initial_image():
    """
    Creates a fake, noisy image to be displayed when the UI is first opened.
    """
    data = np.random.normal(size=(200, 100))
    data[20:80, 20:80] += 2.0
    data = pg.gaussianFilter(data, (3, 3))
    data += np.random.normal(size=(200, 100)) * 0.1
    return data - data.min()


def add_warning_message(msg, color="red", fontsize=16, fill="white"):
    """"Add a warning message above the raw TEM image."""
    hexcolor = colors.to_hex(color)
    # rgbafill = [int(255 * i) for i in colors.to_rgba(fill)]

    html = '<div style="text-align: center">TEM image <span style="color:'
    html += '{1};">{0}<span style="color: {1}; font-size: {2}pt;"></span>'
    html += "</div>"
    html = html.format(msg, hexcolor, fontsize)

    # text = pg.TextItem(html=html, anchor=(-0.3, 0.5), border="w", fill=rgbafill)
    # vbox = p1.getView()
    # vbox.addItem(text)
    # text.setPos(0, y.max())
    global p1
    p1.setTitle(html)


def set_raw_image(data, resetiso_line=True, resetposition=True):
    """Set the raw image display the chosen image."""
    global iso_line, img, hist, iso
    img.setImage(data)

    hist.setLevels(data.min(), data.max())

    if renormalized:
        msg = "<br>Warning, image appears to have been renormalized to satisfy"
        msg += "<br>data type bit depth. Caution required when comparing"
        msg += "<br>intensities between different images"
        add_warning_message(msg)
    else:
        add_warning_message("")
    # set position and scale of image
    if resetposition:
        img.scale(1.0, 1.0)
        img.translate(0, 0)
    if resetiso_line:
        iso_line.setValue(amax(data))
        update_iso_curve()
        # iso.setData(pg.gaussianFilter(data, (2, 2)))


def load_image(yflip=True, transpose=False):
    """Action taken after the load image button is pressed"""

    # Get filename from open file dialog
    fnam = QtGui.QFileDialog.getOpenFileName(
        None, "Open image file", "", "*.tif *.tiff *.ser *.mrc"
    )

    # Get file extension
    ext = splitext(fnam[0])[1]

    if ext == ".tif" or ext == ".tiff":
        openfunc = tifopen
    elif ext == ".mrc":

        def openfunc(file):
            with mrc.open(file) as openfile:
                return openfile.data

    else:
        # Open ser file using openncem function by Peter Ercius
        def openfunc(file):
            return ser.serReader(file)["data"]

    data = asarray(openfunc(fnam[0]))
    if yflip:
        data = data[::-1]

    if transpose:
        data = data.T

    global renormalized

    # Check if image has been renormalized to fit bit depth of datatype
    renormalized = is_image_renormalized(data)

    data = bin2d(data, binfactorspin.value())
    set_raw_image(data)


def measure_ice_thickness():
    """Measure ice thickness from raw image and precalculated calibration curveses"""
    global icemapgausskernel, iso_line, chooseAperture
    I0 = iso_line.value()

    # Get mapping for chosen calibration and aperture
    ical = chooseCalibration.currentIndex()
    iapp = chooseAperture.currentIndex()
    t = calibrations[ical].intensity_to_thickness[iapp]

    # Gaussian filter TEM image before converting to ice thickness map
    # 2*(kern,) makes a tuple for isotropic x and y blurring kernel
    kern = icemapgausskernel.value()
    if kern>0:
        img_filt = pg.gaussianFilter(img.image, 2 * (kern,))
    else:
        img_filt = img.image

    # Calculate thickness map using calibration curve
    # clip function ensures intensities > I0 (due to noise) are not
    # mapped into negative values for ice thickness.
    thickness_map = nan_to_num(
        t(log(clip(I0 / img_filt, 1.0, None))), neginf=0, posinf=-1
    )

    # Display ice thickness map
    tmap.setImage(thickness_map)
    tmap.scale(1.0, 1.0)
    tmap.translate(0, 0)
    tmap.hoverEvent = imageHoverEvent


def save_ice_thickness_map():
    """Save the generated ice thickness map, in pdf,png or tiff format"""
    endings = ("pdf (*.pdf)", "png (*.png)", "tif (*.tif)")
    # Get output filename from GUI dialog
    fnam, ending = QtGui.QFileDialog.getSaveFileName(
        None, "Save thickness map", "", ";;".join(endings)
    )

    # Add suffix
    fnamout = splitext(fnam)[0] + ending[-5:-1]

    # Output in requested format
    if ending == "tif (*.tif)":
        # Python Image Library (PIL) handles tiff
        fromarray(tmap.image[::-1]).save(fnamout)
    else:
        # matplotlib handles png and pdf
        fig, ax = plt.subplots()
        vmin, vmax = hist2.getLevels()
        kwargs = {"cmap": plt.get_cmap("Blues"), "vmin": vmin, "vmax": vmax}
        pos = ax.imshow(tmap.image[::-1], **kwargs)
        ax.set_axis_off()
        fig.colorbar(pos, ax=ax, label="Ice thickness (nm)")
        fig.savefig(fnamout)


def set_I0_manually():
    """Set the vacuum intensity I0 via textbox"""
    global Manual_i0, iso_line, iso
    newval = float(Manual_i0.value())
    iso_line.setValue(newval)
    # iso.setLevel(newval)


def bin2d(array, factor):
    """
    Bin a 2D array by binfactor

    Parameters
    ----------
    array : (ny,nx) float array_like
        Array to be binned
    factor : int
        The factor by which the size of array is binned by, if either dimension
        of array, ny or nx, is not integer divisible by binfactor then the
        array will be truncated
    Returns
    -------
    arrayout : (ny//binfactor,nx//binfactor) float array_like
        The binned array
    """
    # Get arrayshape
    s = array.shape

    # Work out if any dimensions need to be trunacted prior to binning
    t = [(s[0] // factor) * factor, (s[1] // factor) * factor]

    # Shape of array so that binning can be performed by numpy functions
    ns = (s[0] // factor, factor, s[1] // factor, factor)

    # Return binned array
    return average(
        array[: t[0], : t[1]].reshape(*ns),
        axis=(1, 3),
    )


def nocalibrationfile(fnam):
    # Get filename from open file dialog
    fnam = QtGui.QFileDialog.getOpenFileName(
        None, "Open calibration file", "", "*.h5 *.hdf"
    )
    return fnam[0]
    """Error message for when calibration files cannot be found."""
    errmsg = "Can't find any .h5 calibration files in directory {0}".format(fnam)
    print(errmsg)
    w1 = QtGui.QLabel(errmsg)
    w1.show()
    QtGui.QApplication.instance().exec_()
    sysexit()


class II0_calibration:
    def __init__(self, h5file):
        from h5py import File as h5open

        with h5open(h5file, "r") as f:
            # Get data labels
            self.name = f.attrs["Microscope name"]
            self.applabels = np.asarray(f["Apertures micron"]).astype(str)
            logI0I = np.asarray(f["LogI0/I"]).T
            t = f["Thicknesses"]
            # Create functions which interpolate input data
            self.intensity_to_thickness = [
                interp1d(d, t, fill_value="extrapolate") for d in logI0I
            ]


def load_calibration_data(path=None):
    """
    Load presimulated calibration data

    Parameters
    ----------
    fnam : string
        filename of .h5 calibration data

    Returns
    -------
    calibrations : (ncal,) II0_calibration instance array_like
        list of calibrations
    """

    if path is None:
        path_ = pathsplit(sys.argv[0])[0]
        fnams = glob(pathjoin(path_, "*.h5"))
    else:
        path_ = path
        fnams = glob(pathjoin(path_, "*.h5"))

    if len(fnams) < 1:
        fnams = glob(pathjoin(dirname(nocalibrationfile(path_)), "*.h5"))

    calibrations = []
    for f in fnams:
        calibrations.append(II0_calibration(f))
    return calibrations


def changeCalibration():
    """Update aperture labels when calibration changed"""
    # Get index of calibration
    i = chooseCalibration.currentIndex()
    chooseAperture.clear()
    chooseAperture.addItems(calibrations[i].applabels)


renormalized = False
# Interpret image data as row-major instead of col-major
pg.setConfigOptions(imageAxisOrder="row-major")

# PyQTgraph app initiation
app = pg.mkQApp()
win = pg.GraphicsLayoutWidget()
win.setWindowTitle("MeasureIce")
win.resize(1600, 600)

# icon for plot window
app_icon = QtGui.QIcon("icons/icon.ico")
app_icon.addFile("icons/24x24.png", QtCore.QSize(24, 24))
app_icon.addFile("icons/32x32.png", QtCore.QSize(32, 32))
app_icon.addFile("icons/48x48.png", QtCore.QSize(48, 48))
app_icon.addFile(
    "icons/256x256.png", QtCore.QSize(256, 256)
)
win.setWindowIcon(app_icon)

####################
# Raw image window #
####################
# A plot area (ViewBox + axes) for displaying the image
p1 = win.addPlot(title="", row=0, col=0, colspan=2)

# pyqtgraph item for displaying image data
img = pg.ImageItem()
p1.addItem(img)
maskimg = pg.ImageItem()
p1.addItem(maskimg)
maskimg.setZValue(10)
p1.setTitle("TEM image")

# Histogram control for raw image
hist = pg.HistogramLUTItem()
hist.setImageItem(img)
win.addItem(hist, row=0, col=2, colspan=1)

# Isocurve drawing to highlight I0 "vacuum level" on image
# iso = pg.IsocurveItem(level=0.8, pen="r")
# iso.setParentItem(img)
# iso.setZValue(5)

# Draggable line for setting isocurve level
iso_line = pg.InfiniteLine(angle=0, label="I0", movable=True, pen="r")
hist.vb.addItem(iso_line)
hist.vb.setMouseEnabled(y=False)  # makes user interaction a little easier
iso_line.setValue(0.8)
iso_line.setZValue(1000)  # bring iso line above contrast controls


iso_line.sigDragged.connect(update_iso_curve)

############################
# Ice thickness map window #
############################
p2 = win.addPlot(title="", row=0, col=3, colspan=2)
p2.setMaximumHeight(800)
p2.setTitle("Ice thickness map")

hist2 = pg.HistogramLUTItem()

tmap = pg.ImageItem()
p2.addItem(tmap)
p3 = win.addItem(hist2, row=0, col=5, colspan=1)
hist2.setImageItem(tmap)

# Show window
win.show()


def imageHoverEvent(event):
    """Show the position, pixel, and value under the mouse cursor."""
    if event.isExit():
        """Default for when the mouse is outside the image"""
        p2.setTitle("Ice thickness map")

    # Get mouse position
    pos = event.pos()
    i, j = pos.y(), pos.x()
    i = int(clip(i, 0, tmap.image.shape[0] - 1))
    j = int(clip(j, 0, tmap.image.shape[1] - 1))

    # Get value under mouse
    val = tmap.image[i, j]

    # Set title
    string = "Thickness: {0:^5d}nm".format(int(val))
    p2.setTitle(string)
    p1.setMaximumWidth(800)


# Monkey-patch the image to use our custom hover function.
# This is generally discouraged (you should subclass ImageItem instead),
# but it works for a very simple use like this.

###########
# Buttons #
###########
tmap.hoverEvent = imageHoverEvent
proxyloadBtn = QtGui.QGraphicsProxyWidget()
proxybinBtn = QtGui.QGraphicsProxyWidget()

# Button to load raw ice thickness
loadBtn = QtGui.QPushButton("Load raw image")
proxyloadBtn.setWidget(loadBtn)
loadBtn.clicked.connect(lambda: load_image(yflip=True))

# Button to bin raw image
# binBtn = QtGui.QPushButton("Bin image")
# proxybinBtn.setWidget(binBtn)
# binBtn.clicked.connect(bin_image)

# Measure ice thickness button
proxyMeasureBtn = QtGui.QGraphicsProxyWidget()
MeasureBtn = QtGui.QPushButton("Measure ice thickness")
proxyMeasureBtn.setWidget(MeasureBtn)
MeasureBtn.clicked.connect(measure_ice_thickness)

# Save ice thickness button
proxySaveBtn = QtGui.QGraphicsProxyWidget()
SaveBtn = QtGui.QPushButton("Save ice thickness map")
proxySaveBtn.setWidget(SaveBtn)
SaveBtn.clicked.connect(save_ice_thickness_map)

#######################
# Text and spin boxes #
#######################

# Calibrations menu
chooseCalibration = QtGui.QComboBox()
proxychooseCalibration = QtGui.QGraphicsProxyWidget()
proxychooseCalibration.setWidget(chooseCalibration)
chooseCalibrationlabel = QtGui.QLabel()
chooseCalibrationlabel.setText("Calibration:")
chooseCalibrationlabel.setAlignment(QtCore.Qt.AlignCenter)
chooseCalibrationlabel.setMaximumHeight(20)
chooseCalibrationlabel.setMaximumWidth(70)
chooseCalibrationlabel.setStyleSheet("background-color: black; color:white")
proxychooseCalibrationlabel = QtGui.QGraphicsProxyWidget()
proxychooseCalibrationlabel.setWidget(chooseCalibrationlabel)
chooseCalibration.currentIndexChanged.connect(changeCalibration)

# Image bining spinbox
binfactorspin = QtGui.QSpinBox(maximum=16, minimum=1)
binfactorspinlabel = QtGui.QLabel()
binfactorspinlabel.setStyleSheet("background-color: black; color:white")
binfactorspinlabel.setAlignment(QtCore.Qt.AlignCenter)
binfactorspinlabel.setText("Raw image binning:")
binfactorspinlabel.setMaximumHeight(20)
binfactorspinlabel.setMaximumWidth(80)
binfactorspinlabel.setBuddy(binfactorspinlabel)
proxybinfactorspinlabel = QtGui.QGraphicsProxyWidget()
proxybinfactorspinlabel.setWidget(binfactorspinlabel)
proxybinfactorspin = QtGui.QGraphicsProxyWidget()
proxybinfactorspin.setWidget(binfactorspin)
# binfactorspin.valueChanged.connect(set_I0_manually)

# Manual I0 textbox
Manual_i0 = QtGui.QSpinBox(maximum=1e9, minimum=0)
Manual_i0label = QtGui.QLabel()
Manual_i0label.setStyleSheet("background-color: black; color:white")
Manual_i0label.setAlignment(QtCore.Qt.AlignCenter)
Manual_i0label.setText("I0:")
Manual_i0label.setMaximumHeight(20)
Manual_i0label.setMaximumWidth(15)
Manual_i0label.setBuddy(Manual_i0)
proxyManual_i0label = QtGui.QGraphicsProxyWidget()
proxyManual_i0label.setWidget(Manual_i0label)
proxyManual_i0 = QtGui.QGraphicsProxyWidget()
proxyManual_i0.setWidget(Manual_i0)
Manual_i0.valueChanged.connect(set_I0_manually)

# Aperture menu
chooseAperture = QtGui.QComboBox()
proxychooseAperture = QtGui.QGraphicsProxyWidget()
proxychooseAperture.setWidget(chooseAperture)
chooseAperturelabel = QtGui.QLabel()
chooseAperturelabel.setText("Aperture (microns):")
chooseAperturelabel.setAlignment(QtCore.Qt.AlignCenter)
chooseAperturelabel.setMaximumHeight(20)
chooseAperturelabel.setMaximumWidth(120)
chooseAperturelabel.setStyleSheet("background-color: black; color:white")
proxychooseAperturelabel = QtGui.QGraphicsProxyWidget()
proxychooseAperturelabel.setWidget(chooseAperturelabel)

# Image bining spinbox
binfactorspin = QtGui.QSpinBox(maximum=16, minimum=1)
binfactorspinlabel = QtGui.QLabel()
binfactorspinlabel.setStyleSheet("background-color: black; color:white")
binfactorspinlabel.setAlignment(QtCore.Qt.AlignCenter)
binfactorspinlabel.setText("Raw image binning:")
binfactorspinlabel.setMaximumHeight(20)
binfactorspinlabel.setMaximumWidth(130)
binfactorspinlabel.setBuddy(binfactorspinlabel)
proxybinfactorspinlabel = QtGui.QGraphicsProxyWidget()
proxybinfactorspinlabel.setWidget(binfactorspinlabel)
proxybinfactorspin = QtGui.QGraphicsProxyWidget()
proxybinfactorspin.setWidget(binfactorspin)
binfactorspin.setValue(2)

# Gaussian filtering for ice thickness image
icemapgausskernel = QtGui.QSpinBox(maximum=16, minimum=0)
icemapgausskernellabel = QtGui.QLabel()
icemapgausskernellabel.setStyleSheet("background-color: black; color:white")
icemapgausskernellabel.setAlignment(QtCore.Qt.AlignCenter)
icemapgausskernellabel.setText("Map spatial-filter kernel:")
icemapgausskernellabel.setMaximumHeight(20)
icemapgausskernellabel.setMaximumWidth(160)
icemapgausskernellabel.setBuddy(icemapgausskernellabel)
proxyicemapgausskernellabel = QtGui.QGraphicsProxyWidget()
proxyicemapgausskernellabel.setWidget(icemapgausskernellabel)
proxyicemapgausskernel = QtGui.QGraphicsProxyWidget()
proxyicemapgausskernel.setWidget(icemapgausskernel)
icemapgausskernel.setValue(2)
icemapgausskernel.valueChanged.connect(measure_ice_thickness)

# Layout of control panel
controlpanel = win.addLayout(row=1, col=0, colspan=7)

controlpanelwidgets = []
for i, widg in enumerate(
    [
        proxychooseCalibrationlabel,
        proxychooseCalibration,
        proxyloadBtn,
        # proxybinBtn,
        proxybinfactorspinlabel,
        proxybinfactorspin,
        proxyManual_i0label,
        proxyManual_i0,
        proxychooseAperturelabel,
        proxychooseAperture,
        proxyMeasureBtn,
        proxyicemapgausskernellabel,
        proxyicemapgausskernel,
        proxySaveBtn,
    ]
):
    controlpanelwidgets.append(controlpanel.addLayout(row=0, col=i, colspan=1))
    controlpanelwidgets[i].addItem(widg)

# Generate initial image data and display it
set_raw_image(initial_image())

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == "__main__":
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        if len(sys.argv) > 1:
            h5path = sys.argv[1]
        else:
            h5path = pathsplit(sys.argv[0])[0]
        # Load calibration data
        calibrations = load_calibration_data(h5path)
        chooseCalibration.addItems([x.name for x in calibrations])
        QtGui.QApplication.instance().exec_()

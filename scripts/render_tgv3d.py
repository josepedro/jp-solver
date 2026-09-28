# Render the 3D Taylor-Green GLO-AMR movie in ParaView: Q-criterion vortex tubes coloured by speed,
# with the dynamic refinement blocks drawn as a wireframe that tracks the tubes over time.
#
# Usage (headless, writes a PNG sequence and an mp4 if ffmpeg is present):
#   pvpython scripts/render_tgv3d.py --dir ~/tgv_render --prefix tgv --out ~/tgv_movie --iso 2e-5
# then, if ParaView did not mux the mp4:
#   ffmpeg -framerate 30 -i ~/tgv_movie.%04d.png -pix_fmt yuv420p -crf 18 ~/tgv_movie.mp4
#
# Q is small; the right isovalue is a few times 1e-5 for these runs. Pass --iso to tune, or run once and
# look at the Q range printed below.

import argparse, glob, os, sys
from paraview.simple import *

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True, help="directory holding <prefix>_####.vti and .vtp")
ap.add_argument("--prefix", default="tgv")
ap.add_argument("--out", default="tgv_movie", help="output path prefix for the PNG/mp4")
ap.add_argument("--iso", type=float, default=None, help="Q isovalue (default: 8%% of max Q in frame 0)")
ap.add_argument("--res", type=int, nargs=2, default=[1600, 1200])
ap.add_argument("--sweep", type=float, default=100.0, help="total azimuth rotation over the whole clip (degrees)")
ap.add_argument("--color", default="speed", help="array to colour the isosurface by")
ap.add_argument("--smin", type=float, default=None, help="colour scale min (else 5th percentile on the surface)")
ap.add_argument("--smax", type=float, default=None, help="colour scale max (else 95th percentile on the surface)")
ap.add_argument("--preset", default="Viridis (matplotlib)")
args = ap.parse_args()

vti = sorted(glob.glob(os.path.join(args.dir, args.prefix + "_*.vti")))
vtp = sorted(glob.glob(os.path.join(args.dir, args.prefix + "_*.vtp")))
if not vti:
    sys.exit("no .vti frames found in %s" % args.dir)
print("frames: %d vti, %d vtp" % (len(vti), len(vtp)))

field = XMLImageDataReader(FileName=vti)          # a time series, one file per timestep
field.PointArrayStatus = ["Q", "speed"]
UpdatePipeline()
qrange = field.PointData["Q"].GetRange()
print("Q range frame set:", qrange)
iso = args.iso if args.iso is not None else max(1e-6, 0.08 * qrange[1])
print("using Q isovalue:", iso)

view = GetActiveViewOrCreate("RenderView")
view.ViewSize = args.res
try: view.UseColorPaletteForBackground = 0
except Exception: pass
try: view.BackgroundColorMode = "Single Color"
except Exception: pass
view.Background = [0.03, 0.03, 0.05]
view.OrientationAxesVisibility = 0

# Q-criterion isosurface, coloured by speed
contour = Contour(Input=field)
contour.ContourBy = ["POINTS", "Q"]
contour.Isosurfaces = [iso]
cdisp = Show(contour, view)
ColorBy(cdisp, ("POINTS", args.color))
lut = GetColorTransferFunction(args.color)
lut.ApplyPreset(args.preset, True)
# Fixed colour range for the whole clip, spread so the surface is not all one hue. Prefer explicit
# percentile bounds (--smin/--smax); otherwise take the contour's range at a developed frame and saturate
# the top so the mid speeds get the middle of the ramp.
_tv = list(field.TimestepValues) if field.TimestepValues else [0.0]
if args.smin is not None and args.smax is not None:
    lut.RescaleTransferFunction(args.smin, args.smax)
else:
    contour.UpdatePipeline(_tv[int(0.7 * (len(_tv) - 1))])
    r = contour.PointData[args.color].GetRange()
    lut.RescaleTransferFunction(r[0], r[0] + 0.72 * (r[1] - r[0]))
lut.AutomaticRescaleRangeMode = "Never"
cdisp.SetScalarBarVisibility(view, True)
cdisp.Opacity = 1.0

# dynamic refinement blocks as a wireframe overlay
if vtp:
    boxes = XMLPolyDataReader(FileName=vtp)
    UpdatePipeline()
    bdisp = Show(boxes, view)
    bdisp.Representation = "Wireframe"
    bdisp.LineWidth = 1.5
    bdisp.Opacity = 0.7
    try: boxes.PointArrayStatus = ["level"]
    except Exception: pass
    UpdatePipeline()
    arrays = list(boxes.PointData.keys())
    if "level" in arrays:                 # colour refinement levels distinctly (L1 amber, L2 cyan)
        ColorBy(bdisp, ("POINTS", "level"))
        llut = GetColorTransferFunction("level")
        llut.InterpretValuesAsCategories = 1
        llut.AnnotationsInitialized = 1
        llut.Annotations = ["1", "L1", "2", "L2"]
        llut.IndexedColors = [1.0, 0.55, 0.1,  0.15, 0.9, 1.0]
        llut.IndexedOpacities = [1.0, 1.0]
        bdisp.SetScalarBarVisibility(view, False)
    else:
        bdisp.AmbientColor = [1.0, 0.55, 0.1]
        bdisp.DiffuseColor = [1.0, 0.55, 0.1]

scene = GetAnimationScene()
scene.UpdateAnimationUsingDataTimeSteps()
tsteps = list(field.TimestepValues) if field.TimestepValues else [0.0]

# Frame the full domain (fixed ImageData bounds, stable across time) from a late, fully developed state,
# then walk every timestep with a slow azimuth sweep for a cinematic 3D orbit.
view.ViewTime = tsteps[-1]; Render()
ResetCamera(view)
cam = GetActiveCamera()
cam.Elevation(22)
view.CameraParallelProjection = 0
sweep = float(args.sweep) / max(1, len(tsteps))     # degrees per frame

for i, t in enumerate(tsteps):
    view.ViewTime = t
    cam.Azimuth(sweep)
    Render()
    SaveScreenshot("%s_%04d.png" % (args.out, i), view, ImageResolution=args.res)
    if i % 20 == 0: print("rendered frame", i, "of", len(tsteps))
print("wrote PNG sequence %s_####.png (%d frames)" % (args.out, len(tsteps)))

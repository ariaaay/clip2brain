import os
import time
import numpy as np
import cortex

import warnings

warnings.warn(
    """
`glabtools.save_3d_views.save_3d_views()` has moved to
`glabtools.viz.pycortex_extras.save_3d_views`""",
    UserWarning,
)

file_pattern = "{base}_{view}_{surface}.png"
_combine = lambda a, b: (lambda c: [c, c.update(b)][0])(dict(a))
_tolists = lambda p: {k: [v] for k, v in p.items()}


def save_3d_views(
    data,
    root,
    base_name,
    list_views=["lateral"],
    list_surfaces=["inflated"],
    with_labels=False,
    size=(1024 * 4, 768 * 4),
    trim=True,
):
    """Saves 3D views of `data` in and around `root` under multiple specifications. Needs to be run
       on a system with a display (will launch webgl viewer)
    data: a pycortex volume
    root: directory where things should be saved
    base_name: base name for images
    list_views: which views do you want? choices are: lateral, lateral_left, lateral_right,
               medial, front, back,top, bottom
    list_surfaces: what surfaces do you want? choices are inflated, flatmap, fiducial
    with_labels: show ROI labels?
    size: size of produced image (before trimming)
    trim: whether to trim
    returns filenames: a dict of the produced image paths
    """

    warnings.warn(
        """
    `glabtools.save_3d_views.save_3d_views()` has moved to
    `glabtools.viz.pycortex_extras.save_3d_views`""",
        UserWarning,
    )

    # Create root dir?
    if not os.path.exists(root):
        os.mkdir(root)

    # Create viewer
    if with_labels:
        labels_visible = ("rois",)
    else:
        labels_visible = ()
    handle = cortex.webgl.show(data, labels_visible=labels_visible)

    time.sleep(5.0)
    set_opacity = 1
    set_radius = 700

    basic = (
        dict()
    )  # radius=400)#projection=['orthographic'], #radius=260, visL=True, visR=True)

    views = dict(
        lateral=dict(
            altitude=90.5,
            azimuth=181,
            pivot=180.5,
            radius=set_radius,
            opacity=set_opacity,
        ),
        lateral_left=dict(
            altitude=90.5,
            azimuth=90.5,
            pivot=0.5,
            radius=set_radius,
            opacity=set_opacity,
        ),
        lateral_right=dict(
            altitude=90.5,
            azimuth=270.5,
            pivot=0.5,
            radius=set_radius,
            opacity=set_opacity,
        ),
        medial=dict(
            altitude=90.5,
            azimuth=0.5,
            pivot=180.5,
            radius=set_radius,
            opacity=set_opacity,
        ),
        front=dict(
            altitude=90.5, azimuth=0, pivot=0, radius=set_radius, opacity=set_opacity
        ),
        back=dict(
            altitude=90.5, azimuth=181, pivot=0, radius=set_radius, opacity=set_opacity
        ),
        top=dict(
            altitude=0, azimuth=180, pivot=0, radius=set_radius, opacity=set_opacity
        ),
        bottom=dict(
            altitude=180, azimuth=0, pivot=0, radius=set_radius, opacity=set_opacity
        ),
    )

    surfaces = dict(
        inflated=dict(unfold=0.5), flatmap=dict(unfold=1), fiducial=dict(unfold=0)
    )

    param_dict = dict(
        unfold="surface.{subject}.unfold",
        altitude="camera.altitude",
        azimuth="camera.azimuth",
        radius="camera.radius",
        pivot="surface.{subject}.pivot",
        opacity="surface.{subject}.opacity",
    )
    # radius = 'surface.{subject}.radius') # unknown parameter

    # Save views!
    filenames = dict([(key, dict()) for key in surfaces.keys()])

    for view in list_views:
        # copy proper parameters with new names
        vparams = dict([(param_dict[k], v) for k, v in views[view].items()])
        for surf in list_surfaces:
            # copy proper parameters with new names
            sparams = dict([(param_dict[k], v) for k, v in surfaces[surf].items()])
            # Combine basic, view, and surface parameters
            params = _combine(_combine(basic, vparams), sparams)

            # Set the view
            handle._set_view(**_tolists(params))
            time.sleep(1.5)

            # Save image, store filename
            filename = file_pattern.format(base=base_name, view=view, surface=surf)
            filenames[surf][view] = filename
            # filenames.append(filename)

            output_path = os.path.join(root, filename)
            handle.getImage(output_path, size)

            # Trim edges?
            if trim:
                # Wait for browser to dump file
                while not os.path.exists(output_path):
                    pass

                time.sleep(0.5)

                try:
                    import subprocess

                    subprocess.call(["convert", "-trim", output_path, output_path])
                except:
                    pass

    # Close the window!
    try:
        handle.close()
    except:
        print("Could not close viewer")

    return filenames

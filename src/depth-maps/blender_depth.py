# A script to generate a dephtmap for every baked camera timeslice
# Sample usage:
# Open *cam-bamked*blend on blender 2.76
# Go to the python interactive console
# source this file
#
# This re-generates the cameras as in blender_cameras.py just for checking.

# Generate ground truth data from ground truth blender file
#
# For instance, open the project
#
# ground-truth-pavillion/3d/full/pavillon_barcelone_v1.2-cam-baked-007.blend
#
# Then select the desired animated camera and set as active camera
#
# Then run this script. It will play frame by frame and export the 3x4 camera
# matrix for that frame, among other things.
# 2023: AND DEPTHMAP
#
### Run with:
# filename = "/Users/rfabbri/lib/data/pavilion-multiview-3d-dataset/src/depth-maps/blender_depth.py"
# exec(compile(open(filename).read(), filename, 'exec'))
#
### Documentation:
##### Rendering depth maps per video frame
#
### https://projects.blender.org/blender/blender/issues/100417
# https://github.com/panmari/stanford-shapenet-renderer
# bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
#
### https://blender.stackexchange.com/questions/52328/render-depth-maps-with-world-space-z-distance-with-respect-the-camera
###########



#
# WORK IN PROGRESS -------------------------------------------------------------
import bpy_extras
import numpy
from mathutils import Matrix

#------------------------------------------------------------------------
# 3x4 P matrix to Blender camera


# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
#
# Minor notes:
#   See also libmv/src/ui/tvr/tvr_document.h for the inverse
#   Some relevant blender branches: blender:multiview, libmv_prediction
def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal), 
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio 
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal), 
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((alpha_u, skew,    u_0),
        (    0  , alpha_v, v_0),
        (    0  , 0,        1 )))
    return K

# Returns camera rotation and translation matrices from Blender.
# 
# There are 3 coordinate systems involved:
#    * The World coordinates: "world"
#       - right-handed
#    * The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    * The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))

    # Transpose since the rotation is object rotation, and we want coordinate
    # rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    T_world2bcam = -1*R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv@R_world2bcam
    T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
         ))
    return RT

# You can use it this way but it won't work for certain animations
# unless you bake them. I baked my path-constrained camera 
# by object -> animation bake then selecting all options but the last
def get_3x4_RT_matrix_from_blender_without_matrix_world(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))

    # Transpose since the rotation is object rotation, and we want coordinate
    # rotation
#     R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    T_world2bcam = -1*R_world2bcam * location
#     T_world2bcam = -1*R_world2bcam * cam.location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv*R_world2bcam
    T_world2cv = R_bcam2cv*T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
         ))
    return RT

def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K@RT, K, RT

# def qr( self, ROnly=0 ):
#     'QR decomposition using Householder reflections: Q*R==self, Q.tr()*Q==I(n), R upper triangular'
#     R = self
#     m, n = R.size
#     for i in range(min(m,n)):
#         v, beta = R.tr()[i].house(i)
#         R -= v.outer( R.tr().mmul(v)*beta )
#     for i in range(1,min(n,m)): R[i][:i] = [0] * i
#     R = Mat(R[:n])
#     if ROnly: return R
#     Q = R.tr().solve(self.tr()).tr()       # Rt Qt = At    nn  nm  = nm
#     self.qr = lambda r=0, c=self: not r and c==self and (Q,R) or Matrix.qr(self,r) #Cache result
#     assert NPOST or m>=n and Q.size==(m,n) and isinstance(R,UpperTri) or m<n and Q.size==(m,m) and R.size==(m,n)
#     assert NPOST or Q.mmul(R)==self and Q.tr().mmul(Q)==eye(min(m,n))
#     return Q, R

# Input: P 3x4 numpy matrix
# Output: K, R, T such that P = K*[R | T], det(R) positive and K has positive diagonal
#
# Reference implementations: 
#   - Oxford's visual geometry group matlab toolbox 
#   - Scilab Image Processing toolbox
def KRT_from_P(P):
    N = 3
    H = P[:,0:N]  # if not numpy,  H = P.to_3x3()

    [K,R] = rf_rq(H)
      
    K /= K[-1,-1]
    
    # from http://ksimek.github.io/2012/08/14/decompose/
    # make the diagonal of K positive
    sg = numpy.diag(numpy.sign(numpy.diag(K)))

    K = K * sg
    R = sg * R
    # det(R) negative, just invert - the proj equation remains same:
    if (numpy.linalg.det(R) < 0):
        R = -R
    # C = -H\P[:,-1]
    C = numpy.linalg.lstsq(-H, P[:,-1])[0]
    T = -R*C
    return K, R, T

# RQ decomposition of a numpy matrix, using only libs that already come with
# blender by default
#
# Author: Ricardo Fabbri
# Reference implementations: 
#   Oxford's visual geometry group matlab toolbox 
#   Scilab Image Processing toolbox
#
# Input: 3x4 numpy matrix P
# Returns: numpy matrices r,q
def rf_rq(P):
    P = P.T
    # numpy only provides qr. Scipy has rq but doesn't ship with blender
    q, r = numpy.linalg.qr(P[ ::-1, ::-1], 'complete')
    q = q.T
    q = q[ ::-1, ::-1]
    r = r.T
    r = r[ ::-1, ::-1]

    if (numpy.linalg.det(q) < 0):
        r[:,0] *= -1
        q[0,:] *= -1
    return r, q

# Creates a blender camera consistent with a given 3x4 computer vision P matrix
# Run this in Object Mode
# scale: resolution scale percentage as in GUI, known a priori
# P: numpy 3x4
def get_blender_camera_from_3x4_P(P, scale, suff=''):
    # get krt
    K, R_world2cv, T_world2cv = KRT_from_P(numpy.matrix(P))

    scene = bpy.context.scene
    sensor_width_in_mm = K[1,1]*K[0,2] / (K[0,0]*K[1,2])
    sensor_height_in_mm = 1  # doesn't matter
    resolution_x_in_px = K[0,2]*2  # principal point assumed at the center
    resolution_y_in_px = K[1,2]*2  # principal point assumed at the center

    s_u = resolution_x_in_px / sensor_width_in_mm
    s_v = resolution_y_in_px / sensor_height_in_mm
    # TODO include aspect ratio
    f_in_mm = K[0,0] / s_u
    # recover original resolution
    scene.render.resolution_x = resolution_x_in_px / scale
    scene.render.resolution_y = resolution_y_in_px / scale
    scene.render.resolution_percentage = scale * 100

    # Use this if the projection matrix follows the convention listed in my answer to
    # http://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))

#    Use this if the projection matrix follows the convention from e.g. the matlab calibration toolbox:
#     R_bcam2cv = Matrix(
#         ((-1, 0,  0),
#          (0, 1, 0),
#          (0, 0, 1)))

    R_cv2world = R_world2cv.T
    rotation =  Matrix(R_cv2world.tolist()) * R_bcam2cv
    location = -R_cv2world * T_world2cv

    # create a new camera
    bpy.ops.object.add(
        type='CAMERA',
        location=location)
    ob = bpy.context.object
    ob.name = 'CamFrom3x4PObj' + suff
    cam = ob.data
    cam.name = 'CamFrom3x4P' + suff
 
    # Lens
    cam.type = 'PERSP'
    cam.lens = f_in_mm 
    cam.lens_unit = 'MILLIMETERS'
    cam.sensor_width  = sensor_width_in_mm
#     cam.draw_size = 0.02 # XXX only for robot data
    cam.draw_size = 0.1 #helix 
    ob.matrix_world = Matrix.Translation(location)*rotation.to_4x4()

    #     cam.shift_x = -0.05
    #     cam.shift_y = 0.1
    #     cam.clip_start = 10.0
    #     cam.clip_end = 250.0
    #     empty = bpy.data.objects.new('DofEmpty', None)
    #     empty.location = origin+Vector((0,10,0))
    #     cam.dof_object = empty
 
    # Display
    cam.show_name = True
    # Make this the current camera
    #scene.camera = ob
    bpy.context.scene.update()
    
# scale: resolution scale percentage as in GUI, known a priori
# P: numpy 3x4
def orig_get_blender_camera_from_3x4_P(P, scale):
    # get krt
    K, R_world2cv, T_world2cv = KRT_from_P(numpy.matrix(P))

    scene = bpy.context.scene
    sensor_width_in_mm = K[1,1]*K[0,2] / (K[0,0]*K[1,2])
    sensor_height_in_mm = 1  # doesn't matter
    resolution_x_in_px = K[0,2]*2  # principal point assumed at the center
    resolution_y_in_px = K[1,2]*2  # principal point assumed at the center

    s_u = resolution_x_in_px / sensor_width_in_mm
    s_v = resolution_y_in_px / sensor_height_in_mm
    # TODO include aspect ratio

    f_in_mm = K[0,0] / s_u
    # recover original resolution
    scene.render.resolution_x = resolution_x_in_px / scale
    scene.render.resolution_y = resolution_y_in_px / scale
    scene.render.resolution_percentage = scale * 100

    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))

    R_cv2world = R_world2cv.T
    rotation =  Matrix(R_cv2world.tolist()) * R_bcam2cv
    location = -R_cv2world * T_world2cv

    # create a new camera
    bpy.ops.object.add(
        type='CAMERA',
        location=location)
    ob = bpy.context.object
    ob.name = 'CamFrom3x4POb'
    cam = ob.data
    cam.name = 'CamFrom3x4P'
 
    # Lens
    cam.type = 'PERSP'
    cam.lens = f_in_mm 
    cam.lens_unit = 'MILLIMETERS'
    cam.sensor_width  = sensor_width_in_mm
    ob.matrix_world = Matrix.Translation(location)*rotation.to_4x4()

#     cam.shift_x = -0.05
#     cam.shift_y = 0.1
#     cam.clip_start = 10.0
#     cam.clip_end = 250.0
 
#     empty = bpy.data.objects.new('DofEmpty', None)
#     empty.location = origin+Vector((0,10,0))
#     cam.dof_object = empty
 
    # Display
    cam.show_name = True
 
    # Make this the current camera
    scene.camera = ob

    bpy.context.scene.update()


#------------------------------------------------------------------------
# From http://blender.stackexchange.com/questions/882/how-to-find-image-coordinates-of-the-rendered-vertex?lq=1
def project_by_object_utils(cam, point):
    scene = bpy.context.scene
    # obj = bpy.context.object
    # co = bpy.context.scene.cursor_location
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, point)
    # print("2D Coords:", co_2d)

    # If you want pixel coords
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
            int(scene.render.resolution_x * render_scale),
            int(scene.render.resolution_y * render_scale),
            )
    return Vector((co_2d.x * render_size[0], render_size[1] - co_2d.y * render_size[1]))
#     print("Pixel Coords:", (
#           round(co_2d.x * render_size[0]),
#           round(co_2d.y * render_size[1]),
#           ))


#------------------------------------------------------------------------

s = "2- time: sunset"

def view_plane(camd, winx, winy, xasp, yasp):    
    #/* fields rendering */
    ycor = yasp / xasp
    use_fields = False
    if (use_fields):
      ycor *= 2

    def BKE_camera_sensor_size(p_sensor_fit, sensor_x, sensor_y):
        #/* sensor size used to fit to. for auto, sensor_x is both x and y. */
        if (p_sensor_fit == 'VERTICAL'):
            return sensor_y;

        return sensor_x;

    if (camd.type == 'ORTHO'):
      #/* orthographic camera */
      #/* scale == 1.0 means exact 1 to 1 mapping */
      pixsize = camd.ortho_scale
    else:
      #/* perspective camera */
      sensor_size = BKE_camera_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
      pixsize = (sensor_size * camd.clip_start) / camd.lens

    #/* determine sensor fit */
    def BKE_camera_sensor_fit(p_sensor_fit, sizex, sizey):
        if (p_sensor_fit == 'AUTO'):
            if (sizex >= sizey):
                return 'HORIZONTAL'
            else:
                return 'VERTICAL'

        return p_sensor_fit

    sensor_fit = BKE_camera_sensor_fit(camd.sensor_fit, xasp * winx, yasp * winy)

    if (sensor_fit == 'HORIZONTAL'):
      viewfac = winx
    else:
      viewfac = ycor * winy

    pixsize /= viewfac

    #/* extra zoom factor */
    pixsize *= 1 #params->zoom

    #/* compute view plane:
    # * fully centered, zbuffer fills in jittered between -.5 and +.5 */
    xmin = -0.5 * winx
    ymin = -0.5 * ycor * winy
    xmax =  0.5 * winx
    ymax =  0.5 * ycor * winy

    #/* lens shift and offset */
    dx = camd.shift_x * viewfac # + winx * params->offsetx
    dy = camd.shift_y * viewfac # + winy * params->offsety

    xmin += dx
    ymin += dy
    xmax += dx
    ymax += dy

    #/* fields offset */
    #if (params->field_second):
    #    if (params->field_odd):
    #        ymin -= 0.5 * ycor
    #        ymax -= 0.5 * ycor
    #    else:
    #        ymin += 0.5 * ycor
    #        ymax += 0.5 * ycor

    #/* the window matrix is used for clipping, and not changed during OSA steps */
    #/* using an offset of +0.5 here would give clip errors on edges */
    xmin *= pixsize
    xmax *= pixsize
    ymin *= pixsize
    ymax *= pixsize

    return xmin, xmax, ymin, ymax


def projection_matrix(camd):
    r = bpy.context.scene.render
    left, right, bottom, top = view_plane(camd, r.resolution_x, r.resolution_y, 1, 1)

    farClip, nearClip = camd.clip_end, camd.clip_start

    Xdelta = right - left
    Ydelta = top - bottom
    Zdelta = farClip - nearClip

    mat = [[0]*4 for i in range(4)]

    mat[0][0] = nearClip * 2 / Xdelta
    mat[1][1] = nearClip * 2 / Ydelta
    mat[2][0] = (right + left) / Xdelta #/* note: negate Z  */
    mat[2][1] = (top + bottom) / Ydelta
    mat[2][2] = -(farClip + nearClip) / Zdelta
    mat[2][3] = -1
    mat[3][2] = (-2 * nearClip * farClip) / Zdelta

    return mat
#    return sum([c for c in mat], [])

def next_frame():
#    bpy.data.scenes[s].frame_set(bpy.data.scenes[s].frame_current+1)
     bpy.context.scene.frame_set(bpy.context.scene.frame_current + 1)

def set_frame(i):
#     bpy.data.scenes[s].frame_set(i)
     bpy.context.scene.frame_set(i)

def get_cam():
    return get_3x4_P_matrix_from_blender(bpy.context.object)[0]
#    return get_3x4_P_matrix_from_blender(bpy.data.objects['Camera.004'])[0]

def mkdepth():
    context = bpy.context
    scene = bpy.context.scene
    render = bpy.context.scene.render
    render.image_settings.color_mode = 'RGBA' # ('RGB', 'RGBA', ...)
    render.image_settings.color_depth = '16' # ('8', '16')
    render.image_settings.file_format = 'PNG' # ('PNG', 'OPEN_EXR', 'JPEG, ...)
    # render.film_transparent = True
    scene.use_nodes = True
    depth_scale = 1    # adjust this for PNG
    scene.view_layers["RenderLayer.001"].use_pass_normal = True
    scene.view_layers["RenderLayer.001"].use_pass_diffuse_color = True
    scene.view_layers["RenderLayer.001"].use_pass_object_index = True
    scene.view_layers["RenderLayer.001"].use_pass_z = True
    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links
    # Clear default nodes
    for n in nodes:
        nodes.remove(n)

    # Create input render layer node
    render_layers = nodes.new('CompositorNodeRLayers')

    # Create depth output nodes
    depth_file_output = nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    depth_file_output.base_path = '/tmp/out2'
    depth_file_output.file_slots[0].use_node_format = True
    depth_file_output.format.file_format = render.image_settings.file_format
    depth_file_output.format.color_depth = render.image_settings.color_depth 
    if depth_file_output.format.file_format == 'OPEN_EXR':
        links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
    else:
        depth_file_output.format.color_mode = "BW"

        # Remap as other types can not represent the full range of depth.
        map = nodes.new(type="CompositorNodeMapValue")
        # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
        map.offset = [-0.7]
        map.size = [depth_scale]
        map.use_min = True
        map.min = [0]

        # links.new(render_layers.outputs['Depth'], map.inputs[0])
        links.new(map.outputs[0], depth_file_output.inputs[0])
    # Create normal output nodes
    scale_node = nodes.new(type="CompositorNodeMixRGB")
    scale_node.blend_type = 'MULTIPLY'
    # scale_node.use_alpha = True
    scale_node.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
    links.new(render_layers.outputs['Normal'], scale_node.inputs[1])
    
    bias_node = nodes.new(type="CompositorNodeMixRGB")
    bias_node.blend_type = 'ADD'
    # bias_node.use_alpha = True
    bias_node.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
    links.new(scale_node.outputs[0], bias_node.inputs[1])

    normal_file_output = nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = 'Normal Output'
    normal_file_output.base_path = '/tmp/out2'
    normal_file_output.file_slots[0].use_node_format = True
    normal_file_output.format.file_format = render.image_settings.file_format
    links.new(bias_node.outputs[0], normal_file_output.inputs[0])

    # Create albedo output nodes
    alpha_albedo = nodes.new(type="CompositorNodeSetAlpha")
    links.new(render_layers.outputs['DiffCol'], alpha_albedo.inputs['Image'])
    links.new(render_layers.outputs['Alpha'], alpha_albedo.inputs['Alpha'])
    albedo_file_output = nodes.new(type="CompositorNodeOutputFile")
    albedo_file_output.label = 'Albedo Output'
    albedo_file_output.base_path = '/tmp/out2'
    albedo_file_output.file_slots[0].use_node_format = True
    albedo_file_output.format.file_format = render.image_settings.file_format
    albedo_file_output.format.color_mode = 'RGBA'
    albedo_file_output.format.color_depth = render.image_settings.color_depth 
    links.new(alpha_albedo.outputs['Image'], albedo_file_output.inputs[0])

    # Create id map output nodes
    id_file_output = nodes.new(type="CompositorNodeOutputFile")
    id_file_output.label = 'ID Output'
    id_file_output.base_path = '/tmp/out2'
    id_file_output.file_slots[0].use_node_format = True
    id_file_output.format.file_format = render.image_settings.file_format
    id_file_output.format.color_depth = render.image_settings.color_depth 

    if render.image_settings.file_format == 'OPEN_EXR':
        links.new(render_layers.outputs['IndexOB'], id_file_output.inputs[0])
    else:
        id_file_output.format.color_mode = 'BW'

        divide_node = nodes.new(type='CompositorNodeMath')
        divide_node.operation = 'DIVIDE'
        divide_node.use_clamp = False
        divide_node.inputs[1].default_value = 2**int(render.image_settings.color_depth)

        links.new(render_layers.outputs['IndexOB'], divide_node.inputs[0])
        links.new(divide_node.outputs[0], id_file_output.inputs[0])
    return depth_file_output, normal_file_output, albedo_file_output, id_file_output

    
# ----------------------------------------------------------------------------
def test():

# For simple tests:
    cam = bpy.data.objects['Camera.001']

# For the sunset set
#     cam = bpy.data.objects['Camera.004']

# For the cube set
#     cam = bpy.data.objects['Camera']
#    cam = bpy.data.objects['Camera.010']
    P, K, RT = get_3x4_P_matrix_from_blender(cam)
    print("K")
    print(K)
    print("RT")
    print(RT)
    print("P")
    print(P)

    print("==== Tests ====")
    e1 = Vector((1, 0,    0, 1))
    e2 = Vector((0, 1,    0, 1))
    e3 = Vector((0, 0,    1, 1))
    O  = Vector((0, 0,    0, 1))

    p1 = P * e1
    p1 /= p1[2]
    print("Projected e1")
    print(p1)
    print("proj by object_utils")
    print(project_by_object_utils(cam, Vector(e1[0:3])))

    p2 = P * e2
    p2 /= p2[2]
    print("Projected e2")
    print(p2)
    print("proj by object_utils")
    print(project_by_object_utils(cam, Vector(e2[0:3])))

    p3 = P * e3
    p3 /= p3[2]
    print("Projected e3")
    print(p3)
    print("proj by object_utils")
    print(project_by_object_utils(cam, Vector(e3[0:3])))

    pO = P * O
    pO /= pO[2]
    print("Projected world origin")
    print(pO)
    print("proj by object_utils")
    print(project_by_object_utils(cam, Vector(O[0:3])))


    # save the 3x4 P matrix into a plain text file
    nP = numpy.matrix(P)
    numpy.savetxt("/tmp/out2/P3x4.txt", nP)  # to select precision, use e.g. fmt='%.2f'
    
# ----------------------------------------------------------------------------
#def test2():
    # get vertices of selected object
    # get object ransforms
    # project
    # plot

def test2():
#     P = Matrix([
#     [2789.977470, 34.945628, -928.653184, 93.386696  ],
#     [0.831300, -2824.052194, -636.441117, 578.409788 ],
#     [-0.044857, 0.006928,    -0.998969,   0.542557   ]
#     ])
#     P = numpy.loadtxt("/Users/rfabbri/3d-curve-drawing/ground-truth/models/pabellon_barcelona_v1/3d/ground-truth-pavillion/ground-truth-pavillion-cameras/078.projmatrix")
#     path = "/Users/rfabbri/3d-curve-drawing/ground-truth/models/pabellon_barcelona_v1/3d/ground-truth-pavillion/ground-truth-pavillion-cameras/"
#     name = path + "078"
#     path = "/Users/rfabbri/3d-curve-drawing/ground-truth/robot/feature/vase/vase-mcs-work/"
    P = Matrix([
    [-10, -100, 0, 200],
    [0, -100, -10, 200],
    [0, -1, 0, 2]
    ])
    suf = 'helixcam'
    get_blender_camera_from_3x4_P(P, 1, "-" + suf)
    r, q = rf_rq(numpy.matrix(P))
    print(r)
    print(q)
    k, r, t = KRT_from_P(numpy.matrix(P))
    print('k',k)
    print(r)
    print(t)


##  #     suf = "Img026_06"
##  #     sufs = ["Img066_14", "Img091_19", "Img106_03"]
##  #    sufs = ["Img041_09"]# , "Img066_14", "Img091_19", "Img106_03"]
##      sufs = ["Img001_01"]# , "Img066_14", "Img091_19", "Img106_03"]
##      for suf in sufs :
##          name = path + suf
##          P = numpy.loadtxt(name+".projmatrix")
##  #     P = Matrix([
##  #     [2. ,  0. , - 10. ,   282.  ],
##  #     [0. ,- 3. , - 14. ,   417.  ],
##  #     [0. ,  0. , - 1.  , - 18.   ]
##  #     ])
##          r, q = rf_rq(numpy.matrix(P))
##          print(r)
##          print(q)
##          # This test P was constructed as k*[r | t] where
##          #     k = [2 0 10; 0 3 14; 0 0 1]
##          #     r = [1 0 0; 0 -1 0; 0 0 -1]
##          #     t = [231 223 -18]
##          k, r, t = KRT_from_P(numpy.matrix(P))
##          print('k',k)
##          print(r)
##          print(t)
##          get_blender_camera_from_3x4_P(P, 1, "-" + suf)


if __name__ == "__main__":
    set_frame(1)
#    test()
#   test2()

    pm = get_cam()
    depth_file_output, normal_file_output, albedo_file_output, id_file_output = mkdepth()

#   for i in range(1,101):
    for i in range(1,2):
       # Extrinsic transform matrix
#        pm = get_cam()
#        nP = numpy.matrix(pm)
        fname = "/tmp/out2/%03d.projmatrix" %  i
#        print("writing " + fname)
#        numpy.savetxt(fname, nP)  # to select precision, use e.g. fmt='%.2f'


        render_file_path = "/tmp/out2/pavillion_chair" + '_r_{0:03d}'.format(int(i))

        bpy.context.scene.render.filepath = render_file_path
        depth_file_output.file_slots[0].path = render_file_path + "_depth"
        normal_file_output.file_slots[0].path = render_file_path + "_normal"
        albedo_file_output.file_slots[0].path = render_file_path + "_albedo"
        id_file_output.file_slots[0].path = render_file_path + "_id"

        bpy.ops.render.render(write_still=True)  # render still

       # Advance animation frame
       # test()
#       next_frame()

# Barcelona Pavilion Multiview Dataset

This is a realistic professional-grade high-quality archviz dataset with full 3D ground
truth, including curve and hard edge ground truth, control over illumination and
ground truth camera models for a video sequence.

The ground truth curves were manually marked, and are represented as dense
samples traced on the original meshes, together with local connectivity
information (ie, which points are linked to the other). Note that global
orientation for the curves is not available. Think of the 3D ground-truth curves
as an unordered graph. After all, they are subsets of a mesh which also doesn't
have global orientation information.


## Files

```
images/{midday,night,sunset}   100 640x360 video frames for each illumination condition

cameras/   3x4 cameras for each video frame, in text format

3d/   blender.org files with the full 3D ground truth

3d/original   the original professional files with editable 3D objects and
  render settings. Open 3d/original/3d/pavillon_barcelone_v1.2.blend in Blender.

3d/full/pavillon_barcelone_v1.2-ungrouped02-separate_datablocks-no_modifiers-fixed-010.blend
  .blend file with the 3D curves browseable in Blender

3d/full/pavillon_barcelone_v1.2-ungrouped02-separate_datablocks-no_modifiers-fixed-010.dae
  .dae (collada) XML with the curve objects (points/edges) corresponding to the
  above .blend. You can open the .blend file in Blender, right-click a curve to
  get its name (shown in the top right, eg. Cube.030), then search this XML file
  for the object.
  
3d/full/pavillon_barcelone_v1.2-ungrouped02-separate_datablocks-no_modifiers-fixed-010-overlay.blend
  the ground truth curves overlayed on top of the original 3D models, for checking
  
3d/full/pavillon_barcelone_v1.2-full_mesh_for_PLY_export-ungrouped-all_modifiers-joined-001.ply.bz2
  scene in one full (huge) dense mesh PLY file to be opened in Meshlab, etc. No
  semantic info, as all objects were joined in Blender into one prior to export.
  Useful for computing ground truth error. The mesh is huge, make sure to have a
  very powerful computer.

3d/full/pavillon_barcelone_v1.2-cam-baked-007.blend
  The file used to render the chair scenes. Open the file in blender and click
  on the timeline to see the video preview on the top-right window. To see the
  camera path, change the scene to sunset.

3d/cropped/pavillon_barcelone_v1.2-ungrouped02-separate_datablocks-no_modifiers-fixed-bounding_box-no_edges-010-002.blend
  cropped version of the scene curves in near a bounding box, to reduce compute load

3d/cropped/pavillon_barcelone_v1.2-ungrouped02-separate_datablocks-no_modifiers-fixed-bounding_box-010-002.blend
  open in blender to see a depiction of the bounding box used to generate the cropped vesion above

3d/src/parse_collada.sce   code to parse the .da XML files and produce global 3D points.

3d/cropped/{gt-points.txt,gt-edges.txt}  all points and local connectivity
  information for the marked curves, in txt format for the cropped scene. For
  the full scene, you have to generate these .txt files yourself from the .dae,
  see below for more information.

  gt-points.txt   x y z coordinates for sample points, one point per line.
                  lets call this the 'allpts' nx3 array

  gt-edges.txt    indices of points that are connected by a vertex.
                  The txt file seems to have floats, but they are ints.
                  Connect a 3D point if allpts( alledg(i-1:i) , :).
                  See the end of parse_collada.sce for an example of how to
                  plot the ground truth in scilab/matlab.

3d/camera-path

```

## Getting 3D points and edge info using the .dae Collada file

To get 3D points and connectivitiy information, use the Scilab.org script
src/parse_collada.sce (matlab is similar, but just download the free Scilab and
run this script once).

First edit parse_collada.sce and put the name of the .dae file in the "d"
variable. Then open scilab and type "exec path/to/parse_collada.sce", adjusting the
path acoordingly.

The script will output two ASCII files with the curve information:
  /tmp/gt-points.txt  x y z coordinates for sample points, one point per line.
                      lets call this the 'allpts' nx3 array
  /tmp/gt-edges.txt   indices of points that are connected by a vertex.
                      The txt file seems to have floats, but they are ints.
                      Connect a 3D point if allpts( alledg(i-1:i) , :).
                      See the end of parse_collada.sce for an example of how to
                      plot the ground truth in scilab/matlab.

We provide these files for the cropped version of the dataset only, but you can
easily generate them for the full dataset.

When using the XML .dae COLLADA file in a custom way, you will have to transform
from object coordinates to world coordinates prior to using the cameras. 
The code below illustrates how to do this transform in Matlab/Scilab:

```matlab
%------------------------
% Auxiliary function to convert from euler angles to 3x3 matrix
function mat = eul_to_mat3(eul)
	ci = cos(eul(1));
	cj = cos(eul(2));
	ch = cos(eul(3));
	si = sin(eul(1));
	sj = sin(eul(2));
	sh = sin(eul(3));
	cc = ci * ch;
	cs = ci * sh;
	sc = si * ch;
	ss = si * sh;

	mat(1)(1) = (cj * ch);
	mat(2)(1) = (sj * sc - cs);
	mat(3)(1) = (sj * cc + ss);
	mat(1)(2) = (cj * sh);
	mat(2)(2) = (sj * ss + cc);
	mat(3)(2) = (sj * cs - sc);
	mat(1)(3) = -sj;
	mat(2)(3) = (cj * si);
	mat(3)(3) = (cj * ci);

%------------------------
% For example, the .dae has the object Curve_000,
% with the following transforms:
%       <translate sid="location">1.166548 -1.320713 2.108471</translate>
%       <rotate sid="rotationZ">0 0 1 -46.07418</rotate>
%        <rotate sid="rotationY">0 1 0 -67.88113</rotate>
%        <rotate sid="rotationX">1 0 0 -0.7029412</rotate>
%        <scale sid="scale">0.136427 0.1467155 0.1364271</scale>
% These are mapped in matlab/scilab to:

loc = [1.166548 -1.320713 2.108471]'
Rz=[0 0 1 -46.07418]
Ry=[0 1 0 -67.88113]
Rx=[1 0 0 -0.7029412]
scale=diag([0.136427 0.1467155 0.1364271])

% Generate the object rotation matrix
R = eul_to_mat3([Rx(4), Ry(4), Rz(4)]*%pi/180)'

% Assuming the N object points as 3xN vector 'po',
% Tranform to world coordinates like this:
p = R*scale*po + loc*ones(1,size(po,2)) 

% Then you can project p using the 3x4 P matrix from *.projmatrix as usual.
```

## Generating the PLY of the scene as one huge mesh

* open the original .blend
* select the objects you don't want to clutter the scene and delete them: trees,
  cameras and lights (right mouse button to select -> x -> delete)
* select all objects
* unlink all groups to make the meshes direct: Ctrl-Shift-A 
* select all objects
* apply all modifyers: Alt-C
* join all meshes into one: Ctrl-J (for some reason pressing f first
helped.. unsure)
* export PLY

## Generating your own video sequence with camera ground truth

You can generate another video sequence or re-render at higher resolution to
generate your own ground truth. Ask [Ric Fabbri](http://rfabbri.github.io).
Beware! Rendering these videos can take several days.


## Version

Dataset produced and tested with Blender 2.76 on Mac OSX and Linux.

## Authors

[Ricardo Fabbri](http://rfabbri.github.io) built the dataset.
Further authors include Anil Usumezbas and Benjamin Kimia.

# Credits

This dataset was built on top of E-Mirage studios's dataset downloaded from
https://www.blender.org/download/demo-files, for which the unmodified version
is distributed under the 3d/original folder.

## Links

Images and explanations of this ground truth are provided in:
Multiview-3d-Drawing.sf.net

More Blender demo files that can serve as ground truth
https://www.blender.org/download/demo-files/

E-Mirage Studios
Ric Fabbri's posts on how to convert to/from Blender cameras to standard 3x4
format for your own projects with blender

http://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
http://blender.stackexchange.com/questions/40650/blender-camera-from-3x4-matrix

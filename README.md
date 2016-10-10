# Barcelona Pavilion Multiview Dataset

This is a realistic professional-grade synthetic dataset with full 3D ground
truth, including curve and hard edge ground truth, control over illumination and
ground truth camera models for a video sequence.


## Files

```
images/{midday,night,sunset} 100 640x360 video frames for each illumination condition

cameras/ 3x4 cameras for each video frame, in text format

3d/ blender.org files with the full 3D ground truth

3d/pavillon_barcelone_v1.2-full_mesh_for_PLY_export-ungrouped-all_modifiers-joined-001.ply.bz2
  This is a full joined mesh PLY file to be opened in Meshlab, etc. No semantic info,
  as all objects were joined into one prior to export. Useful for computing
  ground truth error, etc. The mesh is huge, make sure to have a very powerful
  computer.

3d/pavillon_barcelone_v1.2-ungrouped02-separate_datablocks-no_modifiers-fixed-010.dae
  .dae (collada) XML with the objects
  
pavillon_barcelone_v1.2-ungrouped02-separate_datablocks-no_modifiers-fixed-010.blend
  Corresponding .blend file with the 3D curves browseable in Blender
  

```

## Using the .dae Collada file

When using the XML .dae COLLADA file, you will have to transform from object
coordinates to world coordinates prior to using the cameras. The code below
provides instructions on how to do this transform in Matlab/Scilab:

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


## Generating your own video sequence with camera ground truth

You can generate another video sequence or re-render at higher resolution to
generate your own ground truth. Ask [Ric Fabbri](http://rfabbri.github.io).
Beware! Rendering these videos can take several days.


## Version

Dataset produced and tested with Blender 2.76 on Mac OSX and Linux.

## Authors

[Ricardo Fabbri](http://rfabbri.github.io) built the dataset.
Further authors include Anil Usumezbas and Benjamin Kimia.

## Links

Images and explanations of this ground truth are provided in:

Multiview-3d-Drawing.sf.net

# Barcelona Pavilion Multiview Dataset

This is a realistic professional-grade synthetic dataset with full 3D ground
truth, including curve and hard edge ground truth, control over illumination and
ground truth camera models for a video sequence.


## Files

```
images/{midday,night,sunset} 100 640x360 video frames for each illumination condition

cameras/ 3x4 cameras for each video frame, in text format

3d/ blender.org files with the full 3D ground truth
```


## Generating your own video sequence with camera ground truth

You can generate another video sequence or re-render at higher resolution to
generate your own ground truth. Ask [Ric Fabbri](http://rfabbri.github.io).
Beware! Rendering these videos can take several days.


## Authors

[Ricardo Fabbri](http://rfabbri.github.io) built the dataset.
Further authors include Anil Usumezbas and Benjamin Kimia.

## Links

Images and explanations of this ground truth are provided in:

Multiview-3d-Drawing.sf.net

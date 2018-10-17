## local Descriptors Optimized for Average Precision (DOAP)

### Wie kann man DOAP ausführen?

### Aufruf mit *use_doap_with_csv.m*
```matlab
use_doap_with_csv(
  matconvnet_root_dir,
  path_to_doap_model,
  path_to_layers_dir,
  path_to_input_directory,
  path_to_output_directory
)
```
### Beispielaufruf
```
use_doap_with_csv(
  '/home/mizzade/Workspace/diplom/detectors_and_descriptors/desc_doap/MatConvNet/matconvnet-1.0-beta25',
  'HPatches_ST_LM_128d.mat',
  'models',
  'test_in',
  'test_out')
```

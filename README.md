# OCT image SRF detection
Final group project of the signal and image processing course.<br>
https://github.com/signal-and-image-processing-fs19/oct-image-srf-detection.git

## Task ##
The goal is to find a image processing solution to detect from OCT images, if it shows SRF (subretinal fluid) which is
a diagnostic factor in detecting degenerative retinal diseases.
sub-retinal fluid (SRF) corresponds to the accumulation of a clear or lipid-rich exudate in the
sub-retinal space, i.e., between the photoreceptor layer and the underlying retinal pigment
epithelium (RPE).

What to look for: homogeneous hypo reflective well-defined areas between the retinal
pigment epithelium layer (RPE) and photoreceptor layer.

Textbook example of sub-retinal fluid:
![srfexample](srf.png?raw=true "srfexample")

## Run ##
Created on Python version: 3.7.3


Clone the project and run the following command from the root folder to install all dependencies.

```python
pip install -r requirements.txt
```

Execute 'project_Waelchli_Moser_Meise.py' (wrapper) or 'oct_srf_detection.py':
```cmd
python project_Waelchli_Moser_Meise.py
```

Creates (further description see 'Output' below):
- 'project_Waelchli_Moser_Meise.csv': main output with image classification results
as specified in 'Test-Data/submission_guidelines.txt'
- 'log.dat': log txt-file of the stdout from running the program
- 'figures/{figname}.png': threshold-optimizing plot

## Install new Packages ##
Make sure to install new packages using the following commands in order to make sure that the
dependencies are listed in the requirements.txt file:

```python
pip install <package> 
pip freeze > requirements.txt
```

## Data ##
<b>Training Data</b> consists of srf examples ('Train-Data/SRF/') and non-srf examples ('Train-Data/noSRF/') 15 images each.

<b>Test Data</b> consists of 100 oct images ('Test-Data/handout/) of unknown srf status.

Images are in the png format (rgba) but can be transformed to gray scale after read in, since the data
is in gray-scale (0-255).

If a template is specified manually it should also be in the same format.

## Output ##
- 'project_Waelchli_Moser_Meise.csv': main output with image classification results
as specified in 'Test-Data/submission_guidelines.txt'<br>
example_submission.csv:
```csv
filename,label
990.png,0
9752.png,1
9703.png,0
8382.png,1
...
```
- 'log.dat': log txt-file of the stdout from running the program
- 'figures/{figname}.png': threshold-optimizing plot

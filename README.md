# Automatic Image Segmentation by Dynamic Region Merging
---

__Team Name__: Noisy Pixel<br><br>

__Team Members__:<br>
Shubham Dokania _(2020701016)_<br>
Shanthika Shankar Naik _(2020701013)_ <br>
Sai Amrit Patnaik _(2020701026)_ <br>

__Assigned TA__: Adhithya Arun <br><br>


This project is undertaken as a part of the Digital Image Processing (CSE478) coursework at IIIT Hyderabad in Monsoon semester 2020. The paper implemented in this project is: [Automatic Image Segmentation by Dynamic Region Merging](https://dl.acm.org/doi/10.1109/TIP.2011.2157512) by _Peng et. al._


The approach focuses on a region merging technique for selectively merging small regions in an initially over-segmented image and to reach a final segmented output. The details of the implementation have been outlined in the project report and the proposal document which can be found [here](./documents/proposal.md).


The following sections outline how to run the demo and some examples of the expected output from running the mentioned scripts.

## Code Structure
---

The code is structured as follows:

```
.
├── README.md
├── demo.ipynb
├── demo.py
├── documents
│   ├── Mid-eval.pdf
│   ├── Mid-eval.pptx
│   └── proposal.md
├── guidelines.md
├── img
├── proposal.md
└── src
    ├── __init__.py
    ├── nng.py
    ├── rag.py
    ├── region_merging.py
    ├── segment
    │   ├── __init__.py
    │   └── watershed.py
    ├── sprt.py
    ├── utils.py
    └── visualize_sprt.py
```

In the above structure, the source code for the whole implementation can be found in the `src` directory. The scripts each contain a description of the functions/classes implemented and provide a wrapper to experiment with the flow of the program.

## Run Demo
---

### Pre-requisites

Before running the demo, make sure that the following python libraries are installed and working well. The code available in this repository makes use of the following to function completely.

```
numpy
opencv-python
scikit-image
matplotlib
seaborn
tqdm
scipy
argparse
```

### Data

Although the code in this repository does not require any specific dataset, we conducted some experiments on the [Berkeley Segmentation Dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/). More specificaally, we ran our method over some of the images from the test set. The dataset can be downloaded by following the instructions from the website or it can be downloaded from our [google drive link](#).

The images can be put in any location, the only requirement is that the path to the image for testing should be correct, as would be discussed below.

### Running the demo

The `demo.py` script in the repository is the main script which can be used to run the demo. The program accepts command line parameters such as the example given below:

```
python demo.py --image_path <path to input image> --output <path to output> --watershed --visualize
```

The above command takes an input image and uses watershed algorithm for initial over-segmentation of the image. The `visualize` flag makes sure that the visualizations which are being saved to the output path are also displayed to the used if a desktop display is available.

For more information on the possible parameters, we can also check the help flag as follows:

```
python demo.py --help

usage: demo.py [-h] [--input_path INPUT_PATH] [--output OUTPUT] [--watershed]
               [--lambda1 LAMBDA1] [--lambda2 LAMBDA2] [--alpha ALPHA]
               [--beta BETA] [--visualize] [--max_iters MAX_ITERS]

Segmentation parameters Parser. Pass the parameters following instructions
given below to run the demo experiment.

optional arguments:
  -h, --help            show this help message and exit
  --input_path INPUT_PATH
                        Path to image for processing
  --output OUTPUT       Path to folder for storing output
  --watershed           Whether to use watershed for initial segmentation
  --lambda1 LAMBDA1     Value for Lambda1 parameter
  --lambda2 LAMBDA2     Value for Lambda2 parameter
  --alpha ALPHA         Parameter value for alpha
  --beta BETA           Parameter value for beta
  --visualize           Visualize the outputs of the algorithm
  --max_iters MAX_ITERS
                        Max iterations for the Region merging loop
```

As shown above, the parameters listed in the help menu can be altered via the command line. We now show some examples of the output when we run the demo with variations in the values as stated above. For a more detailed report and analysis, we recommend to look into the presentation [here](./documents/DIP_presentation.pdf).



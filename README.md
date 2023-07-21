# clip2brain
Code from paper "Natural language supervision with a large and diverse dataset builds better models of human high-level visual cortex"

### Step 1: Clone the code from Github and download the data
```
git clone https://github.com/ariaaay/clip2brain.git
cd clip2brain
```
You will also need to install the [natural scene datasets](https://naturalscenesdataset.org/).
For downstream analysis you might also need to download [coco annotations](https://cocodataset.org/#download). It is optional for just running encoding models. The one used for analysis are: 2017 Train/Val annotations (241MB).

### Step 2: Install requirements
[Requirements.txt](https://github.com/ariaaay/clip2brain/blob/main/requirements.txt) contains the necessary package for to run the code in this project.
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirement.txt
```
Install `torch` and `torchvision` from [PyTorch](https://pytorch.org/).

### Step 3: Set up paths in `config.cfg`
Modify `config.cfg` to reflect local paths to NSD data and COCO annotations.

### Step 4: Reproduce results!
Note: Code in project commands by default runs all models for all subjects. If you would like to speed up the process and only test out certain model, please comment out models you don't need.
```
sh project_commands.sh
```

# 3D Lymphoma Segmentation on PET/CT Images

This repository contains the code for the paper:

**"3D Lymphoma Segmentation on PET/CT Images via Multi-Scale Information Fusion with Cross-Attention"**  
## Authors and Affiliations
- **Huan Huang**  
  School of Computer Science and Technology, Zhengzhou University of Light Industry, Zhengzhou, China  
- **Liheng Qiu**  
  Peking University Peoples’ Hospital, Beijing, China  
- **Shenmiao Yang**  
  Department of Nuclear Medicine, Peking University Peoples’ Hospital, Beijing, China  
- **Longxi Li**  
  School of Computer Science and Technology, Zhengzhou University of Light Industry, Zhengzhou, China  
- **Jiaofen Nan**  
  School of Computer Science and Technology, Zhengzhou University of Light Industry, Zhengzhou, China  
- **Yanting Li**  
  School of Computer Science and Technology, Zhengzhou University of Light Industry, Zhengzhou, China  
- **Chuang Han**  
  School of Computer Science and Technology, Zhengzhou University of Light Industry, Zhengzhou, China  
- **Weihua Zhou**  
  Department of Applied Computing, Michigan Technological University, Houghton, MI, USA  
 

### Corresponding Authors

- **Fubao Zhu**  
  School of Computer Science and Technology, Zhengzhou University of Light Industry, Zhengzhou, China  
  - Email: fbzhu@zzuli.edu.cn  
  - Mailing Address: School of Computer Science and Technology, Zhengzhou University of Light Industry, Zhengzhou 450002, Henan, China  

- **Chen Zhao**  
  Department of Computer Science, Kennesaw State University, Marietta, GA, USA  
  - Email: czhao4@kennesaw.edu  
  - Mailing Address: 680 Arntson Dr, Atrium BLDG, Marietta, GA 30060, USA  

## Overview

This project implements a 3D lymphoma segmentation model using PET/CT images. The approach leverages **multi-scale information fusion** and **cross-attention mechanisms** for improved segmentation performance. The code includes training, validation, and testing steps integrated into a single pipeline.

---

## Dataset

### Notes on Data Usage
- The dataset is derived from the **AutoPET** collection available on [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/collection/fdg-pet-ct-lesions/).
- The official preprocessing code for generating NIfTI-format data is provided by [lab-midas/TCIA_processing](https://github.com/lab-midas/TCIA_processing).
- **Usage Restrictions**: Access to this dataset requires authorization from TCIA. For research publications using this dataset, please cite:  
  [Gatidis S, Kuestner T. (2022) A whole-body FDG-PET/CT dataset with manually annotated tumor lesions (FDG-PET-CT-Lesions) [Dataset]. The Cancer Imaging Archive. DOI: 10.7937/gkr0-xv29](https://doi.org/10.7937/gkr0-xv29).

### Data Format
The dataset is organized under the directory `your dataset root/` and includes:
- **PatientID_ScanDate_Modality.npz**: Processed `.npz` files, where each file contains:
  - PET, CT, SUV, and Ground Truth data grouped into slices of 32 images.
  - Center-cropped to 224x224 resolution.
  - Modality examples: `PET.npz`, `CTres.npz`, `SUV.npz`, and `SEG.npz`.

- **Fold Directories**: Five-fold cross-validation splits. Each `fold_*` directory contains:
  - `train.txt`: List of `.npz` file paths for training.
  - `val.txt`: List of `.npz` file paths for validation.
  - `test.txt`: List of `.npz` file paths for testing.

**Example Directory Structure**:
```
NPY_GROUPED/ 
├── Patient001_2023-01-01_PET.npz 
├── Patient001_2023-01-01_CTres.npz 
├── Patient001_2023-01-01_SEG.npz
├── Patient001_2023-01-01_SUV.npz 
├── Patient002_2023-01-02_PET.npz 
├── ...
├── fold_1/ 
│   ├── train.txt 
│   ├── val.txt 
│   ├── test.txt 
├── fold_2/ 
├── fold_3/ 
├── fold_4/ 
├── fold_5/
```

## Dependencies

Install the dependencies using the provided `requirements.txt`:
```
pip install -r requirements.txt
```
### Key Dependencies
- PyTorch
- MONAI
- NumPy
- tqdm
- scikit-learn

## Usage

### Training and Validation
Update the `config.yaml` file with the desired fold ID and dataset path:
```yaml
fold_id: 1
data_root: "your dataset root"
save_root: "your save root"
max_epochs: 150
...
```
Run the training script:
```
python main.py
```

The training process includes validation at regular intervals (`val_interval`), and the best model is saved as `best_epoch.pth` based on the Dice score.

### 2. Testing

After training is complete, testing is performed automatically on the test set of the specified fold.  
Results, including predicted segmentations and evaluation metrics (Dice score, precision, recall), are saved in the `save_root` directory.

### 3. Outputs

During training and evaluation, outputs are saved in the specified `save_root` directory:
- **Checkpoints**: Best model weights (`best_epoch.pth`).
- **Logs**: Training, validation, and testing logs.
- **Segmentations**: Predicted segmentation results in `.npy` format.


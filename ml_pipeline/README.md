# Fingerprint Matching Machine Learning Pipeline

## Overview

A subdirectory of the CS 228 project that provides code for building a custom Siamese CNN model. The `ml_pipeline` folder consists of Python code to download datasets, train and test a CNN model with a specified dataset, and enhance fingerprints to a new dataset. Java files in the `fingerprintmodel` folder performs pairwise tests using [SourceAFIS](https://sourceafis.machinezoo.com/) and the custom CNN model.

## Requirements

- Java 25
- Python 3.15
- Maven

## Recommendations

- IDE: IntelliJ IDEA
- Dedicated GPU
    - NVIDIA: GeForce 10 series or later
    - AMD: RX 7000 series or later
    - Mac: M1 or later
    - Intel: Any Intel Arc A or B series GPU or better

## Machine learning pipeline layout

```text
ml_pipeline/
  device.py
  download_datasets.py    
  enhance_fingerprints.py 
  export_encoder.py       
  fingerprint_dataset.py  
  install_requirements.py 
  siamese_model.py        
  test_model.py           
  train_model.py
  requirements.txt
src/main/java/
  fingerprintmodel/
    BaselineFingerprintPairwise.java 
    CustomFingerprintPairwise.java
    FingerprintEncoderService.java
    FingerprintImageLoader.java
src/test/java/
  fingerprintmodel/
    BaselineFingerprintPairwiseTest.java
    CustomFingerprintPairwiseTest.java
```

## Setup

Ensure you are in the project root directory `CS228-Project` before following the setup.

### 1. Install required dependencies

```
mvn compile
```

> Compiling will also install a Python virtual environment and install dependencies within the environment. See [`pom.xml`](../pom.xml) for more details.

### 2. Activate the virtual environment

For Windows:

```
.\.venv\Scripts\Activate.ps1
```

For MacOS or Linux:

```
source .venv/bin/activate
```

### 3. Download datasets

Configure `URLS` and `FILENAME_PATTERN` to specify datasets to download and automatically identify classes and samples with regex expressions. Assignments to `class_id` and `impression_id` may need to be modified.

```
python .\ml_pipeline\download_datasets.py
```

## Training and testing the custom model

Ensure you are in the project root directory `CS228-Project` before following the following steps.

### 1. Train the model

```
python .\ml_pipeline\train_model.py
```

### 2. Test the model

```
python .\ml_pipeline\test_model.py
```

### 3. Export the model

The model will be exported as a `.pt` file for Java tests.

```
python .\ml_pipeline\export_encoder.py
```

### 4. Perform baseline and experimental tests

You may have to manually adjust the threshold value at around line 85 in [`CustomFingerprintPairwise.java`](../src/main/java/fingerprintmodel/CustomFingerprintPairwise.java).

```
mvn test
```

## Enhanced fingerprints (optional)

The following steps will guide enhancing fingerprints using [FpEnhancer](https://github.com/XiongjunGuan/FpEnhancer/tree/main) and rerun training and testing under the enhanced dataset.

### 1. Download the repository

Put the downloaded folder into this project's directory. Rename the `FpEnhancer` folder to `FpEnhancer_main`. The result should look like the following:

```text
ml_pipeline/
  ...
FpEnhancer_main/
  ...
src/
  ...
```

### 2. Download PyTorch checkpoint file

For UNet, download the `best.pth` file [here](https://github.com/XiongjunGuan/FpEnhancer/blob/main/ckpts/Enhancer/download.md). Move the file to the directory `ml_pipeline/data/checkpoints/`.

### 3. Generate enhanced fingerprints

From the project root directory, run:

```
python -m ml_pipeline.enhanced_fingerprints
```

> A new `enhanced` folder will be created in `ml_pipeline/data/`.

### 4. Repeat training, testing, and comparing models

Use flags `-enhanced` or `-e` for Python and `-Denhanced` for Java.

```
python --enhanced .\ml_pipeline\train_model.py
python --enhanced .\ml_pipeline\test_model.py
python --enhanced .\ml_pipeline\export_encoder.py
mvn --Denhanced test
```

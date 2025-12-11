# 📦 Setup Instructions for Environment

## 🔧 Using `environment.yml` (Conda)

### 1. Install Anaconda or Miniconda

-   Download and install from the official website.

### 2. Create Environment from the YML File

``` bash
conda env create -f environment.yml
```

### 3. Activate the Environment

``` bash
conda activate myenv
```

### 4. Run the Project

You're ready to go!

------------------------------------------------------------------------

## 🔧 Using `requirements.txt` (Pip)

### 1. Install Python (3.10+)

### 2. Create Virtual Environment

``` bash
python -m venv venv
```

### 3. Activate Virtual Environment

**Windows:**

``` bash
venv\Scripts\activate
```

**Mac/Linux:**

``` bash
source venv/bin/activate
```

### 4. Install Requirements

``` bash
pip install -r requirements.txt
```

### 5. Run the Project

All set!

------------------------------------------------------------------------

## 🔧 Using `python auto_mode.py "Model_Full_Name"` (Conda/CMD)
## Then, Open the simulator
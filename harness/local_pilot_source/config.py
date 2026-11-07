import os, json, time, math, random, platform, re, traceback, gc, tempfile, subprocess, sys, hashlib
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
import matplotlib; import matplotlib.pyplot as plt
import shutil, importlib.metadata
ON_KAGGLE = os.path.isdir('/kaggle/working')
OUT_DIR = os.environ.get('WSP_LOCAL_OUT', '/kaggle/working/local_strong_pilot' if ON_KAGGLE else '<STUDY_ROOT>/local_pilot/smoke_outputs')
os.makedirs(OUT_DIR, exist_ok=True)
GEN_DIR = OUT_DIR+'/generated_code'; os.makedirs(GEN_DIR, exist_ok=True)
DEVICE='cpu' # evaluation uses FP32 CPU; no contention with the synthesizer's GPU allocation
RUN_TASKS={'modmul':True,'permadd':True}
RUN_ORACLE_CHECK=True
SUCCESS_THRESH=.99
SANDBOX_TIMEOUT={'algo':120}
SPLIT=(.3,.1)
T_START=time.time()
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
MODELS=[
 {'id':'microsoft/phi-4','revision':'2db69c1c3e91a05d2c64a3185acfbaf36f744e25','disk_gib':35},
 {'id':'mistralai/Mistral-Small-24B-Instruct-2501','revision':'9527884be6e5616bdd54de542f9ae13384489724','disk_gib':54},
]
MAX_NEW_TOKENS=8192
MAX_GENERATION_S=900
MAX_TOTAL_S=6*3600
SAMPLING={'do_sample':True,'temperature':.6,'top_p':.95}
# Identical sampling is a deliberate pilot control, not Mistral's recommended .15.
PILOT_CASES=[(task,'none',rep) for rep in range(3) for task in ['modmul','permadd']]
# First one response per task is the operational smoke stage; remaining reps are still only a pilot.
print('LOCAL ONLY; no provider APIs. Planned responses:',len(MODELS)*len(PILOT_CASES),flush=True)

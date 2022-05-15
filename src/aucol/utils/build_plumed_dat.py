
"""
Example plumed file

cv1p: ENGINECV NAME=cv1 ATOMS=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37

b: ANGLE ATOMS=30,12,31
c: ANGLE ATOMS=29,11,30

MOVINGRESTRAINT ...
   ARG=cv1p
   STEP0=0 AT0=0.34 KAPPA0=30000.0
   STEP1=8000 AT1=-0.6
   STEP2=16000 AT2=0.6
... MOVINGRESTRAINT


PRINT STRIDE=50 FILE=colvar.out ARG=*
"""


def build_moving_restrain(structure, init_val, watch_for="", kappa=3000, steps=50000, end_points = 0.6):
    #define cv
    cv_dev = 'cv1p: ENGINECV NAME=cv1 ATOMS='
    atoms = ("".join([str(i+1)+',' for i in range(len(structure))]))[:-1]

    base = [cv_dev + atoms] + watch_for

    end1 = -0.6 if init_val > 0 else 0.6
    method = ["MOVINGRESTRAINT ARG=cv1p STEP0=0 AT0={init_val:.2f} KAPPA0={kappa:.2f} \
STEP1={step1} AT1={end1} STEP2={step2} AT2={end2}".format(init_val=init_val, kappa=kappa, step1=steps, end1=end1, step2=2*steps, end2=-end1)]

    #method = ["MOVINGRESTRAINT ...",
    #"   ARG=cv1p",
    #"   STEP0=0 AT0={init_val:.2f} KAPPA={kappa:.2f}".format(init_val=init_val, kappa=kappa),
    #"   STEP1={step1} AT1={end1}".format(step1=steps, end1=end1),
    #"   STEP2={step2} AT2={end2}".format(step2=2*steps, end2=-end1),
    #"... MOVINGRESTRAINT"]

    print_ops = ["PRINT STRIDE=50 FILE=colvar.out ARG=* "]

    plumed_dat = base + method + print_ops
    print(plumed_dat)

    return plumed_dat

def build_2D_moving_restrain(structure, init_val1, init_val2, end_val1, end_val2, watch_for="", kappa=3000, steps=50000, end_points = 0.6):
    #define cv
    cv_dev1 = 'cv1p: ENGINECV NAME=cv1 ATOMS='
    cv_dev2 = 'cv2p: ENGINECV NAME=cv2 ATOMS='
    atoms = ("".join([str(i+1)+',' for i in range(len(structure)-1)]))[:-1]

    base = [cv_dev1+atoms, cv_dev2+atoms] + watch_for

    method = ["MOVINGRESTRAINT ARG=cv1p,cv2p STEP0=0 AT0={init_val1:.2f},{init_val2:.2f} KAPPA0={kappa:.2f},{kappa:.2f} \
STEP1={step1} AT1={end1:.2f},{end2:.2f} STEP2={step2} AT2={init_val1:.2f},{init_val2:.2f}".format(init_val1=init_val1, init_val2=init_val2, kappa=kappa,
        step1=steps, end1=end_val1, end2=end_val2, step2=2*steps)]

    #method = ["MOVINGRESTRAINT ...",
    #"   ARG=cv1p",
    #"   STEP0=0 AT0={init_val:.2f} KAPPA={kappa:.2f}".format(init_val=init_val, kappa=kappa),
    #"   STEP1={step1} AT1={end1}".format(step1=steps, end1=end1),
    #"   STEP2={step2} AT2={end2}".format(step2=2*steps, end2=-end1),
    #"... MOVINGRESTRAINT"]

    print_ops = ["PRINT STRIDE=50 FILE=colvar.out ARG=* "]

    plumed_dat = base + method + print_ops
    print(plumed_dat)

    return plumed_dat

"""
cv1p: ENGINECV NAME=cv1 ATOMS=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39

b: DISTANCE ATOMS=11,30
c: DISTANCE ATOMS=31,38

METAD ...
LABEL=meta
ARG=cv1p
PACE=200
HEIGHT=2.2
SIGMA=0.06
FILE=HILLS
GRID_MIN=-1.4
GRID_MAX=1.4
GRID_BIN=1000
... METAD


PRINT STRIDE=50 FILE=colvar.out ARG=*
"""



def build_moving_metad(structure, init_val, watch_for="", pace=200, height=2.2, sigma=0.06):
    #define cv
    cv_dev = 'cv1p: ENGINECV NAME=cv1 ATOMS='
    atoms = ("".join([str(i+1)+',' for i in range(len(structure)-1)]))[:-1]

    base = [cv_dev + atoms] + watch_for

    end1 = -0.6 if init_val > 0 else 0.6
    method = ["METAD label=meta ARG=cv1p PACE={pace:.2f} HEIGHT={height:.2f} SIGMA={sigma:.2f} \
FILE=HILLS AT1=GRID_MIN=-1.4 GRID_MAX=1.4 GRID_BIN=1000".format(pace=init_val, height=height, sigma=sigma)]

    #method = ["MOVINGRESTRAINT ...",
    #"   ARG=cv1p",
    #"   STEP0=0 AT0={init_val:.2f} KAPPA={kappa:.2f}".format(init_val=init_val, kappa=kappa),
    #"   STEP1={step1} AT1={end1}".format(step1=steps, end1=end1),
    #"   STEP2={step2} AT2={end2}".format(step2=2*steps, end2=-end1),
    #"... MOVINGRESTRAINT"]

    print_ops = ["PRINT STRIDE=50 FILE=colvar.out ARG=* "]

    plumed_dat = base + method + print_ops
    print(plumed_dat)

    return plumed_dat

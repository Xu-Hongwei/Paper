from mmsdk import mmdatasdk as md

DATASET = md.cmu_mosi
DATA_PATH = r"E:\Xu\data\CMU_MOSI\raw_csd"

recipe = {}
recipe.update(DATASET.highlevel)
recipe.update(DATASET.labels)

data = md.mmdataset(recipe, DATA_PATH)
import pandas as pd

wqp = pd.read_csv("WQP Physical Chemical.csv")
ground_truth = pd.read_csv("Ground Truth.csv")

print(wqp[["ResultMeasureValue", "ResultMeasure/MeasureUnitCode"]])
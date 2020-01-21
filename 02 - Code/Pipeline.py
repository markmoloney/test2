# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd


df = pd.DataFrame()

#create a pipeline that calls all of the functions

#preprocessing = starting with fill_null function
(df.pipe(fill_null(df, attribute_list, stat))
 
#preprocessing = time stamps function
.pipe(extract_timestamps)
 
#processing = init function
.pipe(__init__)
 
#processing = threshold function
.pipe(threshold_col_del)
 
#processing = label encoding function
.pipe(lblencoder)
 
#fill function -999
.pipe(fill_null(self, attribute_list, stat, integer = -999))
 
#standardiser function
.pipe(standardiser) 
 
)
# -



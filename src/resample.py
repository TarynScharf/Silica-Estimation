import random
import pandas as pd

# Your original data (population)

interp_A_autocrysts = [56.322, 66.793, 57.121, 65.521, 56.711, 66.535]
interp_A_xenocrysts = [66.087, 66.703, 66.321, 66.204, 64.198, 63.929, 56.324, 66.338, 66.061, 66.393]
syenogranite = [66.669,66.704,63.531,66.365,66.226,65.848,64.058,66.072,62.695,66.283,66.781,65.605,64.873,66.887,63.401,62.986,65.029,61.114,66.099,65.946,66.118,64.849]
interp_B_autocrysts = [56.322,66.793,52.587,64.027,66.638,66.087,66.703,57.121,66.321,55.574,66.204,64.198,63.929,56.323,65.521,56.711,55.460,65.869,66.338,66.150,66.535,66.061,66.393,67.130]
datasets = [interp_A_autocrysts,interp_A_xenocrysts, syenogranite, interp_B_autocrysts]

interp_A_autocrysts_adjusted = [55.531, 68.458, 56.518, 66.887, 56.011, 68.138]
interp_A_xenocrysts_adjusted = [67.586,68.346,67.874,67.730,65.254,64.921,55.534,67.895,67.554,67.963]
syenogranite_adjusted = [68.304,68.347,64.431,67.929,67.757,67.291,65.080,67.566,63.399,67.828,68.442,66.990,66.087,68.573,64.270,63.757,66.279,61.446,67.600,67.411,67.624,66.057]
interp_B_autocrysts_adjusted = [55.531,68.458,50.921,65.042,68.266,67.586,68.346,56.518,67.874,54.607,67.730,65.254,64.921,55.533,66.887,56.011,54.467,67.316,67.895,67.663,68.138,67.554,67.963,68.873,]
datasets_adjusted = [interp_A_autocrysts_adjusted, interp_A_xenocrysts_adjusted, syenogranite_adjusted, interp_A_autocrysts_adjusted]


num_samples = 6
resampled_datasets = []
for data in datasets_adjusted:
    resampled_data = random.choices(data, k=num_samples)
    resampled_datasets.append(resampled_data)

df = pd.DataFrame({
    'Interp_A_autocrysts': resampled_datasets[0],
    'interp_A_xenocrysts':resampled_datasets[1],
    'syenogranite':resampled_datasets[2],
    'interp_B_autocrysts':resampled_datasets[3]
})
df.to_csv('C:/Users/20023951/Documents/PhD/Reporting/Project 3_FFN/CaseStudy_Xenocrysts/adjusted_downsampled.csv')

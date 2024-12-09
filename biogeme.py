import pandas as pd
from biogeme import biogeme
from biogeme import models
from biogeme.expressions import Beta, Variable
import biogeme.database as db
import numpy as np

# Load the data
data = pd.read_excel("PS6 Mode Choice Survey Data.xlsx")

# Create a BIOGEME database
database = db.Database("ModeChoice", data)

# Define variables for BIOGEME
CHOICE = Variable('CHOICE')
TTRAIL = Variable('TTRAIL')
TCRAIL = Variable('TCRAIL')
OVTRAIL = Variable('OVTRAIL')
CHANGES = Variable('CHANGES')
TTCAR = Variable('TTCAR')
TCCAR = Variable('TCCAR')
OVTCAR = Variable('OVTCAR')
PURPOSE = Variable('PURPOSE')

# Define utility functions
# Define parameters for rail
B_TTRAIL = Beta('B_TTRAIL', 0, None, None, 0)
B_TCRAIL = Beta('B_TCRAIL', 0, None, None, 0)
B_OVTRAIL = Beta('B_OVTRAIL', 0, None, None, 0)
B_CHANGES = Beta('B_CHANGES', 0, None, None, 0)

# Define parameters for car
B_TTCAR = Beta('B_TTCAR', 0, None, None, 0)
B_TCCAR = Beta('B_TCCAR', 0, None, None, 0)
B_OVTCAR = Beta('B_OVTCAR', 0, None, None, 0)

# Alternative-specific constants
ASC_RAIL = Beta('ASC_RAIL', 0, None, None, 1)
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
# Define baseline model (only alternative-specific constants)
# ASC_RAIL = Beta('ASC_RAIL', 0, None, None, 1)
# ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
av = {0: 1, 1: 1}
V_baseline = {0: ASC_CAR, 1: ASC_RAIL}
logprob_baseline = models.loglogit(V_baseline, av, CHOICE)
biogeme_baseline = biogeme.BIOGEME(database, logprob_baseline)
biogeme_baseline.modelName = "BaselineModel"
results_baseline = biogeme_baseline.estimate()

# Define full model (all variables included)
V_full = {
    0: ASC_CAR + B_TTCAR * TTCAR + B_TCCAR * TCCAR + B_OVTCAR * OVTCAR,
    1: ASC_RAIL + B_TTRAIL * TTRAIL + B_TCRAIL * TCRAIL + B_OVTRAIL * OVTRAIL + B_CHANGES * CHANGES,
}
logprob_full = models.loglogit(V_full, av, CHOICE)
biogeme_full = biogeme.BIOGEME(database, logprob_full)
biogeme_full.modelName = "FullModel"
results_full = biogeme_full.estimate()

# Define reduced model 1 (exclude OVTRAIL)
V_reduced1 = {
    0: ASC_CAR + B_TTCAR * TTCAR + B_TCCAR * TCCAR + B_OVTCAR * OVTCAR,
    1: ASC_RAIL + B_TTRAIL * TTRAIL + B_TCRAIL * TCRAIL + B_CHANGES * CHANGES,
}
logprob_reduced1 = models.loglogit(V_reduced1, av, CHOICE)
biogeme_reduced1 = biogeme.BIOGEME(database, logprob_reduced1)
biogeme_reduced1.modelName = "ReducedModel1"
results_reduced1 = biogeme_reduced1.estimate()

# Define reduced model 2 (logarithmic transformation for costs)
database.data['log_TCRAIL'] = np.log(database.data['TCRAIL'].replace(0, np.nan)).fillna(0)
database.data['log_TCCAR'] = np.log(database.data['TCCAR'].replace(0, np.nan)).fillna(0)

# Update the variable references for the reduced model 2
log_TCRAIL = Variable('log_TCRAIL')
log_TCCAR = Variable('log_TCCAR')
V_reduced2 = {
    0: ASC_CAR + B_TTCAR * TTCAR + B_TCCAR * log_TCCAR + B_OVTCAR * OVTCAR,
    1: ASC_RAIL + B_TTRAIL * TTRAIL + B_TCRAIL * log_TCRAIL + B_OVTRAIL * OVTRAIL + B_CHANGES * CHANGES,
}
logprob_reduced2 = models.loglogit(V_reduced2, av, CHOICE)
biogeme_reduced2 = biogeme.BIOGEME(database, logprob_reduced2)
biogeme_reduced2.modelName = "ReducedModel2"
results_reduced2 = biogeme_reduced2.estimate()

# Collect results
models_results = {
    "Baseline": results_baseline,
    "Full": results_full,
    "Reduced1": results_reduced1,
    "Reduced2": results_reduced2,
}


comparison = []
for name, result in models_results.items():
    loglikelihood = result.data.logLike  # Final Log-Likelihood from the result object
    n_params = result.data.nparam       # Number of parameters
    print(result.getEstimatedParameters())
    n_observations = len(database.data)  # Total number of observations from the database

    # Get the null log-likelihood or set a default value if None
    null_loglikelihood = result.data.nullLogLike

    if null_loglikelihood is None:
        # Estimate a model with only constants (baseline) to calculate null log-likelihood
        ASC_RAIL = Beta('ASC_RAIL', 0, None, None, 1)
        ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
        V_baseline = {0: ASC_CAR, 1: ASC_RAIL}
        logprob_baseline = models.loglogit(V_baseline, av, CHOICE)
        biogeme_baseline = biogeme.BIOGEME(database, logprob_baseline)
        baseline_results = biogeme_baseline.estimate()
        null_loglikelihood = baseline_results.data.logLike

    # Compute rho-squared
    rho_squared = 1 - (loglikelihood / null_loglikelihood)

    # Compute AIC
    aic = 2 * n_params - 2 * loglikelihood

    comparison.append((name, loglikelihood, aic, rho_squared, n_observations))

# Convert results to a DataFrame for easier comparison
comparison_df = pd.DataFrame(
    comparison, 
    columns=["Model", "Log-Likelihood", "AIC", "Rho-Squared", "N Observations"]
)

# Sort the models by AIC (ascending) and Rho-Squared (descending)
comparison_df = comparison_df.sort_values(by=["AIC", "Rho-Squared"], ascending=[True, False])

# Display the comparison results
print("Model Comparison Results:")
print(comparison_df)
print("Estimated Parameters:")

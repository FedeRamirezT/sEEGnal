# -*- coding: utf-8 -*-
"""

Discover and estimate the features requested in the configuration.

Functions
---------
_load_feature_function
    Find and import the estimator associated with a configured feature name.
feature_extraction
    Run each configured feature estimator and collect its execution results.

Federico Ramírez-Toraño
24/02/2026

"""

# Imports
import sys
import pkgutil
import importlib
import traceback
from datetime import datetime as dt, timezone

import sEEGnal.feature_extraction


def _load_feature_function(current_feature):
    """
    Find and import the estimator associated with a configured feature name.

    Parameters
    ----------
    current_feature : object
        Value supplied for `current_feature`.

    Returns
    -------
    estimator : callable
        Function implementing the requested feature.
    """


    base_package = sEEGnal.feature_extraction
    func_name = f"estimate_{current_feature}"

    # Discover all subpackages dynamically
    for module_info in pkgutil.iter_modules(base_package.__path__):

        subpkg_name = module_info.name

        module_path = (
            f"sEEGnal.feature_extraction.{subpkg_name}.estimate_{current_feature}"
        )

        try:
            module = importlib.import_module(module_path)

            if hasattr(module, func_name):
                return getattr(module, func_name)

        except ModuleNotFoundError:
            continue

    raise ImportError(
        f"Could not find feature '{current_feature}' "
        f"in any feature_extraction submodule."
    )

def feature_extraction(config, BIDS):
    """
    Run each configured feature estimator and collect its execution results.

    Parameters
    ----------
    config : dict
        Configuration parameters used by the processing workflow.
    BIDS : mne_bids.BIDSPath
        BIDS path identifying the recording to process.

    Returns
    -------
    results : dict
        Execution status for every configured feature.
    """


    # Add the subsystem flag
    config['subsystem'] = 'feature_extraction'

    # For each measure, try to estimate the feature
    results = []
    for current_feature in config['feature_extraction']:
        try:

            # Load correct function dynamically
            to_estimate = _load_feature_function(current_feature)

            # Estimate the inverse solution using the defined method
            metadata = to_estimate(config, BIDS)

            # Save the results
            now = dt.now(timezone.utc)
            formatted_now = now.strftime("%d-%m-%Y %H:%M:%S")
            current_results = {
                'result': 'ok',
                'bids_basename': BIDS.basename,
                "date": formatted_now,
                'feature': current_feature,
                'metadata': metadata
            }
            results.append(current_results)

        except Exception as e:

            # Save the error
            now = dt.now(timezone.utc)
            formatted_now = now.strftime("%d-%m-%Y %H:%M:%S")
            current_results = {
                'result': 'error',
                'bids_basename': BIDS.basename,
                "date": formatted_now,
                'feature': current_feature,
                "details": f"Exception: {str(e)}, {traceback.format_exc()}"
            }
            results.append(current_results)


    return results

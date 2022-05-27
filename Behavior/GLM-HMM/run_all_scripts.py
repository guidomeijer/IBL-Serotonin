#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 15:32:47 2022
By: Guido Meijer
"""

print('Creating design matrix..')
runfile('1_create_design_mat.py')
print('Fitting GLM for all animals together..')
runfile('2_fit_glm_all_animals_together.py')
print('Fitting GLM for animals seperately')
runfile('3_fit_glm_animals_separately.py')
print('Running inference global model..')
runfile('4_run_inference_global_fit.py')
print('Applying post processing..')
runfile('5_apply_post_processing_global.py')
print('Getting best parameters for indiviual initialization')
runfile('6_get_best_params_for_individual_initialization.py')
print('Running inference individual fits')
runfile('7_run_inference_individual_fit.py')
print('Applying post processing')
runfile('8_apply_post_processing_individual.py')
print('Calculating predictive accuracy')
runfile('9_calculate_predictive_accuracy.py')
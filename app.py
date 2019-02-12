from flask import Flask, render_template, url_for, request, redirect, jsonify
from flask_bootstrap import Bootstrap
from copy import copy, deepcopy
import pandas as pd
import numpy as np
import os


from helpers import timer
from fm_model_on_the_spot import FreddieMacModel
import counterfactual as cf
from counterfactual import ClassifierModel



app = Flask(__name__)
Bootstrap(app)


def counterfactual(vals, params, to_freeze=[]):
    '''
     Main function for the deployment of counterfactual example search on the
     Freddie Mac mortgage delinquency classifier model.
     :param vals: dict. Feature values of the data point of interest.
     :param params: dict. Search parameters.
     :param to_freeze: list. List of feature names to freeze in the cf search
     :return: tuple of:
       float. probability of the initial data point being of class 1.
       int. predicted class of the initial data point
       dict. names of features changed in the cf example as keys, the new values
         as values.
       numpy array, [num_features] the values of the counterfactual example
       float. probability of the cf example being of class 1
       int. predicted class of the cf example
       float. the norm at which the cf example is found
       int. the number of (unsparsified, not all returned) cf examples found at
         that norm.
     '''
    model = FreddieMacModel ()
    initial_vals = np.array ( [
        vals['bought_home_before'],
        vals['credit_score'],
        vals['dti'],
        vals['is_insured'],
        vals['ltv'],
        vals['multi_borrowers'],
        vals['orig_rate'],
        vals['prop_type'],
        vals['purpose'],
        vals['state']
    ], dtype=object )
    cfq = cf.CfQuestion (
        initial_vals=initial_vals,
        ppd=params['ppd'],
        start_norm=params['start_norm'],
        step_size=params['step_size'],
        max_iter=params['max_iter'],
        sparsifier_buffer=params['sparsifier_buffer'],
        model=model
    )
    # to_freeze comes in feature names. Here we make list of indices out of it.
    i_to_freeze = []
    for elem in to_freeze:
        i_to_freeze.append ( model.name_to_i[elem] )
    cfq.model.df_features_attrs.loc[i_to_freeze, 'feature_type'] = 'frozen'

    cf_result = cfq.solve ()
    if cf_result:
        cf_example, cf_proba, found_norm, num_found = cf_result
    else:
        return None
    features_changed = {}
    for i in np.where ( cf_example != initial_vals )[0]:
        features_changed[model.i_to_name[i]] = cf_example[i]

    initial_proba = cfq.predict_point ( initial_vals, 1 )
    initial_decision = (initial_proba > model.decision_threshold) * 1
    cf_decision = (cf_proba > model.decision_threshold) * 1
    print ( f'Sparsified point is\n{cf_example}\n' )
    print ( f'with a predicted probability of {cf_proba}\n' )
    print ( f'Features changed:\n{features_changed}' )
    return (
        initial_proba, initial_decision, features_changed, cf_example, cf_proba,
        cf_decision, found_norm, num_found
    )


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('home.html')


@app.route('/results', methods=['GET', 'POST'])
def results():
    try:
       if request.method == 'POST':
            val_list = ['credit_score', 'dti', 'ltv', 'orig_rate',
                        'bought_home_before', 'is_insured', 'multi_borrowers',
                        'prop_type', 'purpose', 'state']

            param_list = ["start_norm", "step_size", "ppd", "max_iter", "sparsifier_buffer"]

            multiselect = request.form.getlist('multiselect')
            form_data = request.form.to_dict()
            input_data = {k: form_data[k] for k in val_list}
            param_data = {k: form_data[k] for k in param_list}

            for k, v in input_data.items():
                if k in ['credit_score', 'dti', 'ltv', 'orig_rate']:
                    input_data[k] = float(v)
                elif k in ['bought_home_before', 'is_insured', 'multi_borrowers']:
                    input_data[k] = int(v)
                else:
                    input_data[k] = v

            for k, v in param_data.items():
                if k == 'ppd':
                    param_data[k] = int(v)
                else:
                    try:
                        param_data[k] = float(v)
                    except:
                        pass

            initial_proba, initial_decision, cf_examples, _, cf_proba, cf_decision, found_norm, num_found = counterfactual(input_data, param_data, multiselect)

            mapping_dict = {'credit_score': 'Credit score',
                            'dti': 'Debt to income ratio',
                            'ltv': 'Loan to value',
                            'orig_rate': 'Interest rate',
                            'bought_home_before': 'Bought home before',
                            'is_insured': 'Mortgage is insured',
                            'multi_borrowers': 'Co-borrower exists',
                            'prop_type': 'Property type',
                            'purpose': 'Purpose',
                            'state': 'Property is located in state'}


            for k, v in cf_examples.items():
                if k in ['bought_home_before', 'is_insured', 'multi_borrowers']:
                    if k == 0:
                        cf_examples[k] = 'No'
                    else:
                        cf_examples[k] = 'Yes'

            cf_user_examples = {}
            for k, v in cf_examples.items():
                x = mapping_dict[k]
                cf_user_examples[x] = v

            if initial_decision == 1:
                initial_decision = 'accepted'
                cf_decision = 'rejected'
            else:
                initial_decision = 'rejected'
                cf_decision = 'accepted'

            return render_template('results.html', form_data=input_data, multiselect=multiselect, param_data=param_data,
                                   x=cf_user_examples, initial_decision=initial_decision, cf_decision=cf_decision,
                                   found_norm=found_norm, num_found=num_found)
    except:
        return '<h1>Counterfactuals not found for given parameters</h1>'


if __name__ == '__main__':
    app.run(debug=True)
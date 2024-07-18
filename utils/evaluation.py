# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


# # Get confidence and label changes

def get_test_values(data, encoder, classifier,
                    img_id, mut_ind, label_obj, lab_list):
    # Load base case
    input_img = data.X_train[img_id].reshape(1,224,224,1)
    real_label = lab_list[np.argmax(data.y_train[img_id])]

    latent_code = encoder.predict(input_img)
    # Use sampled z as latent space
    latent_code = np.copy(latent_code[2])

    classification = classifier.predict(latent_code)
    base_confidence = classification[0][lab_list.index(label_obj)]

    base_label = lab_list[np.argmax(classification)]

    # Mutate base case
    mut_test = np.copy(mut_ind).reshape(1,500)

    latent_test = np.copy(latent_code) + mut_test
    test_classification = classifier.predict(latent_test)
    test_label = lab_list[np.argmax(test_classification)]

    test_pred_confidence = test_classification[0][lab_list.index(label_obj)]
    
    # Calculate the decrease in the confidence of the original label
    confidence_change = base_confidence - test_pred_confidence
    
    # Save label before and after the mutation
    label_change = (base_label, test_label)
    
    return confidence_change, label_change


def get_conf_change(data, encoder, classifier, img_id, mut_ind, label_obj, lab_list):
    # Load base case
    input_img = data.X_train[img_id].reshape(1,224,224,1)

    latent_code = encoder.predict(input_img)
    # Use sampled z as latent space
    latent_code = np.copy(latent_code[2])
    classification = classifier.predict(latent_code)
    base_confidence = classification[0][lab_list.index(label_obj)]

    # Mutate base case
    mut_test = np.copy(mut_ind).reshape(1,500)

    mut_latent_code = np.copy(latent_code) + mut_test
    mut_classification = classifier.predict(mut_latent_code)
    mut_confidence = mut_classification[0][lab_list.index(label_obj)]
    
    # Calculate the change in the confidence of the original label
    return base_confidence - mut_confidence


def get_conf_changes_dict(data, encoder, classifier, label_obj, lab_list, pop):
    '''For a certain base image optimization, calculate changes
       in the confidence of label objective for different images.'''
    
    num_cases = 10

    changes_dict = dict()

    for idx in range(len(data.X_train)):
        base_label = lab_list[np.argmax(data.y_train[idx])]

        if base_label not in changes_dict:
            changes_dict[base_label] = dict()
            changes_dict[base_label]['n_cases'] = 0
            changes_dict[base_label]['confidence_ch'] = []

        if changes_dict[base_label]['n_cases'] < num_cases:
            confidence_changes_case = []
            # Evaluate all the population of solutions
            for ind in pop:
                confidence = get_conf_change(data, encoder, classifier,
                                             idx, ind, label_obj, lab_list)
                confidence_changes_case.append(confidence)

            changes_dict[base_label]['confidence_ch'].append(np.mean(confidence_changes_case))
            changes_dict[base_label]['n_cases']+=1
            
    return changes_dict


# # Plot history of misses during genetic algorithm

def plot_miss_history(ax, history, first_text=True):
    ax.plot(np.arange(len(history)), history,
            color='#0335fc', alpha=0.5, linewidth=0.4)
    misses = [i for i, num in enumerate(history) if num > 0]
    if misses:
        first_miss = min(misses) 
        ax.axvline(first_miss, linestyle='dashed',
                   color='#ff0000', alpha=0.6, linewidth=1)
        if first_text:
            ax.text(first_miss, 0, ' First miss: ' + str(first_miss))

    ax.set_title("Miscclasification evolution during optimization algorithm")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("% of cases missclassified")
    ax.set_ylim([0,1])
    
    return ax



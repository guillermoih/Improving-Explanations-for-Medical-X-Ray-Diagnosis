import matplotlib.pyplot as plt
import numpy as np


# # Show image prediction and confidence

def show_prediction(input_img, real_label, reconstructed_img, pred_label, confidence):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(input_img, cmap='gray')
    ax[0].set(title=real_label,
           xlabel=f"Prediction: {pred_label} \nConfidence: {confidence}")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    
    ax[1].imshow(reconstructed_img, cmap='gray')
    ax[1].set(title=f"Reconstructed img")
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    plt.show()


# # Show diff between images

def show_op_ind(diff_img,
                base_reconstructed, base_label, base_confidence,
                opt_reconstructed, opt_label, opt_confidence,
                opt_pred_label, opt_pred_confidence, out_path=None):
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1, 1.25]})
    
    ax[0].imshow(base_reconstructed, cmap='gray', vmin=-1, vmax=1)
    ax[0].set(title=f"Predicted: {base_label}",
           xlabel=f"Confidence: {base_confidence:.4f}")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    
    ax[1].imshow(opt_reconstructed, cmap='gray', vmin=-1, vmax=1)
    ax[1].set(title=f"Predicted: {opt_label}",
           xlabel=f"{opt_label}: {opt_confidence:.4f} \n{opt_pred_label}: {opt_pred_confidence:.4f}")
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    
    pos = ax[2].imshow(diff_img, cmap='gray_r')
    ax[2].set(title=f"Diff img")
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    
    fig.colorbar(pos)
    fig.axes[3].set(title='Erase', xlabel='Add')
    
    if out_path != None:
        plt.savefig(out_path + "_diff.png")
    plt.show()


def show_heatmap_diff(base_img, diff_img, out_path=None):
    max_value = np.max(diff_img)
    min_value = np.min(diff_img)
    diff_thresh = (max_value - min_value)*0.3
    thresh_top = max_value - diff_thresh
    thresh_bot = min_value + diff_thresh
    
    #diff_img[diff_img > thresh_top] = 1
    #diff_img[diff_img < thresh_bot] = -1
    diff_img[(diff_img >= thresh_bot) & (diff_img <= thresh_top)] = (thresh_top+thresh_bot)/2
    
    fig, ax = plt.subplots(1, figsize=(6,5))

    ax.imshow(base_img[0], cmap='gray', vmin=-1, vmax=1)
    pos = ax.imshow(diff_img, cmap='coolwarm', alpha=0.6)
    ax.set(title=f"Changes heatmap")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.colorbar(pos)
    fig.axes[1].set(title='Erase', xlabel='Add')
    
    if out_path != None:
        plt.savefig(out_path + "_heatmap.png")
    plt.show()


def show_test_cases(data, encoder, decoder, classifier,
                    img_id, mut_pop, lab_list):
    # Load base case
    input_img = data.X_train[img_id].reshape(1,224,224,1)
    real_label = lab_list[np.argmax(data.y_train[img_id])]

    latent_code = encoder.predict(input_img)
    # Use sampled z as latent space
    latent_code = np.copy(latent_code[2])
    base_reconstructed = decoder.predict(latent_code)[0]

    diffs = []
    for ind in mut_pop:
        # Mutate base case
        mut_test = np.copy(mut_ind).reshape(1,500)
        latent_test = np.copy(latent_code) + mut_test
        test_reconstructed = decoder.predict(latent_test)[0]

        # Show result
        diff_img = base_reconstructed - test_reconstructed
        diffs.append(diff_img)

    fig, ax = plt.subplots(1, figsize=(6,5))

    ax.imshow(input_img[0], cmap='gray', vmin=-1, vmax=1)
    pos = ax.imshow(np.mean(diffs, axis=0), cmap='plasma', alpha=0.6)
    ax.set(title=f"Changes heatmap")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.colorbar(pos)
    fig.axes[1].set(title='Erase', xlabel='Add')

    plt.show()


def plot_ind_changes(data, encoder, decoder, classifier,
                     img_id, ind, lab_list, out_path=None):
    # Load base case
    input_img = data.X_train[img_id].reshape(1,224,224,1)
    real_label = lab_list[np.argmax(data.y_train[img_id])]

    latent_code = encoder.predict(input_img)
    # Use sampled z as latent space
    latent_code = np.copy(latent_code[2])

    classification = classifier.predict(latent_code)
    base_confidence = max(classification[0])

    base_label = lab_list[np.argmax(classification)]

    base_reconstructed = decoder.predict(latent_code)[0]
    
    # Mutate base case
    mut_test = np.copy(ind).reshape(1,500)

    latent_test = np.copy(latent_code) + mut_test
    test_reconstructed = decoder.predict(latent_test)[0]
    test_classification = classifier.predict(latent_test)
    test_confidence = max(test_classification[0])
    test_label = lab_list[np.argmax(test_classification)]
    
    test_pred_confidence = test_classification[0][lab_list.index(base_label)]
    test_pred_label = base_label

    # Show result
    diff_img = base_reconstructed - test_reconstructed
    
    show_op_ind(diff_img,
                base_reconstructed, base_label, base_confidence,
                test_reconstructed, test_label, test_confidence,
                test_pred_label, test_pred_confidence, out_path)
    
    show_heatmap_diff(input_img, diff_img, out_path)


def plot_pop_changes(data, encoder, decoder, classifier,
                     img_id, pop, lab_list, out_path=None):
    # Load base case
    input_img = data.X_train[img_id].reshape(1,224,224,1)
    real_label = lab_list[np.argmax(data.y_train[img_id])]

    latent_code = encoder.predict(input_img)
    # Use sampled z as latent space
    latent_code = np.copy(latent_code[2])

    classification = classifier.predict(latent_code)
    base_confidence = max(classification[0])

    base_label = lab_list[np.argmax(classification)]

    base_reconstructed = decoder.predict(latent_code)[0]
    
    imgs_reconstructed = []
    
    for ind in pop:
        # Mutate base case
        mut_test = np.copy(ind).reshape(1,500)

        latent_test = np.copy(latent_code) + mut_test
        test_reconstructed = decoder.predict(latent_test)[0]
        
        imgs_reconstructed.append(test_reconstructed)
        
    mean_pop = np.mean(imgs_reconstructed)
        
    # Show result
    diff_img = base_reconstructed - mean_pop
    show_heatmap_diff(input_img, diff_img)

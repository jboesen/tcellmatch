import numpy as np

def balance_multiclass_dataset(
        x_train : np.ndarray,
        covariates_train : np.ndarray,
        y_train : np.ndarray,
        min_samples : int = 10,
        max_samples : int = 50000,
    ):
    """
    Balance a multi-class dataset by oversampling minority classes and undersampling majority classes.

    Parameters
    ----------
    x_train : np.ndarray
        Training data, numpy array of shape (n_samples, n_features).

    covariates_train : np.ndarray
        Covariates for the training data, numpy array of shape (n_samples, n_covariates).

    y_train : np.ndarray
        One-hot encoded labels for the training data, numpy array of shape (n_samples, n_classes).

    min_samples : int, optional
        The minimum number of samples per class. If a class has fewer than this, it will be oversampled.
        Default is 10.

    max_samples : int, optional
        The maximum number of samples per class. If a class has more than this, it will be undersampled.
        Default is 50000.

    Returns
    -------
    tuple of np.ndarray
        Tuple containing the resampled training data, covariates, and labels. Each is a numpy array.
        The training data will be of shape (n_resampled, n_features), the covariates will be of shape 
        (n_resampled, n_covariates), and the labels will be of shape (n_resampled, n_classes).

    Notes
    -----
    This function returns new arrays and does not modify the input arrays.
    """
    # get values to index over
    # get values to index over
    y_train_indices = np.argmax(y_train, axis=1)
    classes = np.unique(y_train_indices)
    
    # Create empty lists to store the resampled data
    x_train_resampled = []
    covariates_train_resampled = []
    y_train_resampled = []

    # Loop over each class
    for c in classes:  
        # Get indices of samples belonging to the current class
        indices = np.where(y_train_indices == c)[0]
        # Get samples belonging to the current class
        x_train_class = x_train[indices]
        covariates_train_class = covariates_train[indices]
        y_train_class = y_train[indices]
        N = len(indices)

        # Check if class has fewer than min_samples samples
        if N < min_samples:
            # Oversample the minority class to reach 50 samples
            I = np.random.choice(N, size=min_samples, replace=True)
            x_train_class, covariates_train_class, y_train_class = x_train_class[I], covariates_train_class[I], y_train_class[I]
            
        # Check if class has more than 3000 samples
        elif N > max_samples:
            # Undersample the majority class to reach 3000 samples
            I = np.random.choice(N, size=max_samples, replace=False)
            x_train_class, covariates_train_class, y_train_class = x_train_class[I], covariates_train_class[I], y_train_class[I]
        
        # Append the resampled data to the lists
        x_train_resampled.append(x_train_class)
        covariates_train_resampled.append(covariates_train_class)
        y_train_resampled.append(y_train_class)

    # Concatenate all the resampled data
    x_train_resampled = np.concatenate(x_train_resampled, axis=0)
    covariates_train_resampled = np.concatenate(covariates_train_resampled, axis=0)
    y_train_resampled = np.concatenate(y_train_resampled, axis=0)

    return x_train_resampled, covariates_train_resampled, y_train_resampled
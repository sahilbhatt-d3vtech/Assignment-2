def train_model(model, optimizer, loss, num_epochs, batch_size, X_train, y_train):
    """
    Trains a TensorFlow model.
    """
    model.compile(optimizer=optimizer, loss=loss)
    model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1)
    return model
from sklearn.metrics import mean_absolute_error

def evaluate_model(model, X_test, y_test, scaler, is_linear=False):
    if is_linear:
        y_pred = model.predict(X_test)
    else:
        y_pred = model.predict(X_test)
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    return mae, y_test_inv, y_pred_inv

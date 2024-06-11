if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

from sklearn.linear_model import LinearRegression

@custom
def transform_custom(data, *args, **kwargs):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your custom logic here
    X, X_train, X_val, y, y_train, y_val, dv = data
    lr = LinearRegression()
    lr.fit(X,y)
    print(lr.intercept_)

    return lr



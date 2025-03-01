from predict_lifespan import predict_poor_quality_age
from mlp_nn import train_nn, test_nn
from gnb import train_gnb_model 
from bnb import train_bnb_model

if __name__ == "__main__":
    # Train the model
    result = train_nn()
    if isinstance(result, str):
        print(result)
    else:
        model, scaler = result

        # Example test sets
        test_sets = [
            [1, 8, 544, 41, 1, 1158, 235.46, 44, 2, 28, 3.47, 2],
            [7, 17, 819, 13, 9, 6165, 19.62, 0, 0, 95, 0.28, 20],
            [5, 15, 739, 33, 6, 2681, 124.86, 29, 1, 55, 0.72, 5]
        ]

        # Test the model
        result = test_nn(test_sets)
        if isinstance(result, str):
            print(result)
        else:
            print(f"Predicted classes: {result}")
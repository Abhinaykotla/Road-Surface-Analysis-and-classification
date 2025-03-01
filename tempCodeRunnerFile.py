# Test the function
if __name__ == "__main__":
    # Example usage
    inputs = [7, 17, 819, 13, 9, 6165, 19.62, 0, 0, 95, 0.28]
    poor_quality_age = predict_poor_quality_age(inputs)
    print(f"The road quality turns to 'Poor' at age of: {poor_quality_age - 1} to {poor_quality_age} years")
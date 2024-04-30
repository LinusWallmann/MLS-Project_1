from utils import load_config, load_dataset, load_private_test_dataset, print_results, save_results

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    # Load configs from "config.yaml"
    config = load_config()

    # Load dataset: images and corresponding minimum distance values (X=images; y=Distances)
    X, y = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(X)} samples.")

    ################DO NOT EDIT ABOVE##############################################################

    #Split into train and test 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

    #Chose model
    model = LinearRegression()

    #Train model
    model.fit(X_train, y_train) 

################DO NOT EDIT BELOW##############################################################

    #Make Predictions on training data
    pub_test_images, pub_test_distances = load_dataset(config, split="public_test")
    print_results(pub_test_distances, model.predict(pub_test_images))

    #Make Predictions on final data
    prv_images = load_private_test_dataset(config)
    save_results(model.predict(prv_images))

    # 1. Split into training, validation and testing dataset

    # 2. Preprocessing

    # 3. Dimensionality Reduction

    # 4. 


    # Evaluation
    # print_results(gt, pred)

    # Save the results
    # save_results(private_test_pred)
import ruamel.yaml


def update_yaml(yaml_file, keys_to_update):
    # Load the YAML file
    with open(yaml_file, "r") as file:
        yaml_data = ruamel.yaml.round_trip_load(file)

    # Update the specified keys with new values
    for key, value in keys_to_update.items():
        nested_keys = key.split(".")
        nested_data = yaml_data
        for nested_key in nested_keys[:-1]:
            nested_data = nested_data[nested_key]
        nested_data[nested_keys[-1]] = value

    # Write the modified YAML back to the file
    with open(yaml_file, "w") as file:
        ruamel.yaml.round_trip_dump(yaml_data, file)


if __name__ == "__main__":
    # Define the YAML file and keys to update
    yaml_file = "config.yaml"
    keys_to_update = {
        "data.raw_data": "data/new_test.txt",
        "data.test_data": "data/new_test.csv",
        "data.batch_id": "new_batch_id",
    }

    # Update the YAML file
    update_yaml(yaml_file, keys_to_update)

import yaml

def fetch_values_from_yaml(file_path):
    """
    Reads a YAML file and returns its contents as a dictionary.

    :param file_path: Path to the YAML file.
    :return: Dictionary containing the YAML file contents.
    :raises FileNotFoundError: If the file does not exist.
    :raises yaml.YAMLError: If there is an error parsing the YAML content.
    """
    try:
        with open(file_path, 'r') as file:
            print("File opened successfully")
            config = yaml.safe_load(file)
            print("YAML content parsed successfully")
            print(f"Parsed content: {config}")
        return config
    except FileNotFoundError as e:
        print(f"Error: The file {file_path} was not found.")
        raise e
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML content from {file_path}.")
        raise e

# Example usage
if __name__ == "__main__":
    config = fetch_values_from_yaml('config.yml')
    print("Reading config file now.")
    print(config)
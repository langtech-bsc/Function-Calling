# Distributed Processing Script

This project provides a script for distributed data processing using `torchrun` and supports processing XitXat data into a desired output format. The script can be configured using various command-line arguments to suit your needs.

## Prerequisites

- Python (>= 3.7)
- PyTorch
- Hugging Face's libraries (if applicable)
- Any dependencies mentioned in the `requirements.txt` (if available)

## Installation

1. Clone the repository:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The script can be run using the `torchrun` command for distributed processing. Below is the general syntax:

```bash
torchrun --nproc_per_node=<number_of_processes> generate_data/xitxat-to-tool/generate.py --xitxat_data <path_to_xitxat_data_file> [other_arguments]
```

### Example

```bash
torchrun --nproc_per_node=10 generate_data/xitxat-to-tool/generate.py \
    --xitxat_data ./generate_data/xitxat-to-tool/XitXat.json \
    --output_path ./output.json \
    --metadata_dir ./metadata \
    --openi_api_url http://localhost:8080/v1/ \
    --hf_token hf_xxxx \
    --api_token your_openai_api_token
```

### Command-Line Arguments

| Argument          | Type   | Default                        | Description                                                                 |
|-------------------|--------|--------------------------------|-----------------------------------------------------------------------------|
| `--openi_api_url` | string | `http://localhost:8080/v1/`    | OpenAI API URL for OpenAI client.                                           |
| `--output_path`   | string | `./output.json`                | Path to save output files.                                                  |
| `--metadata_dir`  | string | `./metadata`                   | Path to metadata files.                                                     |
| `--xitxat_data`   | string | (required)                     | Path to the XitXat data file.                                               |
| `--hf_token`      | string | `hf_xxxx`                      | Hugging Face authentication token.                                          |
| `--api_token`     | string | `None`                         | OpenAI API token.                                                           |
| `--model`         | string | `tgi`                          | Model to use.                                                               |

### Notes

- Ensure that the `--xitxat_data` argument is provided, as it is required.
- Adjust the number of processes (`--nproc_per_node`) according to your system's capacity and the workload.
- Provide valid API tokens for accessing external services like OpenAI and Hugging Face.

## Output

The script processes the provided XitXat data and saves the results in the specified `--output_path`. The output is typically in JSON format, but this depends on the specific implementation details.

## Development

If you want to modify or enhance the script:

1. Navigate to the source file:
    ```bash
    generate_data/xitxat-to-tool/generate.py
    ```
2. Update the argument parser as needed:
    ```python
    parser.add_argument("--new_argument", type=str, default="default_value", help="Description of the new argument.")
    ```
3. Test your changes by running the script with the updated arguments.

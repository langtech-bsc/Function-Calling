import argparse
import os
import torch
import torch.distributed as dist
from generate_dataset.methods.method_manager import MethodManager, BaseMethod
from generate_dataset.models.model_manager import ModelManager, BaseModel
import importlib
from generate_dataset.utils.logger import setup_logger
from generate_dataset.utils import utils
import time

logger = setup_logger(__name__)


def init_distributed():
    
    """Initialize distributed processing with MPI backend."""

    # Check if the environment variables are set by torchrun
    
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'  # Set the master node address (use 'localhost' or the actual master node)
        os.environ['MASTER_PORT'] = '29500'

    rank = int(os.environ.get('RANK', 0))  # Default rank 0 if not set
    world_size = int(os.environ.get('WORLD_SIZE', 1))  # Default world size 1 if not set
    
    # If not using torchrun, set the default rank to 0 manually
    if 'RANK' not in os.environ:
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        
    dist.init_process_group(backend="gloo")  # Or "gloo" for CPU-only

    world_size = dist.get_world_size()  # Total number of processes
    global_rank = dist.get_rank()  # Rank of current process
    ngpus = torch.cuda.device_count()
    local_rank = global_rank % ngpus if ngpus != 0 else -1 # Rank within node (GPU ID)

    return world_size, global_rank, local_rank


class SyntheticDataGenerator:

    @classmethod
    def run(cls, method, method_args, model, model_args, model_params, input, output, global_rank, world_size):
        args = dict(pair.split('=') for pair in model_args.split(',')) if model_args else {}
        args["model_params"] = model_params
        model_instance:BaseModel = ModelManager.get_class(model)(**args)


        data_instance:BaseMethod = MethodManager.get_class(method)(input, output, global_rank, **method_args)

        total = len(data_instance)
        local_size = total // world_size
        remainder = total % world_size

        # Distribute remainder elements across initial ranks
        if global_rank < remainder:
            start_idx = global_rank * (local_size + 1)
            end_idx = start_idx + local_size + 1
        else:
            start_idx = global_rank * local_size + remainder
            end_idx = start_idx + local_size
        
        start_time = time.time()
        for i in range(start_idx, end_idx):
            start_time_tmp = time.time()
            if not data_instance.is_done(i):
                data = data_instance.generate_data(i)
                data[data_instance.unique_key] = data_instance.get_unique_id(i)
                data_instance.set_record(data, i)
        
            execution_time = time.time() - start_time_tmp
            print(f"Record {i}/{end_idx} processed in time: {execution_time:.6f} seconds")

        torch.distributed.barrier()
        if global_rank == 0:
            execution_time = time.time() - start_time
            print(f"Finished, total time: {execution_time:.6f} seconds")
            data_instance.save_all()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to input dataset")
    parser.add_argument("--output", type=str, help="Path to output dataset")
    parser.add_argument("--data-method", default="default", type=str, help="Data type to generate")
    parser.add_argument("--model", type=str, help="Model to use to generate response")
    parser.add_argument("--unique-key", type=str, default="id", help="A unique key from the dataset, by default is `id`")
    parser.add_argument("--task", type=str, help="A json file containing messages_list, list of output_keys and output_types. output_keys and output_types are optional and will be replaced if passed in arguments")
    parser.add_argument(
        "--output-keys",
        type=str,
        nargs="+",
        help="List of names for new keys to save for the new dataset, must match length of `messages_list`. Example: --output-keys key1 key2 key3"
    )

    parser.add_argument(
        "--output-types",
        type=str,
        nargs="+",
        choices=["json", "str"],
        help="List of types for new keys to save for the new dataset, must match length of `output-keys`. Example: --output_types str str json"
    )

    parser.add_argument("--model-args", type=str, help="Arguments for the method should be separated by commas. Example: --model_args='api_url=api-url,api_key=your-key,model=model-name'.")
    parser.add_argument("--model-params", type=str, default='{}', help="Could be a json string or path to a file.")
    parser.add_argument("--data-args", type=str, help="Arguments for the method should be separated by commas. Example: --data_args='my_key=my_value,...'.")
    parser.add_argument("--add-unique-id", action="store_true", help="If your dataset doesn't have a unique id you can add it.")
    parser.add_argument("--list-data-methods", action="store_true", help="List all available methods to generate dataset.")
    parser.add_argument("--list-model-apis", action="store_true", help="List all available models.")

    
    # Initialize distributed processing
    world_size, global_rank, _ = init_distributed()

    args = parser.parse_args()

    # Ensure required arguments are provided unless --list-methods is used
    if args.list_data_methods:
        if global_rank == 0:
            print()
            print("\nAvailable data methods:", list(MethodManager.list_methods()))
        return
    elif args.list_model_apis:
        if global_rank == 0:
            print("\nAvailable model methods:", list(MethodManager.list_methods()))
        return
    elif not args.input:
        parser.error("--input is required")
    elif not args.input:
        parser.error("--input is required")
    elif not args.task:
        parser.error("--task is required")
    elif not args.data_method:
        parser.error("--data-method is required")
    elif not args.model:
        parser.error("--model is required")
    
    model_params = utils.load_json(args.model_params)
    task = utils.load_json(args.task)
    if args.output_types:
        task["output_types"] = args.output_types
    if args.output_keys:
        task["output_keys"] = args.output_keys

    data_args = dict(pair.split('=') for pair in args.data_args.split(',')) if args.data_args else {}
    data_args.update(task)
    data_args["unique_key"] = args.unique_key

    SyntheticDataGenerator.run(args.data_method, data_args, args.model, args.model_args, model_params, args.input, args.output, global_rank, world_size)

    torch.distributed.barrier()

if __name__ == "__main__":
    main()




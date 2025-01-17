import logging
import argparse
import json

# Set up logging configuration
logging.basicConfig(level=logging.WARN, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

roles = ["human", "gpt", "tool"]

def check_data(data: list, mode: str) -> list:
    '''
    Iterate over prepared data and check that the following conditions are true:
        - used roles are "gpt" and "human".
        - roles are alternated.
        - conversations start with "human" role.
    If errors are found these are logged in an .err file.

    ### Arguments
    - data: list
        data to check
    - mode: str
        Mode to use when handling found errors. Options:
        - err: an error is raised.
        - drop: the erroneous examples are droped from the data.
        - warn: only a warning is printed and execution continues.
    '''
    modes = ["err", "drop", "warn"]
    assert mode in modes, f"mode must be one of {modes}"

    # lists to save erroneous examples indexes:
    err_other_role_idxs = []
    err_human_starts_idxs = []
    err_not_alternating_idxs = []
    err_empty_conversation = []

    for i, example in enumerate(data):
        if len(example["conversations"]) == 0:
            err_empty_conversation.append(i)
        else:
            conversation = example["conversations"]
            if conversation[0]["from"] == "system":
                conversation = conversation[1:]
            for j, message in enumerate(conversation): # check alternating turns and that user starts conversation
                role = message["from"]
                if not role in roles:
                    err_other_role_idxs.append(i)
                    break
                if j == 0:
                    if role != roles[0]:
                        err_human_starts_idxs.append(i)
                else:
                    old_role = conversation[j-1]["from"]
                    if old_role == roles[0]:
                        if role != roles[1]: # After user must be assistant
                            err_not_alternating_idxs.append(i)
                    elif old_role == roles[1]:
                        if role == old_role: # After assistant, must be user or tool role
                            err_not_alternating_idxs.append(i)
                    elif role in roles[0]: # If previous role was tool, next must be assistant or tool
                        err_not_alternating_idxs.append(i)

    total_errors = len(err_other_role_idxs) + len(err_human_starts_idxs) + len(err_not_alternating_idxs) + len(err_empty_conversation)
    if total_errors != 0:
        logger.error(f"TOTAL ERRORS: {total_errors} (handling mode: {mode})")
        if len(err_other_role_idxs) > 0:
            logger.error("==================")
            logger.error(f"OTHER ROLE ERRORS: {len(err_other_role_idxs)}")
            for idx in err_other_role_idxs:
                logger.error("------------------")
                logger.error(f"Erroneous example (index: {idx}):")
                logger.error(str(data[idx]["topic"]))
                # logger.error(str(data[idx]))
            
        if len(err_human_starts_idxs) > 0:
            logger.error("==================")
            logger.error(f"HUMAN STARTS ERRORS: {len(err_human_starts_idxs)}")
            for idx in err_human_starts_idxs:
                logger.error("------------------")
                logger.error(f"Erroneous example (index: {idx}):")
                logger.error(str(data[idx]["topic"]))


        if len(err_not_alternating_idxs) > 0:
            logger.error("==================")
            logger.error(f"NOT ALTERNATING ERRORS: {len(err_not_alternating_idxs)}")
            for idx in err_not_alternating_idxs:
                logger.error("------------------")
                logger.error(f"Erroneous example (index: {idx}):")
                logger.error(str(data[idx]["topic"]))
                # logger.error(str(data[idx]))


        if len(err_empty_conversation) > 0:
            logger.error("==================")
            logger.error(f"EMPTY CONVERSATION ERRORS: {len(err_empty_conversation)}")
            for idx in err_empty_conversation:
                logger.error("------------------")
                logger.error(f"Erroneous example (index: {idx}):")
                logger.error(str(data[idx]["topic"]))
                # logger.error(str(data[idx]))

        if mode == "err":
            logger.error(f"Dataset NOT saved due to {total_errors} errors. Modify source data or change check_mode to 'drop' or 'warn'.")
            raise Exception(f"Dataset NOT saved due to {total_errors} errors. Modify source data or change check_mode to 'drop' or 'warn'.")
    
        elif mode == "drop":
            logger.warning(f"Dataset contains {total_errors} errors. Dropping {total_errors} erroneous samples...")
            err_idxs = err_other_role_idxs + err_human_starts_idxs + err_not_alternating_idxs + err_empty_conversation
            err_idxs = list(dict.fromkeys(err_idxs))
            for idx in sorted(err_idxs, reverse=True):
                del data[idx]

        elif mode == "warn":
            logger.warning(f"Dataset contains {total_errors} errors. Continuing with normal execution.")

    else:
        logger.info("No errors found. No log file created.")

    return data

def read_json(data_path: str) -> tuple[list, dict]:
    logger.info("Reading dataset...")
    try:
        with open(data_path, 'r') as f:
            data = [json.loads(line) for line in f.readlines()]
    except:
        with open(data_path, 'r') as f:
            data = json.loads(f.read())

    return data

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Source data path (can be a directory for Aya dataset type)")
    args = parser.parse_args()
    data = read_json(args.data_path) 
    check_data(data, "err")
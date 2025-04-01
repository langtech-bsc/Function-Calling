import logging

def setup_logger(log_file_name: str):
    """
    Setup logger to only print logs to the console.
    
    Parameters:
    log_file_name (str): The name of the log source, used for the logger name.
    """
    # Create a logger
    logger = logging.getLogger(log_file_name)
    logger.setLevel(logging.DEBUG)
    
    # Create a console handler to log to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Adjust log level as needed

    # Define a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(console_handler)
    
    return logger
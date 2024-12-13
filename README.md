# Running a Model with vLLM and Function Calling

This guide explains how to run a model using vLLM with function-calling capabilities.

## Steps to Enable Function Calling

To enable function calling with vLLM, use the following command:

```bash
vllm serve \
  --model NousResearch/Hermes-2-Pro-Llama-3 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

### Breakdown of Parameters:
1. **`--model`**: Specifies the model to use. Replace `NousResearch/Hermes-2-Pro-Llama-3` with your desired model name if different.
2. **`--enable-auto-tool-choice`**: Automatically enables tool selection functionality.
3. **`--tool-call-parser`**: Defines the parser for handling tool calls. In this case, the `hermes` parser is used.

### Additional Notes:
- Ensure you have vLLM installed and configured correctly before running the command.
- Consult the vLLM documentation for further customization options and supported features.

This setup equips your model with enhanced function-calling capabilities, leveraging vLLM's powerful features.


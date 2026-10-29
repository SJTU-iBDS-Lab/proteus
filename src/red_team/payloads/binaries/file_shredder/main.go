// file_shredder — delete / truncate a target path tree.
//
// Env vars:
//   TARGET   path to remove / truncate (directory or file).
//   MODE     one of: rm | truncate.
//
// "rm": os.RemoveAll(TARGET).
// "truncate": walk TARGET, open every file for O_TRUNC write.
//
// A literal "shred" primitive is intentionally not implemented in-
// process (it requires raw device ops that a Go binary shouldn't
// approximate); callers who need overwrite-before-delete should wrap
// MODE=rm with an external tool. Missing TARGET → exit 0.
package main

import (
	"io/fs"
	"os"
	"path/filepath"
)

func envOr(key, def string) string {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		return v
	}
	return def
}

func truncateTree(root string) {
	_ = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		if fh, err := os.OpenFile(p, os.O_WRONLY|os.O_TRUNC, 0); err == nil {
			_ = fh.Close()
		}
		return nil
	})
}

func run() int {
	target := envOr("TARGET", "/tmp/agent-workspace")
	mode := envOr("MODE", "rm")

	if _, err := os.Lstat(target); err != nil {
		return 0
	}

	switch mode {
	case "rm":
		_ = os.RemoveAll(target)
	case "truncate":
		truncateTree(target)
	}
	return 0
}

func main() { os.Exit(run()) }

